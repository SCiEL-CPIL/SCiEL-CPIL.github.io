"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with deadline guard and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points in F (vectorized)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        # F[i] is dominated if exists j != i with F[j] <= F[i] in all dims and F[j] < F[i] in at least one
        # all_leq[i,j] = (F[j] <= F[i]).all(axis=1)
        # any_less[i,j] = (F[j] < F[i]).any(axis=1)
        # F[j] <= F[i] for all dims: F[i] - F[j] >= 0 for all dims
        # Shape: (n, n) where [i,j] corresponds to pair (i,j)
        diff = F[None, :, :] - F[:, None, :]  # diff[i,j,d] = F[j,d] - F[i,d]
        all_leq = np.all(diff <= 0, axis=2)    # [i,j] = True if F[j] <= F[i] in all dims
        any_less = np.any(diff < 0, axis=2)    # [i,j] = True if F[j] < F[i] in at least one dim
        # Diagonal: F[i] vs F[i] gives all_leq=True, any_less=False, so not dominated by self
        dominated = np.any(all_leq & any_less, axis=1)
        return np.where(~dominated)[0]

    def _hv_contribution(self, F, ref):
        """Approximate hypervolume contribution using distance to reference."""
        n = len(F)
        if n == 0:
            return np.array([])
        # Product of (ref[j] - F[i,j]) for each objective, clamped at 0
        # Points worse than ref in any objective get 0 contribution
        contribs = np.ones(n)
        for j in range(self.n_obj):
            contribs *= np.maximum(ref[j] - F[:, j], 0.0)
        contribs = contribs + 1e-10
        return contribs

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Initial population: mix of random and structured points
        pop_size = min(30, self.budget // 5)
        X = np.zeros((pop_size, self.dim))
        # First half: uniform random
        X[:pop_size//2] = lo + range_ * rng.random((pop_size//2, self.dim))
        # Second half: LHS-like structured (Latin hypercube approximation)
        if pop_size > pop_size//2:
            n_lhs = pop_size - pop_size//2
            for d in range(self.dim):
                perm = rng.permutation(n_lhs)
                X[pop_size//2:, d] = lo[d] + range_[d] * (perm + 0.5) / n_lhs
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume (from task)
        ref = np.array([3.0, 5.0, 7.0])

        # Reserve budget for local polish (Nelder-Mead)
        polish_budget = max(30, self.budget // 4)
        evo_budget = self.budget - polish_budget

        # Main evolutionary loop
        while used < evo_budget:
            if time.time() > deadline:
                break

            non_dom_idx = self._non_dominated_indices(F)
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(len(F))

            # Select parent: weighted by HV contribution among non-dominated
            contribs = self._hv_contribution(F[non_dom_idx], ref)
            total = contribs.sum()
            if total <= 0:
                parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
            else:
                probs = contribs / total
                parent_idx = non_dom_idx[rng.choice(len(non_dom_idx), p=probs)]

            # Adaptive mutation
            progress = used / max(1, evo_budget)
            if rng.random() < 0.3:
                n_mut = max(1, int(rng.integers(1, max(2, self.dim // 2))))
                mut_dims = rng.choice(self.dim, size=n_mut, replace=False)
                child = X[parent_idx].copy()
                for d in mut_dims:
                    step = 0.15 * (1 - 0.7 * progress) * range_[d]
                    child[d] += rng.standard_normal() * step
            else:
                scale = 0.12 * (1 - 0.7 * progress)
                child = X[parent_idx] + scale * range_ * rng.standard_normal(self.dim)

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replacement: use vectorized dominance
            diff = F[None, :, :] - F[:, None, :]
            all_leq = np.all(diff <= 0, axis=2)
            any_less = np.any(diff < 0, axis=2)
            dominated_mask = np.any(all_leq & any_less, axis=1)

            if np.any(dominated_mask):
                dom_indices = np.where(dominated_mask)[0]
                worst = dom_indices[rng.integers(len(dom_indices))]
            else:
                contribs_all = self._hv_contribution(F, ref)
                worst = int(np.argmin(contribs_all))

            X[worst] = child
            F[worst] = fc

        # Local polish: Nelder-Mead refinement on top non-dominated points
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            non_dom_idx = self._non_dominated_indices(F)
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(len(F))

            # Sort non-dominated by HV contribution, polish top ones
            contribs = self._hv_contribution(F[non_dom_idx], ref)
            order = np.argsort(-contribs)
            n_polish = min(6, len(order))
            top_idx = non_dom_idx[order[:n_polish]]

            # Nelder-Mead on a scalarized objective (Tchebycheff/weighted sum hybrid)
            # We minimize a weighted sum that encourages being close to the ideal point
            ideal = np.min(F, axis=0)  # best value per objective in population

            for idx in top_idx:
                if used >= self.budget or time.time() > deadline:
                    break

                # Define scalar objective: weighted sum of objectives
                # Use weights based on distance to ideal to encourage Pareto improvement
                x0 = X[idx].copy()

                def scalar_obj(x):
                    if used >= self.budget or time.time() > deadline:
                        return 1e10
                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x))
                    used[0] += 1
                    # Weighted sum with adaptive weights emphasizing worst objectives
                    w = 1.0 / (f - ideal + 1e-6)
                    w = w / w.sum()
                    return np.dot(w, f)

                used_arr = [used]
                best_x = x0.copy()
                best_f = F[idx].copy()

                # Simple Nelder-Mead implementation
                n_dim = self.dim
                # Initial simplex
                simplex = np.zeros((n_dim + 1, n_dim))
                simplex[0] = x0
                for k in range(n_dim):
                    step = 0.05 * range_[k]
                    simplex[k + 1] = x0.copy()
                    simplex[k + 1][k] += step
                    simplex[k + 1] = np.clip(simplex[k + 1], lo, hi)

                # Evaluate initial simplex
                simplex_f = np.zeros(n_dim + 1)
                for k in range(n_dim + 1):
                    if used_arr[0] >= self.budget or time.time() > deadline:
                        break
                    simplex_f[k] = scalar_obj(simplex[k])

                # Nelder-Mead iterations
                max_nm_iter = max(5, remaining // (n_dim + 1) // 2)
                alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

                for _ in range(max_nm_iter):
                    if used_arr[0] >= self.budget or time.time() > deadline:
                        break

                    # Sort simplex by objective value
                    order_nm = np.argsort(simplex_f)
                    simplex = simplex[order_nm]
                    simplex_f = simplex_f[order_nm]

                    # Centroid of all but worst
                    centroid = simplex[:n_dim].mean(axis=0)

                    # Reflection
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, lo, hi)
                    f_r = scalar_obj(x_r)

                    if f_r < simplex_f[0]:
                        # Expansion
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, lo, hi)
                        f_e = scalar_obj(x_e)
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    elif f_r < simplex_f[-2]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    else:
                        # Contraction
                        if f_r < simplex_f[-1]:
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            x_c = centroid - rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, lo, hi)
                        f_c = scalar_obj(x_c)
                        if f_c < min(f_r, simplex_f[-1]):
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for k in range(1, n_dim + 1):
                                simplex[k] = simplex[0] + sigma * (simplex[k] - simplex[0])
                                simplex[k] = np.clip(simplex[k], lo, hi)
                                simplex_f[k] = scalar_obj(simplex[k])

                    used = used_arr[0]

                # Update best if NM found improvement
                best_f_nm = np.array(problem(np.clip(simplex[0], lo, hi)))
                used += 1
                if np.all(best_f_nm <= best_f) or np.any(best_f_nm < best_f):
                    X[idx] = np.clip(simplex[0], lo, hi)
                    F[idx] = best_f_nm

        return X
# EVOLVE-BLOCK-END
