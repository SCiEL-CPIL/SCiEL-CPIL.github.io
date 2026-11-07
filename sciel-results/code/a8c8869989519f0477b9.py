"""Multi-objective optimizer on WFG7: Tchebycheff decomposition with Nelder-Mead local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective search using Tchebycheff decomposition with Nelder-Mead local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 330s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            f_arr = np.array(f, dtype=np.float64)
            used += 1
            all_X.append(x.copy())
            all_F.append(f_arr)
            return f_arr

        # Phase 1: Generate Tchebycheff weight vectors (30 uniformly spaced on simplex)
        n_weights = 30
        weights = []
        for i in range(n_weights + 1):
            for j in range(n_weights - i + 1):
                k = n_weights - i - j
                if i + j + k == n_weights:
                    w = np.array([i, j, k], dtype=np.float64) / n_weights
                    if np.all(w > 0.02):
                        weights.append(w)
        if len(weights) < 10:
            weights = [np.array([1/3, 1/3, 1/3])]
            for _ in range(9):
                w = rng.dirichlet(np.ones(3))
                weights.append(w)
        weights = np.array(weights[:30])

        # Phase 2: Initial diverse evaluations
        n_init = min(40, budget - len(weights) * 10)
        for i in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = mid + 0.2 * span * rng.standard_normal(dim)
            x = np.clip(x, lo, hi)
            evaluate(x)

        # Phase 3: Tchebycheff-based Nelder-Mead local search for each weight
        # Use the best point for each weight as initial simplex center
        F_arr = np.array(all_F) if all_F else None
        
        # Compute Tchebycheff values for each weight to find best starting points
        weight_best_idx = []
        if F_arr is not None and len(F_arr) > 0:
            best_f = np.min(F_arr, axis=0)
            for w in weights:
                tcheb_vals = np.max(w * (F_arr - best_f), axis=1)
                best_local_idx = np.argmin(tcheb_vals)
                weight_best_idx.append(best_local_idx)
        else:
            weight_best_idx = [0] * len(weights)

        # Allocate remaining budget across weights
        remaining_budget = budget - used
        evals_per_weight = max(5, remaining_budget // len(weights))
        evals_per_weight = min(evals_per_weight, 15)

        # Sort weights by balance (prefer uniform-like weights for better HV coverage)
        weight_scores = [np.sum(np.abs(w - 1.0/n_obj)) for w in weights]
        search_order = np.argsort(weight_scores)

        for w_idx in search_order:
            if used >= budget or time.time() > deadline:
                break
            w = weights[w_idx]
            
            # Get starting point for this weight
            if F_arr is not None and len(F_arr) > 0:
                start_idx = weight_best_idx[w_idx]
                center_x = all_X[start_idx].copy()
            else:
                center_x = mid.copy()

            # Nelder-Mead simplex for Tchebycheff scalarization
            best_f_global = np.min(F_arr, axis=0) if F_arr is not None and len(F_arr) > 0 else np.zeros(n_obj)
            
            # Initialize simplex
            simplex = [center_x.copy()]
            center_f = all_F[weight_best_idx[w_idx]].copy() if all_F else None
            simplex_f = [self._tchebycheff_from_f(center_f, w, best_f_global) if center_f is not None else 0.0]
            
            # Add simplex vertices
            step = 0.05 * span
            for d in range(dim):
                vertex = center_x.copy()
                vertex[d] += step[d] if vertex[d] + step[d] <= hi[d] else -step[d]
                vertex = np.clip(vertex, lo, hi)
                f_val = evaluate(vertex)
                if f_val is None:
                    break
                simplex.append(vertex)
                simplex_f.append(self._tchebycheff_from_f(f_val, w, best_f_global))
            
            if len(simplex) < 2:
                continue

            # Nelder-Mead iterations
            alpha, gamma, rho = 1.0, 2.0, 0.5
            max_iter = evals_per_weight - len(simplex)
            
            for iteration in range(max(0, max_iter)):
                if used >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by Tchebycheff value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                f_r_eval = evaluate(xr)
                if f_r_eval is None:
                    break
                fr = self._tchebycheff_from_f(f_r_eval, w, best_f_global)
                
                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, lo, hi)
                    f_e_eval = evaluate(xe)
                    if f_e_eval is None:
                        break
                    fe = self._tchebycheff_from_f(f_e_eval, w, best_f_global)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lo, hi)
                    f_c_eval = evaluate(xc)
                    if f_c_eval is None:
                        break
                    fc = self._tchebycheff_from_f(f_c_eval, w, best_f_global)
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        shrink_ok = True
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            f_s_eval = evaluate(simplex[i])
                            if f_s_eval is None:
                                shrink_ok = False
                                break
                            simplex_f[i] = self._tchebycheff_from_f(f_s_eval, w, best_f_global)
                        if not shrink_ok:
                            break

            # Update best_f_global with new points
            if all_F:
                F_arr_new = np.array(all_F)
                best_f_global = np.min(F_arr_new, axis=0)

        # Phase 4: Final polish of best non-dominated points
        if used < budget and all_F:
            F_arr = np.array(all_F)
            non_dom_idx = self._get_non_dominated_indices(F_arr)
            
            if len(non_dom_idx) > 0:
                # Select diverse non-dominated points
                selected = []
                uniform_w = np.ones(n_obj) / n_obj
                best_f = np.min(F_arr, axis=0)
                tcheb_vals = np.max(uniform_w * (F_arr - best_f), axis=1)
                best_start = np.argmin(tcheb_vals)
                if best_start in non_dom_idx:
                    selected.append(best_start)
                
                # Add up to 4 more diverse points
                remaining = [i for i in non_dom_idx if i not in selected]
                for _ in range(4):
                    if not remaining or used >= budget or time.time() > deadline:
                        break
                    best_idx = -1
                    best_min_dist = -1
                    for idx in remaining:
                        f_point = F_arr[idx]
                        min_dist = np.min([np.linalg.norm(f_point - F_arr[s]) for s in selected])
                        if min_dist > best_min_dist:
                            best_min_dist = min_dist
                            best_idx = idx
                    if best_idx == -1:
                        break
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                
                # Polish each selected point with small-step Nelder-Mead
                remaining_steps = budget - used
                steps_per_point = max(2, remaining_steps // max(1, len(selected)))
                steps_per_point = min(steps_per_point, 6)
                
                for idx in selected:
                    if used >= budget or time.time() > deadline:
                        break
                    center_x = all_X[idx].copy()
                    center_f = all_F[idx].copy()
                    step_size = 0.02 * span
                    
                    for _ in range(steps_per_point):
                        if used >= budget or time.time() > deadline:
                            break
                        perturbation = step_size * rng.standard_normal(dim)
                        trial_x = np.clip(center_x + perturbation, lo, hi)
                        trial_f = evaluate(trial_x)
                        if trial_f is None:
                            break
                        if np.all(trial_f <= center_f) and np.any(trial_f < center_f):
                            center_x = trial_x
                            center_f = trial_f
                            step_size *= 0.8
                        else:
                            step_size *= 0.7

        if all_F:
            return np.array(all_X)
        return np.array([])

    def _tchebycheff_from_f(self, f, w, ref_f):
        """Compute Tchebycheff scalarization value from objective vector f."""
        return np.max(w * (f - ref_f))

    def _get_non_dominated_indices(self, F):
        """Return indices of non-dominated points using vectorized comparison."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        if n == 1:
            return np.array([0], dtype=int)

        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            leq = np.all(F <= F[i], axis=1)
            strict = np.any(F < F[i], axis=1)
            leq[i] = False
            if np.any(leq & strict):
                is_dominated[i] = True
            else:
                for j in range(i + 1, n):
                    if not is_dominated[j]:
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            is_dominated[j] = True

        return np.where(~is_dominated)[0]
# EVOLVE-BLOCK-END
