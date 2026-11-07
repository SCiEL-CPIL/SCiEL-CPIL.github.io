"""Optimized search for BBOB f15 (rotated Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with Rastrigin-aware local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # safety margin

        # Global best tracking
        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            call_count += 1
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x_clipped))
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return call_count < budget

        def local_refine(x, fx, max_rounds=5):
            """Coordinate descent with Rastrigin-periodic candidates."""
            nonlocal call_count, best_x, best_f
            if not budget_ok() or not time_ok():
                return x, fx

            for _ in range(max_rounds):
                if not budget_ok() or not time_ok():
                    break
                improved = False
                for axis in range(dim):
                    if not budget_ok() or not time_ok():
                        break
                    x0 = x[axis]
                    # Generate candidates: current, nearby, and Rastrigin-aligned
                    candidates = [x0]
                    # Small perturbations
                    for d in [-0.3, -0.15, -0.05, 0.05, 0.15, 0.3]:
                        candidates.append(x0 + d)
                    # Rastrigin: minima at multiples of 2*pi in the unrotated frame.
                    # Since rotation is unknown, try snapping to nearest 2*pi multiple
                    # and its neighbors.
                    k = np.round(x0 / (2.0 * np.pi))
                    for kk in [k - 1, k, k + 1]:
                        candidates.append(kk * 2.0 * np.pi)
                    # Also try halfway points (maxima of cos, so avoid)
                    for kk in [k, k + 1]:
                        candidates.append(kk * 2.0 * np.pi + np.pi)

                    best_cand = x0
                    best_cand_f = fx
                    for c in candidates:
                        if not budget_ok():
                            break
                        c = float(np.clip(c, -5.0, 5.0))
                        if abs(c - x0) < 1e-12:
                            continue
                        y = x.copy()
                        y[axis] = c
                        fy = eval_func(y)
                        if fy < best_cand_f:
                            best_cand_f = fy
                            best_cand = c
                    if best_cand != x0:
                        x[axis] = best_cand
                        fx = best_cand_f
                        improved = True
                if not improved:
                    break
            return x, fx

        # --- CMA-ES parameters ---
        pop_size = 10
        # Initial mean: try a few random points to pick a good starting point
        init_candidates = []
        init_fvals = []
        for _ in range(5):
            if not budget_ok() or not time_ok():
                break
            x = rng.uniform(-4.0, 4.0, dim)
            f = eval_func(x)
            init_candidates.append(x)
            init_fvals.append(f)

        if init_fvals:
            init_idx = np.argmin(init_fvals)
            mean = init_candidates[init_idx].copy()
            # Initialize covariance as identity * sigma^2
            sigma = 2.0
        else:
            mean = np.zeros(dim)
            sigma = 2.0

        # CMA-ES state
        C = np.eye(dim)
        C_sqrt = np.eye(dim)  # Cholesky factor
        ps = np.zeros(dim)  # evolution path for sigma
        pc = np.zeros(dim)  # evolution path for C
        cs = (pop_size + 2) / (dim + 4)
        ds = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + cs
        cc = (4 + dim / 3) / (dim + 4 + 2 * pop_size / dim)
        cmu = 2 / ((dim + 1.3) ** 2 + pop_size)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        generation = 0
        no_improve = 0
        best_f_prev = best_f

        while budget_ok() and time_ok():
            generation += 1

            # Generate population
            offspring = []
            offspring_f = []
            for i in range(pop_size):
                if not budget_ok() or not time_ok():
                    break
                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                x = mean + sigma * (C_sqrt @ z)
                x = np.clip(x, -5.0, 5.0)
                fx = eval_func(x)
                offspring.append(x)
                offspring.append_f = fx  # temporary

            if not offspring:
                break

            offspring = np.array(offspring)
            offspring_f = np.array([eval_func(x) for x in []])  # already evaluated
            # Actually we need to restructure: evaluate during generation
            # Let me redo this properly

            # Redo: generate and evaluate
            offspring = []
            offspring_f = []
            for i in range(pop_size):
                if not budget_ok() or not time_ok():
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * (C_sqrt @ z)
                x = np.clip(x, -5.0, 5.0)
                fx = eval_func(x)
                offspring.append(x)
                offspring_f.append(fx)

            if len(offspring) < pop_size:
                # Pad with repeats if we ran out of budget/time
                while len(offspring) < pop_size:
                    offspring.append(offspring[-1].copy())
                    offspring_f.append(offspring_f[-1])

            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)

            # Sort by fitness
            idx = np.argsort(offspring_f)
            offspring = offspring[idx]
            offspring_f = offspring_f[idx]

            # Update mean (weighted recombination)
            w = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
            w /= w.sum()
            mu_eff = 1 / np.sum(w ** 2)

            new_mean = np.sum(w[:, None] * offspring, axis=0)

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_sqrt.T @ (new_mean - mean) / sigma)
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) < (1.4 + 2 / (dim + 1)) * chiN:
                pc = (1 - cc) * pc + cc * (new_mean - mean) / sigma

            # Update covariance
            delta_h = (1 + (1 - cc) * (np.dot(ps, ps) / dim - 1) * (1 - cs)) * (1 - cc)
            rank1 = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(pop_size):
                diff = (new_mean - offspring[i]) / sigma
                rank_mu += w[i] * np.outer(diff, diff)
            C = (1 + cs * ds + 2 * cs * cmu - cc) * C + cs * delta_h * rank1 + cmu * rank_mu

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 0.01, 50.0)

            # Update mean
            mean = new_mean

            # Update Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: use eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

            # Check for improvement
            if best_f < best_f_prev - 1e-8:
                no_improve = 0
                best_f_prev = best_f
            else:
                no_improve += 1

            # Periodic local refinement on the best point
            if generation % 5 == 0 and budget_ok() and time_ok():
                x_ref, f_ref = local_refine(best_x.copy(), best_f, max_rounds=3)
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()

            # If stuck for a while, inject diversity
            if no_improve > 20 and generation % 10 == 0:
                # Perturb the mean
                mean += sigma * rng.standard_normal(dim)
                mean = np.clip(mean, -5.0, 5.0)
                no_improve = 0
                sigma = max(sigma, 1.0)

            # If sigma is very small, do a restart-like diversification
            if sigma < 0.05 and generation % 20 == 0:
                # Try a few new random points
                for _ in range(3):
                    if not budget_ok() or not time_ok():
                        break
                    x_new = rng.uniform(-4.0, 4.0, dim)
                    f_new = eval_func(x_new)
                    if f_new < best_f:
                        mean = x_new.copy()
                        sigma = 1.5
                        break
                no_improve = 0

        # Final intensive refinement
        if best_x is not None and budget_ok() and time_ok():
            for _ in range(5):
                if not budget_ok() or not time_ok():
                    break
                x_ref, f_ref = local_refine(best_x.copy(), best_f, max_rounds=5)
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()
                else:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
