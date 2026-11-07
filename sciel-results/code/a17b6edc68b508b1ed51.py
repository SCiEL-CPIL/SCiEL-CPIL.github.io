"""Multi-objective optimizer for RE21 with Halton init and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _halton(n, dims, base=None):
    """Generate n Halton sequence points in [0,1)^dims."""
    if base is None:
        # Use first 'dims' primes
        primes = []
        candidate = 2
        while len(primes) < dims:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        base = primes

    points = np.zeros((n, dims))
    for i in range(1, n + 1):
        for d in range(dims):
            f = 1.0
            result = 0.0
            j = i
            while j > 0:
                f /= base[d]
                result += f * (j % base[d])
                j //= base[d]
            points[i - 1, d] = result
    return points


class Optimizer:
    """Multi-objective optimizer using Halton initialization + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Phase 1: Halton sequence for diverse initial sampling
        n_init = min(40, self.budget // 5)
        halton_pts = _halton(n_init, self.dim)
        X_init = lo + (hi - lo) * halton_pts
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Track all evaluated points
        all_X = [list(X_init)]
        all_F = [list(F_init)]

        # Reference point for HV calculation
        ref = np.array([3000.0, 0.06])

        # Phase 2: Local refinement around best points
        # Sort by sum of objectives to find most promising
        sums = np.sum(F_init, axis=1)
        # Pick top candidates for local refinement
        n_refine = min(8, self.budget - used)

        # Get indices of best points (lowest sum)
        top_indices = np.argsort(sums)[:n_refine]

        for idx in top_indices:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break

            # Nelder-Mead-like local search around this point
            x0 = X_init[idx].copy()
            f0 = F_init[idx].copy()

            # Simple local search: try small perturbations
            best_x = x0.copy()
            best_f = f0.copy()

            # Adaptive step size
            step = 0.05 * (hi - lo)
            n_local = min(15, self.budget - used)

            for _ in range(n_local):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Random perturbation
                direction = rng.standard_normal(self.dim)
                direction /= (np.linalg.norm(direction) + 1e-12)
                x_trial = best_x + step * direction
                x_trial = np.clip(x_trial, lo, hi)

                f_trial = np.array(problem(x_trial))
                used += 1

                # Check if better (both objectives or one significantly better)
                if np.all(f_trial <= best_f) and np.any(f_trial < best_f):
                    best_x = x_trial
                    best_f = f_trial
                    step *= 1.2  # expand
                elif np.sum(f_trial) < np.sum(best_f):
                    # Partial improvement
                    if rng.random() < 0.3:
                        best_x = x_trial
                        best_f = f_trial
                    step *= 0.8  # contract
                else:
                    step *= 0.7

                all_X.append(best_x.copy())
                all_F.append(best_f.copy())

            # Record the best found in this local search
            all_X.append(best_x.copy())
            all_F.append(best_f.copy())

        # Phase 3: If budget remains, do additional random sampling in promising regions
        if used < self.budget:
            # Find the best point overall
            all_F_arr = np.array(all_F)
            all_X_arr = np.array(all_X)
            valid_mask = np.all(all_F_arr < ref, axis=1)
            if np.any(valid_mask):
                valid_sums = np.sum(all_F_arr[valid_mask], axis=1)
                best_idx = np.argmin(valid_sums)
                center = all_X_arr[valid_mask][best_idx]
            else:
                center = all_X_arr[np.argmin(np.sum(all_F_arr, axis=1))]

            # Sample around the best region
            while used < self.budget:
                if time.time() > deadline:
                    break
                spread = 0.1 * (hi - lo)
                x_trial = center + spread * rng.standard_normal(self.dim)
                x_trial = np.clip(x_trial, lo, hi)
                f_trial = np.array(problem(x_trial))
                used += 1
                all_X.append(x_trial)
                all_F.append(f_trial)

        return all_X[0]  # Return value is ignored by evaluator
# EVOLVE-BLOCK-END
