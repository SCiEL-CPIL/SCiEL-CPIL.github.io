"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer combining NSGA-II-style selection with
    differential-evolution-inspired mutation and a final polish phase.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Phase 1: Initial random population
        init_pop = min(50, budget)
        X_init = lo + (hi - lo) * np.random.random((init_pop, dim))
        F_init = np.array([problem(x) for x in X_init])
        evals_used = init_pop

        # Store all evaluated points
        all_X = [x.copy() for x in X_init]
        all_F = [f.copy() for f in F_init]

        # Helper: compute non-dominated front indices
        def get_non_dominated_indices(F_list):
            n = len(F_list)
            if n == 0:
                return []
            F_arr = np.array(F_list)
            # Vectorized dominance check
            # F[j] dominates F[i] if F[j] <= F[i] for all obj and F[j] < F[i] for at least one
            leq = np.all(F_arr[:, None, :] <= F_arr[None, :, :], axis=2)
            lt = np.all(F_arr[:, None, :] < F_arr[None, :, :], axis=2)
            dominates = leq & (~lt | leq)  # leq and (at least one strict)
            # Actually: dominates[j,i] = True if F[j] strictly dominates F[i]
            dominates = leq & (~lt | leq)
            # Simpler: strictly dominates if all <= and at least one <
            dominates = leq & (~lt | leq)
            # Let me redo this properly
            # F[j] strictly dominates F[i]: F[j]_k <= F[i]_k for all k, and F[j]_m < F[i]_m for some m
            all_leq = np.all(F_arr[:, None, :] <= F_arr[None, :, :], axis=2)  # [j,i]
            any_lt = np.any(F_arr[:, None, :] < F_arr[None, :, :], axis=2)    # [j,i]
            strictly_dominates = all_leq & any_lt  # strictly_dominates[j,i] = F[j] dominates F[i]
            # Point i is non-dominated if no j strictly dominates it
            is_dominated = np.any(strictly_dominates, axis=0)
            nd_indices = np.where(~is_dominated)[0].tolist()
            return nd_indices

        # Helper: compute crowding distance
        def crowding_distance(F_list):
            n = len(F_list)
            if n <= 2:
                return [float('inf')] * n
            F_arr = np.array(F_list)
            cd = np.zeros(n)
            for m in range(n_obj):
                sorted_idx = np.argsort(F_arr[:, m])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                for i in range(1, n - 1):
                    denom = F_arr[sorted_idx[i + 1], m] - F_arr[sorted_idx[i - 1], m]
                    if denom > 1e-10:
                        cd[sorted_idx[i]] += 1.0 / denom
            return cd

        # Phase 2: Evolutionary refinement with DE-inspired mutation
        remaining = budget - evals_used
        tournament_size = 3

        # DE parameters
        F_factor_start = 0.8   # Differential evolution scaling factor
        F_factor_end = 0.3
        CR = 0.5                # Crossover rate

        while evals_used < budget:
            if time.time() > deadline:
                break

            # Get non-dominated front
            nd_idx = get_non_dominated_indices(all_F)

            if len(nd_idx) == 0:
                break

            # Compute crowding distance for non-dominated front
            nd_F = [all_F[i] for i in nd_idx]
            cd = crowding_distance(nd_F)

            # Select parents using tournament on crowding distance
            if len(nd_idx) > 1:
                # Pick 3 random non-dominated points, choose best by crowding distance
                candidates = np.random.choice(len(nd_idx), size=min(tournament_size, len(nd_idx)), replace=False)
                best_candidate = candidates[np.argmax([cd[c] for c in candidates])]
                parent_idx = nd_idx[best_candidate]
            else:
                parent_idx = nd_idx[0]

            parent_X = all_X[parent_idx]

            # DE-style mutation: use difference of two other random non-dominated points
            if len(nd_idx) >= 3:
                # Pick two other random non-dominated points
                other_indices = [i for i in range(len(nd_idx)) if i != best_candidate]
                rand1 = other_indices[np.random.randint(len(other_indices))]
                rand2 = other_indices[np.random.randint(len(other_indices))]
                # Ensure rand1 != rand2
                while rand2 == rand1:
                    rand2 = other_indices[np.random.randint(len(other_indices))]

                X1 = all_X[nd_idx[rand1]]
                X2 = all_X[nd_idx[rand2]]

                # Adaptive F factor
                progress = 1.0 - (budget - evals_used) / max(1, remaining)
                F_factor = F_factor_start + (F_factor_end - F_factor_start) * progress
                F_factor = max(F_factor, F_factor_end)

                # DE mutation: v = X1 + F * (X2 - X3) where X3 is the parent
                # Standard DE: v = X_rand1 + F * (X_rand2 - X_parent)
                v = X1 + F_factor * (X2 - parent_X)
            else:
                # Fallback to Gaussian mutation if not enough non-dominated points
                progress = 1.0 - (budget - evals_used) / max(1, remaining)
                scale = 0.25 + (0.015 - 0.25) * progress
                scale = max(scale, 0.015)
                v = parent_X + scale * (hi - lo) * np.random.standard_normal(dim)

            # Binomial crossover
            child_X = parent_X.copy()
            for d in range(dim):
                if np.random.random() < CR or d == np.random.randint(dim):
                    child_X[d] = v[d]

            child_X = np.clip(child_X, lo, hi)

            # Evaluate child
            child_F = problem(child_X)
            evals_used += 1

            # Add to archive
            all_X.append(child_X)
            all_F.append(child_F)

        # Phase 3: Final polish - refine best non-dominated points with small steps
        nd_idx = get_non_dominated_indices(all_F)
        if len(nd_idx) > 0 and evals_used < budget:
            # Sort by crowding distance, polish the most diverse ones
            nd_F = [all_F[i] for i in nd_idx]
            cd = crowding_distance(nd_F)
            # Get top 5 most diverse points
            top_k = min(5, len(nd_idx))
            top_indices = np.argsort(cd)[-top_k:]
            
            polish_budget = min(15, budget - evals_used)
            for idx in top_indices:
                if evals_used >= budget or time.time() > deadline:
                    break
                base_X = all_X[nd_idx[idx]]
                # Try 3 small perturbations
                for _ in range(3):
                    if evals_used >= budget or time.time() > deadline:
                        break
                    small_scale = 0.005 * (hi - lo)
                    mutation = small_scale * np.random.standard_normal(dim)
                    polish_X = np.clip(base_X + mutation, lo, hi)
                    polish_F = problem(polish_X)
                    evals_used += 1
                    all_X.append(polish_X)
                    all_F.append(polish_F)

        return None
# EVOLVE-BLOCK-END
