"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using incremental non-dominated front maintenance,
    differential-evolution-style mutation, and final local polish.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Phase 1: Initial random population
        init_pop = min(40, budget)
        X_init = lo + (hi - lo) * np.random.random((init_pop, dim))
        F_init = np.array([problem(x) for x in X_init])
        evals_used = init_pop

        # Store all evaluated points
        all_X = [x.copy() for x in X_init]
        all_F = [f.copy() for f in F_init]

        # Maintain non-dominated front incrementally
        # front_idx: list of indices into all_F that form the current non-dominated front
        front_idx = list(range(init_pop))
        
        # Build initial front properly
        def is_dominated_by_any(point_F, front_F_arr):
            """Check if point_F is dominated by any point in front_F_arr."""
            if len(front_F_arr) == 0:
                return False
            # point_F is dominated if exists f in front such that f <= point_F and f < point_F
            leq = np.all(front_F_arr <= point_F, axis=1)
            lt = np.any(front_F_arr < point_F, axis=1)
            return np.any(leq & lt)

        def remove_dominated_from_front(point_F, front_indices, all_F_list):
            """Remove points from front that are dominated by point_F. Returns new front indices."""
            new_front = []
            for idx in front_indices:
                f = all_F_list[idx]
                # Check if f is dominated by point_F
                if np.all(point_F <= f) and np.any(point_F < f):
                    continue  # f is dominated, remove
                new_front.append(idx)
            return new_front

        # Build initial non-dominated front
        for i in range(init_pop):
            f_i = F_init[i]
            # Check if f_i is dominated by any existing front point
            dominated = False
            for j in front_idx:
                f_j = F_init[j]
                if np.all(f_j <= f_i) and np.any(f_j < f_i):
                    dominated = True
                    break
            if not dominated:
                # Remove any front points dominated by f_i
                front_idx = remove_dominated_from_front(f_i, front_idx, F_init)
                front_idx.append(i)

        # Helper: compute crowding distance for a set of points (indexed by front_idx)
        def crowding_distance(front_indices, F_list):
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            F_arr = np.array([F_list[i] for i in front_indices])
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

        # Phase 2: Main evolutionary loop with differential-evolution-style mutation
        remaining = budget - evals_used
        mutation_scale_start = 0.15
        mutation_scale_end = 0.01
        tournament_size = 3
        de_probability = 0.7  # Probability of using DE-style mutation
        random_inject_every = 12

        while evals_used < budget:
            if time.time() > deadline:
                break

            if len(front_idx) == 0:
                # Fallback: use all points
                front_idx = list(range(len(all_F)))
                continue

            # Compute crowding distance for the non-dominated front
            cd = crowding_distance(front_idx, all_F)

            # Tournament selection: pick tournament_size random non-dominated points,
            # choose the one with the largest crowding distance
            if len(front_idx) > 1:
                candidates = np.random.choice(len(front_idx), size=min(tournament_size, len(front_idx)), replace=False)
                best_candidate = candidates[np.argmax([cd[c] for c in candidates])]
                parent_idx = front_idx[best_candidate]
            else:
                parent_idx = front_idx[0]

            parent_X = all_X[parent_idx]

            # Adaptive mutation step size
            progress = 1.0 - (budget - evals_used) / max(1, remaining)
            scale = mutation_scale_start + (mutation_scale_end - mutation_scale_start) * progress
            scale = max(scale, mutation_scale_end)

            # Decide mutation type
            r = np.random.random()
            if r < de_probability and len(front_idx) >= 3:
                # Differential evolution style: parent + scale * (x1 - x2)
                # Pick two other random front members
                other_indices = [i for i in range(len(front_idx)) if front_idx[i] != parent_idx]
                if len(other_indices) >= 2:
                    idx1 = other_indices[np.random.randint(len(other_indices))]
                    idx2 = other_indices[np.random.randint(len(other_indices) - 1)]
                    if idx2 >= idx1:
                        idx2 -= 1
                    x1 = all_X[front_idx[idx1]]
                    x2 = all_X[front_idx[idx2]]
                    mutation = scale * (x1 - x2)
                    child_X = np.clip(parent_X + mutation, lo, hi)
                else:
                    # Fallback to Gaussian
                    mutation = scale * (hi - lo) * np.random.standard_normal(dim)
                    child_X = np.clip(parent_X + mutation, lo, hi)
            elif r < de_probability + 0.15:
                # Global jump
                mutation = 0.3 * (hi - lo) * np.random.standard_normal(dim)
                child_X = np.clip(parent_X + mutation, lo, hi)
            elif evals_used % random_inject_every == 0 and np.random.random() < 0.3:
                # Random injection
                child_X = lo + (hi - lo) * np.random.random(dim)
            else:
                # Standard Gaussian mutation
                mutation = scale * (hi - lo) * np.random.standard_normal(dim)
                child_X = np.clip(parent_X + mutation, lo, hi)

            # Evaluate child
            child_F = problem(child_X)
            evals_used += 1

            # Add to archive
            all_X.append(child_X)
            all_F.append(child_F)

            # Update non-dominated front incrementally
            new_idx = len(all_F) - 1
            child_F_arr = all_F[new_idx]
            
            # Check if child is dominated by any front point
            dominated = False
            for fidx in front_idx:
                f = all_F[fidx]
                if np.all(f <= child_F_arr) and np.any(f < child_F_arr):
                    dominated = True
                    break
            
            if not dominated:
                # Remove front points dominated by child
                front_idx = remove_dominated_from_front(child_F_arr, front_idx, all_F)
                front_idx.append(new_idx)

        # Phase 3: Final polish - local refinement around best points
        if time.time() < deadline - 5.0 and len(front_idx) > 0:
            # Get the top few non-dominated points by crowding distance
            cd = crowding_distance(front_idx, all_F)
            top_k = min(5, len(front_idx))
            top_indices = np.argsort(cd)[-top_k:]  # largest crowding distance
            top_front_idx = [front_idx[i] for i in top_indices]
            
            polish_budget = min(20, budget - evals_used)
            for _ in range(polish_budget):
                if time.time() > deadline - 2.0:
                    break
                if evals_used >= budget:
                    break
                
                # Pick a random top point and do a small local step
                base_idx = top_front_idx[np.random.randint(len(top_front_idx))]
                base_X = all_X[base_idx]
                
                # Small Gaussian step
                small_scale = 0.02
                mutation = small_scale * (hi - lo) * np.random.standard_normal(dim)
                child_X = np.clip(base_X + mutation, lo, hi)
                
                child_F = problem(child_X)
                evals_used += 1
                
                all_X.append(child_X)
                all_F.append(child_F)
                
                # Update front
                new_idx = len(all_F) - 1
                child_F_arr = all_F[new_idx]
                dominated = False
                for fidx in front_idx:
                    f = all_F[fidx]
                    if np.all(f <= child_F_arr) and np.any(f < child_F_arr):
                        dominated = True
                        break
                if not dominated:
                    front_idx = remove_dominated_from_front(child_F_arr, front_idx, all_F)
                    front_idx.append(new_idx)

        return None
# EVOLVE-BLOCK-END
