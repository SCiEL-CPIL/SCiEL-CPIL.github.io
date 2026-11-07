"""Multi-objective optimizer on WFG7: Multi-Island Differential Evolution."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-Island DE-based multi-objective optimizer with wall-clock deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # 10s safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Multi-Island Setup
        # With 200 budget, we can afford 4 islands of 20 individuals each.
        # Total initial evals: 80. Remaining: 120.
        # This allows for more generations and migration than a single pop of 60.
        n_islands = 4
        pop_size = 20  # Size per island
        total_pop = n_islands * pop_size
        
        # Initialize islands
        # Each island gets a distinct random seed offset to ensure initial diversity
        # We initialize all at once for speed
        X_all = lo + (hi - lo) * rng.random((total_pop, self.dim))
        F_all = np.array([problem(x) for x in X_all], dtype=np.float64)
        used = total_pop

        # Reshape into islands
        X_islands = [X_all[i*pop_size:(i+1)*pop_size].copy() for i in range(n_islands)]
        F_islands = [F_all[i*pop_size:(i+1)*pop_size].copy() for i in range(n_islands)]

        # Track all evaluated points for hypervolume (deduplicated logic handled by evaluator recording, 
        # but we need to keep a record if we want to return something meaningful, though return is ignored.
        # The evaluator records every call to problem(). So we just need to make the calls.)
        
        # DE Parameters
        CR = 0.9
        F_DE = 0.5

        # Helper: Dominance check for a single vector against a population
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Helper: Get non-dominated indices for a small population
        def get_non_dominated_indices(F_mat):
            n = len(F_mat)
            if n == 0:
                return np.array([], dtype=int)
            # Vectorized dominance for small n
            # D[i, j] is True if i dominates j
            le = np.all(F_mat[:, np.newaxis, :] <= F_mat[np.newaxis, :, :], axis=2)
            lt = np.any(F_mat[:, np.newaxis, :] < F_mat[np.newaxis, :, :], axis=2)
            D = le & lt
            # A point is non-dominated if no other point dominates it
            # D[j, i] is True if j dominates i
            dominated = np.any(D, axis=0)
            return np.where(~dominated)[0]

        # Main Loop
        gen = 0
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring for each island
            for k in range(n_islands):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                X_k = X_islands[k]
                F_k = F_islands[k]
                
                for i in range(pop_size):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break

                    # Choose 3 distinct random indices from the island
                    # Exclude i
                    idxs = rng.choice(pop_size, size=3, replace=False)
                    while i in idxs:
                        idxs = rng.choice(pop_size, size=3, replace=False)
                    
                    r1, r2, r3 = idxs
                    
                    # Mutation
                    v = X_k[r1] + F_DE * (X_k[r2] - X_k[r3])
                    
                    # Crossover
                    j_rand = rng.integers(0, self.dim)
                    u = rng.random(self.dim)
                    mask = (u <= CR) | (np.arange(self.dim) == j_rand)
                    
                    child = np.where(mask, v, X_k[i])
                    child = np.clip(child, lo, hi)
                    
                    # Evaluate
                    fc = problem(child)
                    used += 1
                    
                    # Selection
                    # Compare child (fc) with parent (F_k[i])
                    if dominates(fc, F_k[i]):
                        # Child dominates parent
                        X_k[i] = child
                        F_k[i] = fc
                    elif dominates(F_k[i], fc):
                        # Parent dominates child: keep parent
                        pass
                    else:
                        # Non-dominated
                        # Use a simple diversity metric: distance to centroid of island? 
                        # Or just random replacement to maintain exploration?
                        # Let's use sum of objectives as a tie-breaker, but favor diversity if close.
                        # Actually, in multi-island, we want spread. 
                        # If we always pick lower sum, islands converge to same point.
                        # Let's pick the one with larger distance to the island centroid in objective space.
                        
                        # Calculate island centroid in objective space
                        centroid = np.mean(F_k, axis=0)
                        dist_parent = np.linalg.norm(F_k[i] - centroid)
                        dist_child = np.linalg.norm(fc - centroid)
                        
                        # Prefer the one further from centroid (more diverse)
                        # But if one is significantly better in sum, prefer it?
                        # Let's use a hybrid: 
                        # If sum difference is small, pick diverse. Else pick better sum.
                        sum_diff = np.sum(F_k[i]) - np.sum(fc) # Positive if child is better sum
                        if abs(sum_diff) < 0.1 * np.mean(np.sum(F_k)):
                            if dist_child > dist_parent:
                                X_k[i] = child
                                F_k[i] = fc
                        else:
                            if sum_diff > 0: # Child is better sum
                                X_k[i] = child
                                F_k[i] = fc

            # Migration: Every few generations, swap best individuals between islands
            gen += 1
            if gen % 5 == 0:
                for k in range(n_islands):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Find non-dominated in island k
                    nd_idx = get_non_dominated_indices(F_islands[k])
                    if len(nd_idx) == 0:
                        continue
                    
                    # Pick a random non-dominated individual from island k
                    donor_idx = nd_idx[rng.integers(0, len(nd_idx))]
                    donor_X = X_islands[k][donor_idx].copy()
                    donor_F = F_islands[k][donor_idx].copy()
                    
                    # Choose a random island m != k
                    m = rng.integers(0, n_islands)
                    while m == k:
                        m = rng.integers(0, n_islands)
                    
                    # Find non-dominated in island m
                    nd_idx_m = get_non_dominated_indices(F_islands[m])
                    if len(nd_idx_m) == 0:
                        continue
                    
                    # Pick a random non-dominated individual from island m to replace
                    # We replace the one with the worst crowding or just random?
                    # Let's replace a random non-dominated individual to allow mixing
                    replace_idx = nd_idx_m[rng.integers(0, len(nd_idx_m))]
                    
                    # Check if donor is better than the one being replaced
                    # If donor dominates replacee, replace.
                    # If replacee dominates donor, keep.
                    # If non-dominated, replace with probability 0.5 to encourage mixing
                    if dominates(donor_F, F_islands[m][replace_idx]):
                        X_islands[m][replace_idx] = donor_X
                        F_islands[m][replace_idx] = donor_F
                    elif dominates(F_islands[m][replace_idx], donor_F):
                        pass
                    else:
                        if rng.random() < 0.5:
                            X_islands[m][replace_idx] = donor_X
                            F_islands[m][replace_idx] = donor_F

        return X_all
# EVOLVE-BLOCK-END
