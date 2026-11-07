"""Multi-objective optimizer on WFG7: Fast Differential Evolution with vectorized dominance."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """DE-based multi-objective optimizer with wall-clock deadline guard."""

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

        # Population size: Large enough for DE diversity, small enough for speed
        # With 200 budget, we can afford a population of ~50-100 if we run many generations
        # Let's start with 60.
        pop_size = 60
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track all evaluated points for hypervolume (deduplicated)
        all_F = F.copy()

        # DE Parameters
        CR = 0.9  # Crossover rate
        F_DE = 0.5 # Differential weight (usually 0.5-0.8)

        # Vectorized dominance check
        # Returns a boolean matrix D where D[i,j] is True if i dominates j
        # i dominates j if: all(F[i] <= F[j]) and any(F[i] < F[j])
        def vectorized_dominance(F_mat):
            n = len(F_mat)
            # F_mat shape (n, m)
            # less_equal: (n, n, m) -> (n, n)
            # F_mat[i, :] <= F_mat[j, :] for all m
            # We can do this with broadcasting
            # A[i, j] is True if F_mat[i] <= F_mat[j] component-wise
            # B[i, j] is True if F_mat[i] < F_mat[j] in at least one component
            
            # To save memory, we can compute in chunks or use efficient logic
            # n is ~60-100, so n*n*m is small (100*100*3 = 30k). This is very fast.
            
            # F_mat[:, np.newaxis, :] shape (n, 1, m)
            # F_mat[np.newaxis, :, :] shape (1, n, m)
            
            le = np.all(F_mat[:, np.newaxis, :] <= F_mat[np.newaxis, :, :], axis=2)
            lt = np.any(F_mat[:, np.newaxis, :] < F_mat[np.newaxis, :, :], axis=2)
            
            # i dominates j if le[i,j] and lt[i,j]
            return le & lt

        # Find non-dominated set (front 0)
        def get_non_dominated(F_mat):
            D = vectorized_dominance(F_mat)
            # A point is non-dominated if no other point dominates it
            # D[j, i] is True if j dominates i
            # So i is dominated if any(D[:, i]) is True
            dominated = np.any(D, axis=0)
            return np.where(~dominated)[0]

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            # DE/rand/1: v = x_r1 + F * (x_r2 - x_r3)
            # We generate one offspring per parent in the population for this generation
            # To keep it simple and fast, we iterate through the population
            
            new_X = np.zeros_like(X)
            new_F = np.zeros_like(F)
            
            # We will replace the population with the better of (parent, child) 
            # but since we have a budget constraint, we must be careful.
            # Standard DE for MOEA:
            # 1. Create trial vector
            # 2. Evaluate
            # 3. Select: if trial dominates parent OR (trial is non-dominated and parent is dominated) -> replace
            #    Actually, standard DE selection: 
            #    If trial dominates parent: replace
            #    If parent dominates trial: keep parent
            #    If non-dominated: keep both? 
            #    In constrained budget, we usually do a "tournament" or simple replacement.
            #    Let's use a simple replacement strategy: 
            #    If child dominates parent, replace.
            #    Else if parent does not dominate child, keep child with some probability? 
            #    No, to maintain diversity, let's just do:
            #    If child is better (dominates) or equal, replace.
            #    But this might collapse diversity.
            
            # Better strategy for MOEA-DE:
            # Keep a separate archive? No, budget is tight.
            # Let's use the "Pareto selection" from DE literature:
            # For each i:
            #   Generate v_i
            #   f(v_i)
            #   If v_i dominates X_i: X_i = v_i
            #   Else if X_i does not dominate v_i: 
            #       # Both non-dominated relative to each other
            #       # Use a fitness indicator or random?
            #       # To encourage spread, let's keep v_i with 50% probability?
            #       # Or better: add to a temporary pool and select best pop_size?
            #       # That doubles evaluations. We have 200 budget.
            #       # Pop 60. 60 evals per gen. 3 gens = 180. 
            #       # If we double, we only get 1.5 gens.
            
            # Let's stick to single replacement to maximize generations.
            # To prevent premature convergence, we rely on the large initial population
            # and the fact that DE explores well.
            
            # Optimization: We can vectorize the creation of v
            # But selection depends on individual dominance.
            
            # Let's loop through individuals. It's fast enough for N=60.
            
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break

                # Choose 3 distinct random indices
                # Efficient way: sample 3 without replacement from range(pop_size)
                # Exclude i? DE/rand/1 usually picks r1, r2, r3 distinct from i and each other.
                # rng.choice is slow. Let's use manual selection or pre-computed?
                # For N=60, rng.choice is fine.
                idxs = rng.choice(pop_size, size=3, replace=False)
                # Ensure none is i
                # If i is in idxs, re-sample or swap.
                # Simple retry loop
                while i in idxs:
                    idxs = rng.choice(pop_size, size=3, replace=False)
                    
                r1, r2, r3 = idxs
                
                # v = X[r1] + F_DE * (X[r2] - X[r3])
                v = X[r1] + F_DE * (X[r2] - X[r3])
                
                # Crossover: binomial crossover
                # j_rand
                j_rand = rng.integers(0, self.dim)
                u = rng.random(self.dim)
                
                # u <= CR or d == j_rand
                mask = (u <= CR) | (np.arange(self.dim) == j_rand)
                
                child = np.where(mask, v, X[i])
                
                # Clip to bounds
                child = np.clip(child, lo, hi)
                
                # Evaluate
                fc = problem(child)
                used += 1
                all_F = np.vstack([all_F, fc.reshape(1, -1)])
                
                # Selection
                # Check dominance between child (fc) and parent (F[i])
                # fc dominates F[i]?
                if np.all(fc <= F[i]) and np.any(fc < F[i]):
                    # Child dominates parent
                    X[i] = child
                    F[i] = fc
                elif np.all(F[i] <= fc) and np.any(F[i] < fc):
                    # Parent dominates child: keep parent
                    pass
                else:
                    # Non-dominated
                    # To maintain diversity, we can use a random choice or 
                    # check against the whole population? 
                    # Checking against whole population is expensive.
                    # Let's just keep the child with 50% probability to allow exploration
                    # Or better: keep the one with lower sum of objectives?
                    # Lower sum is a good proxy for quality in non-dominated case.
                    if np.sum(fc) < np.sum(F[i]):
                        X[i] = child
                        F[i] = fc
                    # Else keep parent

        return X
# EVOLVE-BLOCK-END
