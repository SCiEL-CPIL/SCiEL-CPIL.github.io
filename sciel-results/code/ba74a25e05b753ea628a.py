"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using MOEA/D-DE (Differential Evolution with decomposition)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Set population size. We need enough to cover the front but not so many
        # that we run out of budget for iterations.
        # Budget is 200. If pop=20, we have 180 iterations.
        # If pop=40, we have 160 iterations.
        # Let's try pop=20 to allow more iterations for convergence.
        N = 20
        self.budget = min(self.budget, 200)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        evals = N

        # Weight vectors for Tchebycheff decomposition
        # For 3 objectives, we generate weights summing to 1.
        # We use a systematic grid for better coverage.
        weights = []
        # For 3 objectives, we can use a simplex lattice.
        # Number of points in a simplex lattice of dimension d-1 with steps H is C(H+d-1, d-1).
        # Let's just generate random weights or a simple grid.
        # A simple grid for 3 objectives:
        # i from 0 to K, j from 0 to K-i
        # w = [i/K, j/K, (K-i-j)/K]
        # Let K=3 gives 10 points. K=4 gives 15 points.
        # We want N weights. Let's generate N random weights and normalize.
        # Or better, use a fixed set of weights that cover the simplex well.
        # For N=20, let's create 20 random weights.
        W = rng.random((N, self.n_obj))
        W /= W.sum(axis=1, keepdims=True)
        
        # Sort weights to ensure consistent neighborhood
        # This is a bit complex to do perfectly, so we'll just use random neighbors
        # or a simple index-based neighborhood.
        # For MOEA/D, each subproblem i has a neighborhood.
        # Let's define neighborhood as the N/2 closest weights.
        # Computing distance matrix
        dist = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
        neighbors = np.argsort(dist, axis=1)[:, :N//2 + 1]
        
        # Ideal point (lower bound of objectives seen so far)
        # WFG7 objectives are positive. 
        # We don't know the exact ideal point. We will estimate it.
        # Start with min of F.
        ideal_point = F.min(axis=0)
        
        # Reference point for Tchebycheff is usually the ideal point or a known bound.
        # We will update ideal_point as we go.
        
        # DE parameters
        CR = 0.5  # Crossover rate
        F_de = 0.8 # Differential weight (often denoted F, but F is used for objectives, so F_de)
        
        while evals < self.budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Update ideal point
            new_ideal = F.min(axis=0)
            # To avoid division by zero or instability, add a small epsilon if needed
            # But Tchebycheff uses (f_i - z_i^*) / w_i.
            # If z_i^* is the min, the min value will be 0.
            
            # Calculate Tchebycheff values for all individuals
            # t_i = max_j (w_ij * |f_ij - z_j^*|)
            # We need to handle the case where z_j^* is not the true ideal.
            # Using current min is standard online.
            
            # Vectorized calculation
            # F shape (N, M), W shape (N, M)
            # t[i, j] is the tchebycheff value of individual j for weight vector i
            # diff = abs(F - ideal_point)
            diff = np.abs(F - ideal_point)
            # t[i, j] = max_k (W[i, k] * diff[j, k])
            # This is a matrix multiplication-like op but with max instead of sum.
            # t = np.max(W[:, None, :] * diff[None, :, :], axis=2)
            # Shape: (N, N)
            t = np.max(W[:, None, :] * diff[None, :, :], axis=2)
            
            # For each subproblem i, select parents from its neighborhood
            for i in range(N):
                if evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Select 2 parents from neighborhood of i
                neigh = neighbors[i]
                # Ensure i is not selected as parent to avoid self-mutation issues? 
                # Usually parents are distinct from the target.
                # Let's pick 2 distinct parents from the neighborhood.
                # If neighborhood is small, handle carefully.
                parent_idx = rng.choice(neigh, size=2, replace=False)
                p1 = X[parent_idx[0]]
                p2 = X[parent_idx[1]]
                
                # Differential Evolution mutation
                v = p1 + F_de * (p1 - p2)
                
                # Crossover
                u = rng.random(self.dim)
                j_rand = rng.integers(self.dim)
                child = np.where(u < CR, v, X[i])
                child[j_rand] = v[j_rand]
                
                # Clip to bounds
                child = np.clip(child, lo, hi)
                
                # Evaluate
                fc = problem(child)
                evals += 1
                
                # Update population
                # Acceptance criterion: Does child improve the Tchebycheff value for subproblem i?
                # Or does it dominate?
                # Standard MOEA/D: Accept if t_child[i] < t_current[i]
                # t_child[i] = max_k (W[i, k] * |fc_k - ideal_point_k|)
                
                # Note: ideal_point might have changed? No, we updated it at start of loop.
                # But fc might be better than ideal_point.
                # If fc < ideal_point, the diff is negative? No, we use abs.
                # Wait, if fc is better (smaller), it should definitely be accepted.
                # The Tchebycheff value uses |f - z*|. If f < z*, |f-z*| is positive.
                # If z* is the min, and f < z*, then z* is not the min anymore.
                # We should update ideal_point before calculating t?
                # Or just compare the new Tchebycheff value.
                
                # Let's update ideal point locally for this child check?
                # Actually, if child is better than current ideal, it's a huge improvement.
                # Let's just compute the new Tchebycheff value using the CURRENT ideal_point.
                # If the child improves the ideal point, its Tchebycheff value might not reflect that fully
                # if the weight for that objective is low.
                # However, usually we check: if child dominates current, accept.
                # Else if tchebycheff(child) < tchebycheff(current), accept.
                
                # Dominance check
                # Child dominates X[i] if fc <= X[i] and fc < X[i] in at least one obj.
                dominates_current = np.all(fc <= F[i]) and np.any(fc < F[i])
                
                # Tchebycheff check
                # diff_child = np.abs(fc - ideal_point)
                # t_child_i = np.max(W[i] * diff_child)
                # t_current_i = t[i, i] # From the matrix computed above
                
                # But wait, if fc is better than ideal_point, ideal_point is stale.
                # Let's just use the dominance check primarily, and Tchebycheff as secondary.
                # If not dominating, compare Tchebycheff.
                
                if dominates_current:
                    X[i] = child
                    F[i] = fc
                    # Update ideal point if improved
                    # This is O(N) to update min, but we do it in the main loop.
                    # We can just let the main loop update it.
                else:
                    # Compute t for child specifically
                    diff_child = np.abs(fc - ideal_point)
                    t_child_i = np.max(W[i] * diff_child)
                    t_current_i = t[i, i]
                    
                    if t_child_i < t_current_i:
                        X[i] = child
                        F[i] = fc
                        
        return X
# EVOLVE-BLOCK-END
