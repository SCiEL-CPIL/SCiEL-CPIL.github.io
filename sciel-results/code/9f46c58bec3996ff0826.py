"""MOEA/D based multi-objective optimizer for DTLZ1."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size N=21 corresponds to a 5x5 grid on the 2-simplex for 3 objectives
        N = 21
        
        # Generate weight vectors on the simplex
        # For 3 objectives, points are (i/k, j/k, (k-i-j)/k) where i,j >= 0, i+j <= k
        k = 5
        weights = []
        for i in range(k + 1):
            for j in range(k + 1 - i):
                l1 = i / k
                l2 = j / k
                l3 = 1.0 - l1 - l2
                weights.append([l1, l2, l3])
        weights = np.array(weights)
        N = len(weights)
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Ideal point (z*)
        z_star = np.min(F, axis=0)
        
        # Neighborhood: For each weight, find the 10 nearest neighbors
        T = 10
        neighbors = []
        for i in range(N):
            dists = np.sum((weights - weights[i])**2, axis=1)
            # Sort indices by distance, exclude self
            sorted_idx = np.argsort(dists)
            neighbors.append(sorted_idx[1:T+1].tolist())
            
        # Crossover and Mutation parameters
        eta_c = 20.0 # SBX distribution index
        eta_m = 20.0 # Polynomial mutation distribution index
        p_m = 1.0 / dim # Mutation probability per variable

        def tchebycheff(f, w, z_star):
            # g(x, lambda) = max { lambda_i * |f_i(x) - z_i*| }
            return np.max(w * np.abs(f - z_star))

        def crossover(x1, x2):
            # SBX Crossover
            c = np.zeros(dim)
            for i in range(dim):
                if np.random.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        c[i] = x1[i]
                    else:
                        u = np.random.random()
                        if u <= 0.5:
                            alpha = 2.0 * u
                            beta = ((1 + 1.0/eta_c) / (1 + (2.0*(min(x1[i], x2[i]) - lo[i]) / (abs(x1[i] - x2[i])))))**(eta_c + 1.0)
                            c[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        else:
                            alpha = 2.0 * (1.0 - u)
                            beta = ((1 + 1.0/eta_c) / (1 + (2.0*(hi[i] - max(x1[i], x2[i])) / (abs(x1[i] - x2[i])))))**(eta_c + 1.0)
                            c[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                else:
                    c[i] = x1[i]
            return c

        def mutation(x):
            # Polynomial Mutation
            m = x.copy()
            for i in range(dim):
                if np.random.random() < p_m:
                    u = np.random.random()
                    delta1 = (u + 1) * (m[i] - lo[i]) / (hi[i] - lo[i])
                    delta2 = (u + 1) * (hi[i] - m[i]) / (hi[i] - lo[i])
                    if u < 0.5:
                        theta = (1 + 2.0 * (delta1 - (delta1 ** (eta_m + 1.0)))) / (2.0 * delta1)
                    else:
                        theta = (1 + 2.0 * ((1.0 - delta2) - ((1.0 - delta2) ** (eta_m + 1.0)))) / (2.0 * delta2)
                    m[i] = m[i] + (hi[i] - lo[i]) * (theta - 1.0)
            return np.clip(m, lo, hi)

        # Main Loop
        # We have budget - used evaluations left.
        # We will perform updates until budget is exhausted.
        # Each update generates 1 child (1 eval).
        i = 0
        while used < budget:
            # Select individual i to update
            # We cycle through i
            current_i = i % N
            i += 1
            
            # Select two parents from neighborhood of current_i
            if len(neighbors[current_i]) < 2:
                # Fallback to random if neighborhood is too small
                parents_idx = np.random.choice(N, 2, replace=False)
            else:
                # Choose 2 distinct parents from neighbors
                parent_candidates = neighbors[current_i]
                # Shuffle and pick 2
                np.random.shuffle(parent_candidates)
                parents_idx = [parent_candidates[0], parent_candidates[1]]
            
            x1 = X[parents_idx[0]]
            x2 = X[parents_idx[1]]
            
            # Crossover
            child_x = crossover(x1, x2)
            # Mutation
            child_x = mutation(child_x)
            
            # Evaluate
            child_f = problem(child_x)
            used += 1
            
            # Update z_star
            z_star = np.minimum(z_star, child_f)
            
            # Update Population
            # Try to replace current_i or a neighbor j
            g_child_i = tchebycheff(child_f, weights[current_i], z_star)
            g_curr_i = tchebycheff(F[current_i], weights[current_i], z_star)
            
            if g_child_i < g_curr_i:
                X[current_i] = child_x
                F[current_i] = child_f
            else:
                # Try neighbors
                if neighbors[current_i]:
                    j = np.random.choice(neighbors[current_i])
                    g_child_j = tchebycheff(child_f, weights[j], z_star)
                    g_curr_j = tchebycheff(F[j], weights[j], z_star)
                    if g_child_j < g_curr_j:
                        X[j] = child_x
                        F[j] = child_f

        return X
# EVOLVE-BLOCK-END
