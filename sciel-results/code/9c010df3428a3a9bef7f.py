import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        rng = np.random.default_rng(0)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Parameters
        pop_size = 20
        # Ensure we don't exceed budget
        # Init: pop_size
        # Generations: (budget - pop_size) // pop_size
        n_generations = (self.budget - pop_size) // pop_size
        
        # 1. Generate Weight Vectors (Uniform on Simplex)
        weights = np.zeros((pop_size, n_obj))
        for i in range(pop_size):
            # Generate n_obj - 1 random numbers
            r = np.sort(rng.random(n_obj - 1))
            r = np.concatenate(([0.0], r, [1.0]))
            weights[i] = np.diff(r)
            
        # 2. Find Neighbors (K nearest neighbors in weight space)
        k_neighbors = 5
        neighbors = []
        # Compute pairwise distances
        dists = np.zeros((pop_size, pop_size))
        for i in range(pop_size):
            for j in range(pop_size):
                dists[i, j] = np.linalg.norm(weights[i] - weights[j])
        
        for i in range(pop_size):
            idx = np.argsort(dists[i])
            idx = idx[idx != i] # Exclude self
            neighbors.append(idx[:k_neighbors])

        # 3. Initialize Population (LHS)
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            for j in range(dim):
                # LHS: partition [0,1] into pop_size intervals, pick one random point in each for each dim
                # But standard LHS picks a permutation for each dimension
                pass
                
        # Proper LHS:
        X = np.zeros((pop_size, dim))
        for j in range(dim):
            # Random permutation of indices
            perm = rng.permutation(pop_size)
            # For each row i, the value in column j is based on perm[i]
            # Interval index is perm[i]
            # Random offset in interval
            offsets = rng.random(pop_size)
            X[:, j] = lo[j] + (hi[j] - lo[j]) * (perm + offsets) / pop_size

        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Ideal Point
        z_star = np.min(F, axis=0)
        
        # 4. Evolutionary Loop
        for gen in range(n_generations):
            if time.time() > deadline:
                break
                
            for i in range(pop_size):
                if time.time() > deadline or used >= self.budget:
                    break
                
                # Select parents from neighbors of i
                n_idx = neighbors[i]
                # Pick 2 distinct parents
                r1_idx = n_idx[rng.integers(len(n_idx))]
                r2_idx = n_idx[rng.integers(len(n_idx))]
                while r2_idx == r1_idx:
                    r2_idx = n_idx[rng.integers(len(n_idx))]
                    
                parent1 = X[r1_idx]
                parent2 = X[r2_idx]
                
                # Crossover (SBX)
                child = self._sbx_crossover(parent1, parent2, 0.9, 20.0, lo, hi, rng)
                
                # Mutation (Polynomial)
                child = self._poly_mutation(child, 1.0/dim, 20.0, lo, hi, rng)
                
                # Evaluate
                f_child = problem(child)
                used += 1
                
                # Update Ideal Point
                z_star = np.minimum(z_star, f_child)
                
                # Update Subproblem i
                # Tchebycheff metric
                g_i_old = np.max(weights[i] * np.abs(F[i] - z_star))
                g_i_new = np.max(weights[i] * np.abs(f_child - z_star))
                
                if g_i_new < g_i_old:
                    X[i] = child
                    F[i] = f_child
                    
                    # Update neighbors
                    for j in neighbors[i]:
                        if j == i:
                            continue
                        g_j_old = np.max(weights[j] * np.abs(F[j] - z_star))
                        g_j_new = np.max(weights[j] * np.abs(f_child - z_star))
                        if g_j_new < g_j_old:
                            X[j] = child
                            F[j] = f_child

        return X

    def _sbx_crossover(self, p1, p2, prob, eta, lo, hi, rng):
        if rng.random() > prob:
            # No crossover, return p1 (or p2)
            return p1.copy()
            
        child = np.zeros_like(p1)
        for i in range(len(p1)):
            if rng.random() < 0.5:
                child[i] = p1[i]
            else:
                child[i] = p2[i]
                # Apply SBX
                if p1[i] != p2[i]:
                    u = rng.random()
                    if u <= 0.5:
                        alpha = 2 * u
                        beta = (1 + 2 * (abs(p1[i] - p2[i]) / (hi[i] - lo[i]))) ** (eta + 1)
                        child[i] = 0.5 * (p1[i] + p2[i] - (p2[i] - p1[i]) * (beta - 1))
                    else:
                        alpha = 2 * (1 - u)
                        beta = (1 + 2 * (abs(p1[i] - p2[i]) / (hi[i] - lo[i]))) ** (eta + 1)
                        child[i] = 0.5 * (p1[i] + p2[i] + (p2[i] - p1[i]) * (beta - 1))
        return np.clip(child, lo, hi)

    def _poly_mutation(self, x, prob, eta, lo, hi, rng):
        for i in range(len(x)):
            if rng.random() < prob:
                u = rng.random()
                delta1 = (1 + 2 * (x[i] - lo[i]) / (hi[i] - lo[i])) ** -(eta + 1)
                delta2 = (1 + 2 * (hi[i] - x[i]) / (hi[i] - lo[i])) ** -(eta + 1)
                if u < 0.5:
                    xy = 1 - delta1
                else:
                    xy = 1 - delta2
                x[i] = x[i] + (hi[i] - lo[i]) * (xy ** eta - 1)
        return np.clip(x, lo, hi)
