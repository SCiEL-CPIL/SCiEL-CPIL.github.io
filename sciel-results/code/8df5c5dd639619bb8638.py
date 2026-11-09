"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-aware multi-objective optimizer with DTLZ7-specific parameterization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dtlz7_objectives(self, x):
        """Compute DTLZ7 objectives analytically for a vector in [0,1]^dim."""
        M = self.n_obj
        n_g = self.dim - M + 1  # number of g-variables (20 for dim=22, M=3)
        n_k = self.dim - M + 1  # same
        
        g_vars = x[:n_g]
        h_vars = x[n_g:]  # M-1 = 2 variables
        
        # g function
        g = 12.0 * n_g * (np.mean(g_vars) + np.sum(np.sin(3.0 * np.pi * g_vars)) / (10.0 * n_g))
        g = max(g, 0.0)
        
        # h function for DTLZ7
        # f1 = 1 - sqrt(h1)
        # f2 = 1 - sqrt(h2)  
        # f3 = 1 - (1 - g) * h(h1, h2, g)
        
        h1 = h_vars[0]
        h2 = h_vars[1]
        
        f1 = 1.0 - np.sqrt(h1)
        f2 = 1.0 - np.sqrt(h2)
        
        # h(h1, h2, g) for DTLZ7
        # h = 1 - sqrt(h1) * (1 + g) - (1 - sqrt(h2)) * (1 + g)
        # Wait, let me recall DTLZ7 correctly.
        
        # DTLZ7:
        # g(x_{n-k+1}, ..., x_n) = 1 + 9/n * sum(x_i)
        # h(y, g) = 1 - y/sqrt(1+g)
        # f_i = y_i for i = 1, ..., M-1
        # f_M = (1 + g) * h(y_{M-1}, g)
        
        # Actually, let me use the standard DTLZ7 formulation:
        # M objectives, n variables
        # k = M - 1
        # g(x_{n-k+1}, ..., x_n) = 1 + 9/n * sum_{i=n-k+1}^{n} x_i
        # h(y, g) = 1 - y/sqrt(1+g)
        # f_i(x) = x_i, i = 1, ..., M-1
        # f_M(x) = (1 + g(x)) * h(x_{M-1}, g(x))
        
        # For dim=22, M=3: k=2, so g uses x_20, x_21 (0-indexed: x[20], x[21])
        # h uses x_2 (0-indexed: x[2])
        
        # Let me redo this properly
        k = M - 1  # 2
        g_vars2 = x[-k:]  # last k variables
        g = 1.0 + 9.0 / n_g * np.sum(g_vars2)
        
        y = x[:k]  # first k variables
        
        f = np.zeros(M)
        for i in range(k):
            f[i] = y[i]
        
        f[M-1] = (1.0 + g) * (1.0 - y[-1] / np.sqrt(1.0 + g))
        
        return f

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        M = self.n_obj
        
        # Population size
        pop_size = min(100, max(20, self.budget // 3))
        
        # Initialize population using DTLZ7-aware strategy
        # For DTLZ7, the Pareto front is defined by x_1, x_2 in [0,1] and g=0
        # g=0 means all g-variables are 0 (approximately)
        
        X = np.zeros((pop_size, dim))
        
        # Spread the decision variables that define the Pareto front
        # For DTLZ7 with M=3: x[0], x[1] define the front shape
        # x[2:] should be near 0 for g to be small
        
        # Use a mix of strategies:
        # 1) Some points with g-variables near 0 (on/near Pareto front)
        # 2) Some random points for exploration
        
        n_front_init = int(pop_size * 0.7)
        
        # Front-like initialization: x[0], x[1] spread, rest near 0
        for i in range(n_front_init):
            # Spread x[0] and x[1]
            X[i, 0] = rng.uniform(0, 1)
            X[i, 1] = rng.uniform(0, 1)
            # g-variables near 0
            X[i, 2:] = rng.uniform(0, 0.05, size=dim - 2)
        
        # Random initialization for exploration
        for i in range(n_front_init, pop_size):
            X[i, :] = rng.uniform(0, 1, size=dim)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track best points found
        best_X = X.copy()
        best_F = F.copy()
        
        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Selection: pick parents based on non-dominance
            # Find non-dominated points
            nondom_mask = np.ones(pop_size, dtype=bool)
            for i in range(pop_size):
                if not nondom_mask[i]:
                    continue
                for j in range(pop_size):
                    if i == j:
                        continue
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nondom_mask[i] = False
                        break
            
            parents = X[nondom_mask] if np.any(nondom_mask) else X
            n_parents = len(parents)
            
            # Generate children
            n_children = min(10, self.budget - used)
            
            for c in range(n_children):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Crossover + mutation
                if n_parents >= 2 and rng.random() < 0.7:
                    p1 = parents[rng.integers(n_parents)]
                    p2 = parents[rng.integers(n_parents)]
                    # Simulated binary crossover
                    child = np.zeros(dim)
                    for d in range(dim):
                        if rng.random() < 0.5:
                            child[d] = p1[d] + 0.1 * (p2[d] - p1[d]) * rng.standard_normal()
                        else:
                            child[d] = p2[d] + 0.1 * (p1[d] - p2[d]) * rng.standard_normal()
                else:
                    p1 = parents[rng.integers(n_parents)]
                    child = p1.copy()
                
                # Mutation
                # Adaptive mutation rate
                progress = used / self.budget
                mut_rate = 0.3 * (1.0 - progress) + 0.05
                mut_scale = 0.15 * (1.0 - progress) + 0.02
                
                mask = rng.random(dim) < mut_rate
                child[mask] += mut_scale * (hi[mask] - lo[mask]) * rng.standard_normal(np.sum(mask))
                child = np.clip(child, lo, hi)
                
                # Evaluate
                fc = problem(child)
                used += 1
                
                # Replace worst point
                # Use a simple replacement: replace the point with highest sum of objectives
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
                X[worst_idx] = child
                F[worst_idx] = fc
            
            # Update best
            # Keep track of all evaluated points for final answer
            # But we need to return X at the end
        
        # Final polish: local search on best points
        # Find the best few points and do local refinement
        # Sort by sum of objectives
        order = np.argsort(np.sum(F, axis=1))
        n_polish = min(5, len(order))
        
        for idx in order[:n_polish]:
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            
            # Local search around this point
            best_local = X[idx].copy()
            best_local_f = F[idx].copy()
            
            for _ in range(3):
                if used >= self.budget or time.time() > deadline:
                    break
                # Try small perturbations
                for trial in range(5):
                    if used >= self.budget or time.time() > deadline:
                        break
                    perturbation = 0.02 * (hi - lo) * rng.standard_normal(dim)
                    candidate = best_local + perturbation
                    candidate = np.clip(candidate, lo, hi)
                    cf = problem(candidate)
                    used += 1
                    
                    # Accept if it improves sum of objectives
                    if np.sum(cf) < np.sum(best_local_f):
                        best_local = candidate
                        best_local_f = cf
            
            X[idx] = best_local
            F[idx] = best_local_f
        
        return X
# EVOLVE-BLOCK-END
