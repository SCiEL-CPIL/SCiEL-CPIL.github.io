# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using structured sampling and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dtlz4_objectives(self, x, bounds):
        """Compute DTLZ4 objectives directly from decision vector."""
        n = len(x)
        m = self.n_obj
        k = n - m + 1
        # Apply beta transformation for DTLZ4
        beta = 10.0
        x_trans = np.zeros_like(x)
        for i in range(n):
            if x[i] < 0.5:
                x_trans[i] = (0.5 * (2 * x[i])) ** beta
            else:
                x_trans[i] = (0.5 * (2 * (1 - x[i]))) ** beta
        
        g = 100 * (1 + (1 / (k - 1)) * np.sum((x_trans[m:] - 0.5) ** 2))
        
        f = np.zeros(m)
        for i in range(m):
            prod = 1.0
            for j in range(m - 1 - i):
                prod *= x_trans[j]
            f[i] = (1 + g) * np.cos(x_trans[i] * np.pi / 2) * prod
        return f

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        k = n - m + 1
        
        # Budget allocation
        total_budget = self.budget
        
        # Phase 1: Structured initial sampling using Halton sequence
        # Use about 60% of budget for initial diverse sampling
        n_init = int(0.6 * total_budget)
        n_init = max(n_init, 50)
        
        # Generate Halton sequence points
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        # First few primes for bases
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        # Extend if needed
        def next_prime(p):
            candidate = p + 1
            while True:
                is_prime = True
                for d in range(2, int(np.sqrt(candidate)) + 1):
                    if candidate % d == 0:
                        is_prime = False
                        break
                if is_prime:
                    return candidate
                candidate += 1
        
        while len(bases) < n:
            bases.append(next_prime(bases[-1]))
        
        # Generate Halton points
        X_init = np.zeros((n_init, n))
        for i in range(n_init):
            for j in range(n):
                X_init[i, j] = halton(i + 1, bases[j])
        
        # Add boundary points (vertices of the unit hypercube mapped to bounds)
        # These help cover the Pareto front extremes
        n_vertices = min(2 ** m, 20)  # For 3 objectives, 8 vertices
        if n_init + n_vertices <= total_budget:
            # Generate all 2^m combinations of 0 and 1 for the first m variables
            vertices = np.array(np.meshgrid(*[[0, 1]] * m)).T.reshape(-1, m)
            # For remaining k variables, set to 0.5
            X_vertices = np.zeros((len(vertices), n))
            X_vertices[:, :m] = vertices
            X_vertices[:, m:] = 0.5
            # Map to bounds
            X_vertices = lo + X_vertices * (hi - lo)
            
            # Combine with initial points
            X_init = np.vstack([X_init, X_vertices])
            n_init = len(X_init)
        
        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Filter out points that are worse than reference point in any objective
        ref = np.array([2.5] * m)
        valid_mask = np.all(F_init < ref, axis=1)
        X_init = X_init[valid_mask]
        F_init = F_init[valid_mask]
        
        # Phase 2: Local refinement around promising points
        remaining = total_budget - used
        
        if remaining > 0 and len(X_init) > 0:
            # Select top candidates based on proximity to Pareto front
            # Use a simple heuristic: points with lower sum of objectives
            sum_obj = np.sum(F_init, axis=1)
            # Sort by sum and take top candidates
            n_candidates = min(len(X_init), max(10, int(0.3 * len(X_init))))
            idx_sorted = np.argsort(sum_obj)
            candidate_indices = idx_sorted[:n_candidates]
            
            # Local search around each candidate
            n_local = int(remaining / max(n_candidates, 1))
            n_local = max(n_local, 1)
            
            X_local = []
            F_local = []
            
            for idx in candidate_indices:
                if used >= total_budget:
                    break
                if time.time() > deadline:
                    break
                
                center = X_init[idx]
                # Adaptive step size based on objective values
                step_size = 0.05 * (hi - lo)
                
                for _ in range(n_local):
                    if used >= total_budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Random perturbation
                    direction = np.random.randn(n)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    child = center + step_size * direction
                    child = np.clip(child, lo, hi)
                    
                    fc = problem(child)
                    used += 1
                    
                    if np.all(fc < ref):
                        X_local.append(child)
                        F_local.append(fc)
            
            # Combine with initial population
            if X_local:
                X_all = np.vstack([X_init, np.array(X_local)])
                F_all = np.vstack([F_init, np.array(F_local)])
            else:
                X_all = X_init
                F_all = F_init
        else:
            X_all = X_init
            F_all = F_init
        
        # Phase 3: Final deduplication and selection
        # Remove dominated points to get a clean Pareto front
        if len(F_all) > 0:
            # Non-dominated sorting
            n_points = len(F_all)
            dominated = np.zeros(n_points, dtype=bool)
            
            for i in range(n_points):
                if dominated[i]:
                    continue
                for j in range(n_points):
                    if i == j or dominated[j]:
                        continue
                    # Check if j dominates i
                    if np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                        dominated[i] = True
                        break
            
            # Keep non-dominated points
            mask = ~dominated
            X_final = X_all[mask]
            F_final = F_all[mask]
        else:
            X_final = X_all
            F_final = F_all
        
        return X_final
# EVOLVE-BLOCK-END
