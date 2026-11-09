import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        # Phase 1: Latin Hypercube Sampling for initial exploration
        pop_size = 30
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            X[:, d] = (perm + np.random.rand(pop_size)) / pop_size
            X[:, d] = lo[d] + X[:, d] * (hi[d] - lo[d])
        
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        def fast_non_dominated_sort(F):
            M = len(F)
            S = [set() for _ in range(M)]
            n = [0 for _ in range(M)]
            front = []
            
            for p in range(M):
                for q in range(M):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n[p] += 1
                if n[p] == 0:
                    front.append(p)
            
            fronts = []
            while front:
                fronts.append(front)
                next_front = []
                for p in front:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            next_front.append(q)
                front = next_front
            return fronts

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return [float('inf')] * len(indices)
            
            m = len(indices)
            cd = [0.0] * m
            k = F.shape[1]
            
            for j in range(k):
                sorted_indices = sorted(range(m), key=lambda i: F[indices[i]][j])
                cd[sorted_indices[0]] = float('inf')
                cd[sorted_indices[-1]] = float('inf')
                for i in range(1, m - 1):
                    idx_curr = sorted_indices[i]
                    idx_prev = sorted_indices[i-1]
                    idx_next = sorted_indices[i+1]
                    
                    f_curr = F[indices[idx_curr]][j]
                    f_prev = F[indices[idx_prev]][j]
                    f_next = F[indices[idx_next]][j]
                    
                    denom = f_next - f_prev
                    if denom > 1e-12:
                        cd[idx_curr] += (f_curr - f_prev) / denom
                    else:
                        cd[idx_curr] += float('inf')
            return cd

        # Phase 2: Chebyshev scalarization with L-BFGS-B refinement
        # Generate diverse weight vectors on simplex
        n_weights = 20
        weights = []
        for i in range(n_weights):
            # Generate random points on the simplex
            w = np.random.dirichlet(np.ones(n_obj))
            weights.append(w)
        
        # For each weight vector, run L-BFGS-B from a random starting point
        # Use constrained Chebyshev scalarization: min max_i(w_i * f_i)
        for w in weights:
            if evals >= budget:
                break
            if time.time() > deadline:
                break
            
            # Define the scalarized objective function
            def scalarized_obj(x):
                nonlocal evals
                fx = problem(x)
                evals += 1
                # Constrained Chebyshev: max(w_i * f_i) + epsilon * sum(w_i * f_i)
                weighted = w * fx
                return np.max(weighted) + 0.01 * np.sum(weighted)
            
            # Try from a few random starting points
            for _ in range(2):
                if evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x0 = np.random.uniform(lo, hi)
                
                try:
                    result = minimize(
                        scalarized_obj, x0,
                        method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 50, 'maxfun': 20}
                    )
                    
                    if result.success and evals < budget:
                        fx = problem(result.x)
                        evals += 1
                        F = np.vstack([F, [fx]])
                except Exception:
                    pass
        
        # Combine results and select non-dominated solutions
        fronts = fast_non_dominated_sort(F)
        
        # Select the best non-dominated solutions
        all_keys = []
        for i_f, front in enumerate(fronts):
            if len(front) > 0:
                cd = crowding_distance(F, front)
                for j, idx in enumerate(front):
                    all_keys.append((i_f, -cd[j], idx))
        
        all_keys.sort(key=lambda x: (x[0], x[1]))
        selected_indices = [k[2] for k in all_keys[:pop_size]]
        
        F_final = F[selected_indices]
        
        return F_final
