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
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # 1. Generate Weight Vectors on Simplex - H=6 gives 28 points for 3 obj
        H = 6
        weights_list = []
        for i in range(H + 1):
            for j in range(H + 1 - i):
                k = H - i - j
                w = np.array([i / H, j / H, k / H])
                weights_list.append(w)
        N = len(weights_list)
        weights = np.array(weights_list)
        
        # 2. Define Neighbors - 8 nearest for better exploration
        neighbors = []
        for i in range(N):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            sorted_idx = np.argsort(dists)
            neigh = [sorted_idx[j] for j in range(1, 9) if sorted_idx[j] != i]
            neighbors.append(neigh)
        
        # 3. Initialize Population using LHS
        X = np.zeros((N, dim))
        for d in range(dim):
            edges = np.linspace(lo[d], hi[d], N + 1)
            centers = np.array([(edges[i] + edges[i+1]) / 2 for i in range(N)])
            np.random.shuffle(centers)
            X[:, d] = centers
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = N
        
        # Initialize Ideal Point
        z_star = np.min(F, axis=0)
        
        # 4. Evolution Loop
        budget_for_main = budget - 70  # Reserve 70 for local polish
        
        i_counter = 0
        
        while used < budget_for_main:
            if time.time() > deadline:
                break
            
            i = i_counter % N
            i_counter += 1
            
            # Adaptive mutation factor
            progress = used / max(budget_for_main, 1)
            F_mut = 0.8 - 0.2 * progress  # 0.8 -> 0.6
            
            # Select parents from neighbors
            candidate_indices = neighbors[i]
            if len(candidate_indices) < 3:
                candidate_indices = np.random.choice(N, 3, replace=False)
            else:
                dists_to_i = np.array([np.linalg.norm(weights[c] - weights[i]) for c in candidate_indices])
                probs = 1.0 / (dists_to_i + 1e-6)
                probs = probs / probs.sum()
                candidate_indices = np.random.choice(candidate_indices, 3, replace=False, p=probs)
            
            r1, r2, r3 = candidate_indices
            
            # DE Mutation
            v = X[r1] + F_mut * (X[r2] - X[r3])
            
            # Binomial Crossover
            j_rand = np.random.randint(dim)
            CR = 0.4
            trial = np.zeros(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = X[i][j]
            
            trial = np.clip(trial, lo, hi)
            
            f_trial = np.array(problem(trial))
            used += 1
            
            z_star = np.minimum(z_star, f_trial)
            
            # Update Population (MOEA/D with Tchebycheff)
            indices_to_check = [i] + neighbors[i]
            
            for k in indices_to_check:
                d_trial = np.max(weights[k] * np.abs(f_trial - z_star))
                d_current = np.max(weights[k] * np.abs(F[k] - z_star))
                
                if d_trial < d_current:
                    X[k] = trial
                    F[k] = f_trial
        
        # 5. Local Search using SLSQP on Tchebycheff scalarization
        if time.time() < deadline - 10:
            for i in range(N):
                if used >= budget:
                    break
                if time.time() > deadline - 3:
                    break
                
                w = weights[i].copy()
                x0 = X[i].copy()
                z = z_star.copy()
                
                # Define Tchebycheff scalarized objective
                def objective(x):
                    nonlocal used
                    if used >= budget:
                        return 1e10
                    if time.time() > deadline - 2:
                        return 1e10
                    
                    x_clipped = np.clip(x, lo, hi)
                    f = np.array(problem(x_clipped))
                    used += 1
                    # Tchebycheff: max(w_j * |f_j - z_j|)
                    return np.max(w * np.abs(f - z))
                
                # Run SLSQP
                try:
                    result = minimize(
                        objective,
                        x0,
                        method='SLSQP',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 40, 'ftol': 1e-6, 'disp': False}
                    )
                    
                    if result.success:
                        X[i] = result.x
                        F[i] = np.array(problem(result.x))
                        used += 1
                        z_star = np.minimum(z_star, F[i])
                except:
                    pass
        
        return X
