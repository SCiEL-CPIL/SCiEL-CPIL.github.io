import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size will be set after weight generation
        # 1. Generate Weight Vectors on Simplex - use structured grid for reliability
        # For 3 objectives, use a grid with H subdivisions
        H = 4  # gives (H+1)(H+2)/2 = 15 points, we need 20, so use H=5 -> 21 points
        H = 5
        weights_list = []
        for i in range(H + 1):
            for j in range(H + 1 - i):
                k = H - i - j
                w = np.array([i / H, j / H, k / H])
                weights_list.append(w)
        # We have 21 points, use all of them (N=21)
        N = len(weights_list)
        weights = np.array(weights_list)
        
        # 2. Define Neighbors
        # Find 5 nearest neighbors for each weight vector
        neighbors = []
        for i in range(N):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            # Sort indices by distance, exclude self
            sorted_idx = np.argsort(dists)
            neigh = [sorted_idx[j] for j in range(1, 6) if sorted_idx[j] != i]
            neighbors.append(neigh)
        
        # 3. Initialize Population using Latin Hypercube Sampling
        # For dim=4, N=21
        X = np.zeros((N, dim))
        for d in range(dim):
            # Create 20 bins
            edges = np.linspace(lo[d], hi[d], N + 1)
            centers = np.array([(edges[i] + edges[i+1]) / 2 for i in range(N)])
            # Shuffle to decorrelate dimensions
            np.random.shuffle(centers)
            X[:, d] = centers
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = N
        
        # Initialize Ideal Point (z*)
        z_star = np.min(F, axis=0)
        
        # 4. Evolution Loop
        # We have budget - used evaluations left.
        # We will iterate through individuals cyclically.
        
        i_counter = 0
        
        while used < budget:
            if time.time() > deadline:
                break
            
            i = i_counter % N
            i_counter += 1
            
            # Select parents from neighbors of i
            # If neighbors list is empty or small, use whole population
            candidate_indices = neighbors[i]
            if len(candidate_indices) < 3:
                # Pick random from whole population
                candidate_indices = np.random.choice(N, 3, replace=False)
            else:
                # Pick 3 distinct from neighbors
                candidate_indices = np.random.choice(candidate_indices, 3, replace=False)
            
            r1, r2, r3 = candidate_indices
            
            # DE Mutation - higher mutation factor for better exploration in 4D
            F_mut = 1.0
            v = X[r1] + F_mut * (X[r2] - X[r3])
            
            # Crossover (Binomial) - higher CR for more exploration
            j_rand = np.random.randint(dim)
            CR = 0.7
            trial = np.zeros(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = X[i][j]
            
            # Clip to bounds
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            f_trial = np.array(problem(trial))
            used += 1
            
            # Update Ideal Point
            z_star = np.minimum(z_star, f_trial)
            
            # Update Population (MOEA/D strategy)
            # Check current individual i and its neighbors
            indices_to_check = [i] + neighbors[i]
            
            for k in indices_to_check:
                # Calculate Tchebycheff distance for trial and current X[k]
                # d_te(f, z*, w) = max(w_i * |f_i - z*_i|)
                
                # We need to handle the case where z* is updated.
                # The formula uses the current z*.
                
                d_trial = np.max(weights[k] * np.abs(f_trial - z_star))
                d_current = np.max(weights[k] * np.abs(F[k] - z_star))
                
                if d_trial < d_current:
                    X[k] = trial
                    F[k] = f_trial

        # 5. Local polish phase using Nelder-Mead on non-dominated points
        if time.time() < deadline - 30:
            from scipy.optimize import minimize
            
            # Find non-dominated points in the population
            non_dom_idx = []
            for i in range(N):
                dominated = False
                for j in range(N):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dom_idx.append(i)
            
            # Polish each non-dominated point
            for idx in non_dom_idx[:10]:  # Limit to top 10 for time safety
                if time.time() > deadline - 5:
                    break
                
                x0 = X[idx].copy()
                # Use weighted sum as scalar objective for local optimization
                w = weights[idx]
                
                def scalar_obj(x):
                    fx = np.array(problem(x))
                    return np.dot(w, fx)
                
                try:
                    res = minimize(scalar_obj, x0, method='Nelder-Mead', 
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10})
                    if res.success and res.fun < np.dot(w, F[idx]):
                        X[idx] = res.x
                        F[idx] = np.array(problem(res.x))
                        used += 1
                except:
                    pass
        
        return X
