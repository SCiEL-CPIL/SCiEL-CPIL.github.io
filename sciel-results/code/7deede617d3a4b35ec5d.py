import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: Global Exploration
        n_explore = 20
        if budget < n_explore:
            n_explore = budget
            
        points = rng.uniform(-5, 5, (n_explore, n))
        values = np.zeros(n_explore)
        for i in range(n_explore):
            if time.time() > deadline:
                break
            values[i] = func(points[i])
            
        if time.time() > deadline:
            best_idx = np.argmin(values[:i+1])
            return points[best_idx], values[best_idx]

        best_idx = np.argmin(values)
        best_x = points[best_idx].copy()
        best_f = values[best_idx]
        
        remaining_budget = budget - n_explore
        if remaining_budget <= 0:
            return best_x, best_f
            
        # Phase 2: Nelder-Mead on the best point
        delta = 1.0
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x
        for i in range(n):
            simplex[i+1] = best_x
            simplex[i+1, i] += delta
            
        simplex = np.clip(simplex, -5, 5)
        
        f_simplex = np.zeros(n + 1)
        calls = n_explore
        for i in range(n + 1):
            if calls >= budget or time.time() > deadline:
                break
            f_simplex[i] = func(simplex[i])
            calls += 1
            
        if calls > budget or time.time() > deadline:
            for i in range(n + 1):
                if f_simplex[i] < best_f:
                    best_f = f_simplex[i]
                    best_x = simplex[i].copy()
            return best_x, best_f

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            idx = np.argsort(f_simplex)
            simplex_sorted = simplex[idx]
            f_sorted = f_simplex[idx]
            
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = simplex_sorted[0].copy()
                
            if np.max(f_sorted) - np.min(f_sorted) < 1e-10:
                break
                
            centroid = np.mean(simplex_sorted[:n], axis=0)
            
            x_r = centroid + alpha * (centroid - simplex_sorted[n])
            x_r = np.clip(x_r, -5, 5)
            
            if calls >= budget or time.time() > deadline:
                break
            f_r = func(x_r)
            calls += 1
            
            if f_r < f_sorted[0]:
                if calls >= budget or time.time() > deadline:
                    if f_r < best_f:
                        best_f = f_r
                        best_x = x_r.copy()
                    break
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                
                if f_e < f_r:
                    simplex_sorted[n] = x_e
                    f_sorted[n] = f_e
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                else:
                    simplex_sorted[n] = x_r
                    f_sorted[n] = f_r
                    if f_r < best_f:
                        best_f = f_r
                        best_x = x_r.copy()
            elif f_r < f_sorted[n-1]:
                simplex_sorted[n] = x_r
                f_sorted[n] = f_r
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
            else:
                if f_r < f_sorted[n]:
                    x_c = centroid + rho * (simplex_sorted[n] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    if calls >= budget or time.time() > deadline:
                        break
                    f_c = func(x_c)
                    calls += 1
                    
                    if f_c < f_r:
                        simplex_sorted[n] = x_c
                        f_sorted[n] = f_c
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()
                    else:
                        for i in range(1, n + 1):
                            simplex_sorted[i] = simplex_sorted[0] + sigma * (simplex_sorted[i] - simplex_sorted[0])
                            simplex_sorted[i] = np.clip(simplex_sorted[i], -5, 5)
                        if calls + (n) > budget or time.time() > deadline:
                            break
                        for i in range(1, n + 1):
                            f_sorted[i] = func(simplex_sorted[i])
                            calls += 1
                            if f_sorted[i] < best_f:
                                best_f = f_sorted[i]
                                best_x = simplex_sorted[i].copy()
                else:
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    if calls >= budget or time.time() > deadline:
                        break
                    f_c = func(x_c)
                    calls += 1
                    
                    if f_c < f_r:
                        simplex_sorted[n] = x_c
                        f_sorted[n] = f_c
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()
                    else:
                        for i in range(1, n + 1):
                            simplex_sorted[i] = simplex_sorted[0] + sigma * (simplex_sorted[i] - simplex_sorted[0])
                            simplex_sorted[i] = np.clip(simplex_sorted[i], -5, 5)
                        if calls + (n) > budget or time.time() > deadline:
                            break
                        for i in range(1, n + 1):
                            f_sorted[i] = func(simplex_sorted[i])
                            calls += 1
                            if f_sorted[i] < best_f:
                                best_f = f_sorted[i]
                                best_x = simplex_sorted[i].copy()
            
            simplex = simplex_sorted
            f_simplex = f_sorted
            
        return best_x, best_f
