import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        max_time = 280  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        used = 0
        all_points = []
        
        # PSO Parameters
        # Budget 200. 
        # Let's use a population size of 20.
        # Initial: 20 evals.
        # Remaining: 180 evals.
        # 180 / 20 = 9 iterations.
        # This is a small number of iterations for PSO, but with 4D and simple bounds, 
        # combined with the "global best per objective" strategy, it might spread well.
        # Alternatively, smaller population, more iterations?
        # Let's try Pop=10, Iterations=18. Total 20 + 180 = 200.
        
        pop_size = 10
        max_iter = (budget - pop_size) // pop_size
        
        # Initialize positions
        X = lo + np.random.random((pop_size, dim)) * (hi - lo)
        
        # Evaluate initial population
        V = np.zeros_like(X) # Velocities
        F = np.zeros((pop_size, n_obj))
        
        for i in range(pop_size):
            f = problem(X[i])
            F[i] = f
            all_points.append(f)
            used += 1
            
            if used >= budget or time.time() - start_time > max_time:
                return all_points

        # Initialize personal bests (Pbest)
        Pbest_x = X.copy()
        Pbest_f = F.copy()
        
        # Initialize global bests per objective (Gbest)
        # Gbest[j] is the position of the best point found so far for objective j
        Gbest_x = np.zeros((n_obj, dim))
        Gbest_f = np.zeros(n_obj)
        
        for j in range(n_obj):
            best_idx = np.argmin(F[:, j])
            Gbest_x[j] = Pbest_x[best_idx]
            Gbest_f[j] = Pbest_f[best_idx]

        # Inertia weight decay
        w_start = 0.9
        w_end = 0.4
        c1 = 1.5 # Cognitive
        c2 = 1.5 # Social
        
        for it in range(max_iter):
            if used >= budget or time.time() - start_time > max_time:
                break
                
            # Update inertia
            w = w_start - (w_start - w_end) * (it / max_iter)
            
            # Update velocities
            for i in range(pop_size):
                if used >= budget or time.time() - start_time > max_time:
                    break
                    
                r1 = np.random.random(dim)
                r2 = np.random.random(dim)
                
                # Standard PSO velocity update uses one global best.
                # Here we use a weighted sum of objective-specific global bests?
                # Or just pick one? 
                # A common multi-objective PSO approach uses a reference point or 
                # non-dominated set. 
                # Simpler: For each particle, update towards the global best of 
                # the objective that is currently "worst" for that particle?
                # Or just average the pull from all objective bests?
                # Let's try: Velocity update pulls towards Gbest of each objective,
                # weighted by how bad that objective is for the current particle relative to global best.
                # Actually, simpler and robust: 
                # Use the "Global Best" as the best non-dominated point? 
                # No, let's stick to the "Objective Specific Global Best" but combine them.
                
                # Let's try a strategy where each particle is attracted to the 
                # global best of the objective it is currently worst at.
                # Or simpler: Attract to the centroid of the global bests?
                
                # Let's use a random objective's global best for the social term 
                # to maintain diversity, but weight the cognitive term strongly.
                # Actually, let's just use the average of the Gbests for the social term.
                # Gbest_avg = np.mean(Gbest_x, axis=0)
                
                # Better: For particle i, find which objective j it is worst at 
                # compared to its Pbest? No.
                # Let's use the "Global Best" defined as the point that minimizes 
                # the sum of normalized objectives? No, normalization is tricky.
                
                # Let's try: Social term uses the Gbest of the objective j 
                # that has the largest gap between particle i's value and Gbest_f[j].
                
                gaps = F[i] - Gbest_f
                # If all gaps are negative, particle is better than all Gbests? 
                # Then gaps are negative. We want to attract to the one it's worst at (largest positive gap).
                # If all are negative, it's a Pareto improvement candidate. 
                # Let's just pick a random objective for the social pull to ensure exploration.
                j_social = np.random.randint(n_obj)
                
                v_new = w * V[i] + c1 * r1 * (Pbest_x[i] - X[i]) + c2 * r2 * (Gbest_x[j_social] - X[i])
                
                # Clip velocity to avoid exploding
                # Max velocity step is usually (hi - lo)
                max_v = (hi - lo) * 0.5
                v_new = np.clip(v_new, -max_v, max_v)
                
                V[i] = v_new
                X[i] = X[i] + V[i]
                
                # Clip position to bounds
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                f = problem(X[i])
                F[i] = f
                all_points.append(f)
                used += 1
                
                # Update Pbest
                if np.all(f <= Pbest_f[i]):
                    # Check if strictly better in at least one, or equal
                    if np.any(f < Pbest_f[i]) or np.all(f == Pbest_f[i]):
                         # Standard Pbest update: if new is not dominated by old?
                         # For simplicity in single-objective logic applied here:
                         # We keep the point that is better in the "sum" sense?
                         # No, we want to keep diverse points.
                         # Let's update Pbest if the new point is non-dominated w.r.t Pbest
                         # OR if it is better in the objective we are currently tracking?
                         
                         # Simple heuristic: Update Pbest if new point is better in 
                         # ANY objective and not worse in ANY objective (Pareto improvement)
                         # OR if it is simply better in the "sum" of objectives (for tie-breaking)
                         
                         if np.all(f <= Pbest_f[i]) and np.any(f < Pbest_f[i]):
                             Pbest_x[i] = X[i]
                             Pbest_f[i] = f
                         elif np.all(f == Pbest_f[i]):
                             pass # Keep old
                         else:
                             # If it's worse in some and better in others, 
                             # we might want to keep it if it's "more diverse"?
                             # For standard PSO, we usually track a single scalar.
                             # Here, let's update Pbest only if it's a Pareto improvement 
                             # over the current Pbest. This keeps the Pbest as the 
                             # "best" known point for that particle's history.
                             pass

                # Update Gbests
                for j in range(n_obj):
                    if f[j] < Gbest_f[j]:
                        Gbest_x[j] = X[i]
                        Gbest_f[j] = f[j]
                        
        return all_points
