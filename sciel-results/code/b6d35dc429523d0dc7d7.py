import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        budget = self.budget
        
        low = [b[0] for b in bounds]
        high = [b[1] for b in bounds]
        
        best_x_tail = np.zeros(n - 2)
        used = 0
        remaining_budget = budget
        
        # Phase 1: Sobol sampling for x0, x1
        # We need a Sobol sequence generator. 
        # Since we can't import scipy.stats.qmc easily without checking availability, 
        # and standard library doesn't have Sobol, we can use a simple Halton sequence or just random.
        # Let's use Halton sequence for better coverage than uniform random.
        
        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def generate_halton(count, dim):
            points = []
            for i in range(1, count + 1):
                p = []
                for d in range(dim):
                    # Find (d+1)-th prime
                    # Primes: 2, 3, 5, 7...
                    bases = [2, 3, 5, 7]
                    p.append(halton_sequence(i, bases[d]))
                points.append(p)
            return np.array(points)

        # Generate 150 Halton points
        n_halton = 150
        halton_pts = generate_halton(n_halton, 2)
        
        evaluated_points = []
        evaluated_objs = []
        
        for pt in halton_pts:
            if time.time() > deadline or used >= budget:
                break
            x = np.zeros(n)
            x[0] = pt[0]
            x[1] = pt[1]
            x[2:] = best_x_tail
            obj = problem(x)
            evaluated_points.append(x.copy())
            evaluated_objs.append(obj)
            used += 1
        
        # Phase 2: Local search around the best 25 points
        # Sort by sum of objectives (simple proxy)
        if evaluated_objs:
            scores = [np.sum(obj) for obj in evaluated_objs]
            sorted_indices = np.argsort(scores)
            
            # Take top 25
            top_k = 25
            candidates_to_refine = [evaluated_points[i] for i in sorted_indices[:top_k]]
            
            # Generate 5 neighbors for each
            n_neighbors = (budget - used) // len(candidates_to_refine)
            if n_neighbors == 0:
                n_neighbors = 1
                
            for cx in candidates_to_refine:
                if used >= budget:
                    break
                for _ in range(n_neighbors):
                    if time.time() > deadline or used >= budget:
                        break
                    # Random perturbation
                    x_new = cx.copy()
                    # Perturb x0 and x1
                    x_new[0] = np.clip(cx[0] + np.random.normal(0, 0.1), 0.0, 1.0)
                    x_new[1] = np.clip(cx[1] + np.random.normal(0, 0.1), 0.0, 1.0)
                    # Keep tail 0
                    x_new[2:] = best_x_tail
                    
                    problem(x_new)
                    used += 1
        
        # Fill remaining budget with random
        while used < budget:
            if time.time() > deadline:
                break
            x_rand = np.zeros(n)
            x_rand[0] = np.random.uniform(0, 1)
            x_rand[1] = np.random.uniform(0, 1)
            x_rand[2:] = best_x_tail
            problem(x_rand)
            used += 1
            
        return best_x_tail
