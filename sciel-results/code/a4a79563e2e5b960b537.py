"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.max_ns_size = 15  # Max size of non-dominated set

    def __call__(self, problem, bounds):
        start_time = time.time()
        # 10s safety margin before 300s limit
        deadline = start_time + 290.0 
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # Normalize range for mutation scaling
        ranges = hi - lo
        
        # Initialization
        initial_pop = 10
        used = 0
        NS = [] # List of (x, f)
        
        for _ in range(initial_pop):
            if used >= self.budget or time.time() > deadline:
                break
            x = lo + np.random.random(self.dim) * ranges
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            used += 1
            
            # Check if non-dominated
            is_dom = False
            for _, f_existing in NS:
                if np.all(f_existing <= f) and np.any(f_existing < f):
                    is_dom = True
                    break
            if not is_dom:
                # Remove points dominated by new x
                NS = [(xe, fe) for xe, fe in NS if not (np.all(f <= fe) and np.any(f < fe))]
                NS.append((x, f))
                
        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            if not NS:
                # Fallback if no valid points
                x = lo + np.random.random(self.dim) * ranges
                x = np.clip(x, lo, hi)
                f = np.array(problem(x))
                used += 1
                NS.append((x, f))
                continue
                
            # Select parent
            # Weighted selection: prefer points closer to ideal (0,0,0) roughly
            # Simple random is also fine for small sets, but let's try weighted by inverse normalized sum
            sums = [np.sum(fe / ref_point) for _, fe in NS]
            probs = 1.0 / (np.array(sums) + 1e-6)
            probs /= np.sum(probs)
            idx = np.random.choice(len(NS), p=probs)
            parent_x, parent_f = NS[idx]
            
            # Mutation: Adaptive sigma
            # Start with larger sigma, decrease over time
            progress = used / self.budget
            sigma = 0.15 * (1.0 - 0.5 * progress)
            
            x = parent_x + np.random.normal(0, sigma, self.dim) * ranges
            x = np.clip(x, lo, hi)
            
            f = np.array(problem(x))
            used += 1
            
            # Dominance checks
            # 1. If child dominates any in NS, remove them
            to_remove = []
            for i, (_, fe) in enumerate(NS):
                if np.all(f <= fe) and np.any(f < fe):
                    to_remove.append(i)
            
            if to_remove:
                for i in sorted(to_remove, reverse=True):
                    NS.pop(i)
                
            # 2. If any in NS dominates child, discard
            dominated = False
            for _, fe in NS:
                if np.all(fe <= f) and np.any(fe < f):
                    dominated = True
                    break
            
            if not dominated:
                NS.append((x, f))
                
                # 3. Truncate if too large
                if len(NS) > self.max_ns_size:
                    # Remove the point with the worst normalized sum
                    # Or simply the one with the largest sum of (f - ideal)
                    # Using normalized sum relative to ref point
                    scores = [np.sum(fe / ref_point) for _, fe in NS]
                    worst_idx = np.argmax(scores)
                    NS.pop(worst_idx)
                    
        # Return the non-dominated set (or just X, as per interface, but usually we return the best points)
        # The evaluator ignores the return value, but we must return something.
        # Returning the X coordinates of the NS is standard.
        if NS:
            return np.array([x for x, _ in NS])
        else:
            return np.array([])

# EVOLVE-BLOCK-END
