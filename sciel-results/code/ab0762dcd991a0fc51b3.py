import numpy as np
from itertools import product

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = 20
        # Ensure pop_size <= budget
        if pop_size > self.budget:
            pop_size = self.budget
            
        # Initialize population
        # Use Latin Hypercube or just Random? Random is fine.
        # To be distinct from "Grid", use Random.
        pop = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        
        # Store evaluated points and their objectives
        # We need to track all unique points evaluated to calculate HV correctly for selection
        # But for selection within the population, we just need the pop's HV.
        
        evaluated = {} # hash of x -> f
        current_pop = []
        
        # Evaluate initial population
        for i in range(pop_size):
            x = pop[i]
            key = tuple(np.round(x, 6)) # Use rounded key to avoid floating point issues in dict? 
            # Actually, better to just list and check distance? 
            # For simplicity, assume unique random points.
            f = problem(x)
            current_pop.append((x, f))
            
        used = pop_size
        
        # Main loop
        while used < self.budget:
            # Calculate hypervolume contribution for each point in current_pop
            # Reference point
            ref = np.array([11.0, 11.0])
            
            # Function to calculate hypervolume of a set of points
            def calc_hv(points, ref):
                # points: list of (f1, f2)
                # Filter dominated
                pts = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
                if not pts:
                    return 0.0
                # Sort by f1
                pts.sort(key=lambda x: x[0])
                hv = 0.0
                max_f2 = ref[1]
                for p in pts:
                    if p[1] < max_f2:
                        width = ref[0] - p[0]
                        height = max_f2 - p[1]
                        hv += width * height
                        max_f2 = p[1]
                return hv

            # We need to select parents. 
            # Simple approach: Keep the best half based on individual HV contribution?
            # Or just use non-dominated sorting?
            # Let's use a simple tournament selection on the whole population based on "quality".
            # Quality can be defined by how much HV they add if they were the only point? No.
            
            # Better: 
            # 1. Identify non-dominated set in current_pop.
            # 2. If all are dominated, pick the one with best f1+f2?
            
            # Let's just generate offspring from the entire population using mutation, 
            # and replace the worst points in the population based on HV contribution of the *set*.
            
            # Simpler Strategy:
            # 1. Generate one offspring from a randomly selected parent (or best parent).
            # 2. Evaluate it.
            # 3. If it improves the total HV of the population (or is non-dominated and adds to HV), 
            #    replace the point in the population with the lowest individual contribution.
            
            # This is "Hypervolume-based Evolutionary Algorithm".
            
            # Calculate individual contributions (approximate: HV of set - HV of set\{i})
            # This is O(N^2) or O(N^3) depending on impl. N=20 is small.
            
            total_hv = calc_hv([p[1] for p in current_pop], ref)
            
            contributions = []
            for i, (x, f) in enumerate(current_pop):
                others = [p[1] for j, p in enumerate(current_pop) if j != i]
                hv_without = calc_hv(others, ref)
                contrib = total_hv - hv_without
                contributions.append(contrib)
            
            # Find index with lowest contribution
            min_idx = np.argmin(contributions)
            
            # Select a parent: Randomly pick from top 50% contributors? Or just random?
            # Let's pick randomly from the population to maintain diversity.
            parent_idx = np.random.randint(len(current_pop))
            parent_x = current_pop[parent_idx][0]
            
            # Mutation: Gaussian
            sigma = 0.1 # Initial step size
            # Adaptive sigma? 
            # For ZDT1, we need to converge x[1:] to 0.
            # Standard deviation 0.1 might be too large or small.
            
            child_x = parent_x + np.random.normal(0, sigma, self.dim)
            child_x = np.clip(child_x, lo, hi)
            
            # Evaluate child
            child_f = problem(child_x)
            used += 1
            
            # Check if child is better than the min_idx point in terms of HV contribution
            # We need to see if adding child and removing min_idx improves HV
            # Or simply: if child is non-dominated, it's likely good.
            
            # Let's just replace if the new set has higher HV
            new_pop_set = [p[1] for p in current_pop]
            new_pop_set.remove(current_pop[min_idx][1])
            new_pop_set.append(child_f)
            
            new_hv = calc_hv(new_pop_set, ref)
            
            if new_hv > total_hv:
                current_pop[min_idx] = (child_x, child_f)
                total_hv = new_hv
            # Else, discard child. But we already used a budget.
            # This is a waste if we discard. 
            # To avoid waste, we should always keep the best point?
            # If we discard, the population doesn't change.
            # This is valid, but might converge slowly.
            
            # To ensure progress, maybe we should always replace the worst if the child is non-dominated?
            
        return None
