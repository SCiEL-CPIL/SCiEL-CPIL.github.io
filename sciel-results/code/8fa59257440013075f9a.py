"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer using a hybrid strategy:
    1. Initial exploration with Latin Hypercube Sampling.
    2. Iterative refinement using a simple acquisition function (Expected Improvement-like)
       approximated via random restarts of local search, but guided by the current 
       non-dominated set to fill gaps in the objective space.
    
    However, given the strict budget and the known structure of ZDT1, a pure black-box 
    BO might be too slow to converge to the exact front in 200 evals. 
    The current program scores 1.19 by exploiting the analytic front. 
    A 'different' strategy that is still highly effective: 
    **Adaptive Density Sampling with Local Gradient Descent**.
    
    We will:
    1. Use a small number of initial random points.
    2. Identify the current non-dominated front.
    3. For each gap in the front (in objective space), spawn a local minimization 
       problem to find a point that minimizes the distance to the 'ideal' point 
       while staying on the front. 
       
    Actually, a simpler and more robust 'different' family is **Evolutionary Algorithm 
    with Elitism and Crowding** (NSGA-II lite). This is a standard, distinct design family 
    from the analytic grid. It does not assume the front location.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # State
        X = []
        F = []
        evaluations_used = 0
        
        def evaluate(x):
            nonlocal evaluations_used
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped))
            X.append(x_clipped)
            F.append(f)
            evaluations_used += 1
            return f
            
        # 1. Initial Population: Latin Hypercube Sampling for diversity
        pop_size = min(20, self.budget // 5)
        if pop_size < 10:
            pop_size = 10
            
        # Generate LHS points
        indices = np.arange(pop_size)
        for i in range(self.dim):
            # Shuffle indices for each dimension to create LHS
            np.random.shuffle(indices)
            # This is a simplistic LHS. Let's do it properly.
            
        # Proper LHS
        lhs_points = []
        for i in range(pop_size):
            x = np.zeros(self.dim)
            for d in range(self.dim):
                # Random offset within the bin
                x[d] = (i + np.random.random()) / pop_size
            lhs_points.append(x)
            
        lhs_points = np.array(lhs_points) * span + lo
        
        for x in lhs_points:
            if evaluations_used >= self.budget or time.time() > deadline:
                break
            evaluate(x)
            
        # 2. Evolutionary Loop
        # We maintain a set of non-dominated points.
        # To maximize hypervolume, we want to cover the front densely.
        # Strategy: Pick a parent from the non-dominated set, mutate it, 
        # and if the child is non-dominated or dominates others, keep it.
        
        # Helper: Check if a point is non-dominated by the current set
        def is_non_dominated(f_new, F_set):
            # F_set is a list of numpy arrays
            for f_old in F_set:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return False
            return True
            
        # Helper: Remove dominated points from a set
        def filter_non_dominated(F_list):
            if not F_list:
                return []
            nd = []
            for i, f_i in enumerate(F_list):
                dominated = False
                for j, f_j in enumerate(F_list):
                    if i == j:
                        continue
                    if np.all(f_j <= f_i) and np.any(f_j < f_i):
                        dominated = True
                        break
                if not dominated:
                    nd.append(f_i)
            return nd
            
        # Main Loop
        while evaluations_used < self.budget and time.time() < deadline:
            # Get current non-dominated front
            F_nd = filter_non_dominated(F)
            
            if not F_nd:
                # Should not happen if we have points
                break
                
            # Select a parent: 
            # To improve hypervolume, we should target the largest gaps in the objective space.
            # A simple heuristic: pick the point with the largest 'crowding distance' 
            # or simply pick a random non-dominated point and mutate.
            # Given the small budget, random mutation of ND points is effective.
            
            # Pick a random index from the non-dominated set
            # We need to map F_nd back to indices in F
            # This is inefficient but N is small (<=200)
            
            # Alternative: Just pick a random point from the entire population? 
            # No, picking from ND is better.
            
            # Let's find the indices of the ND points
            nd_indices = []
            for i, f_i in enumerate(F):
                if f_i in F_nd:
                    nd_indices.append(i)
            
            if not nd_indices:
                 # Fallback
                 parent_idx = np.random.randint(0, len(F))
            else:
                 parent_idx = nd_indices[np.random.randint(0, len(nd_indices))]
            
            parent_x = X[parent_idx]
            parent_f = F[parent_idx]
            
            # Mutation: Gaussian noise
            # Scale noise based on remaining budget? 
            # Start with large noise, decrease?
            # Simple: fixed sigma = 0.1 * span
            sigma = 0.1
            mutant_x = parent_x + np.random.normal(0, sigma, self.dim)
            mutant_x = np.clip(mutant_x, lo, hi)
            
            # Evaluate mutant
            f_mut = evaluate(mutant_x)
            
            # Optional: Local improvement? 
            # If the mutant is non-dominated, we could try to polish it with a few gradient steps?
            # But we don't have gradients. 
            # We can do a simple local search: try small steps in each dimension.
            # This consumes budget. 
            # With 200 budget, we can't afford expensive local search for every point.
            
            # Let's just stick to the evolutionary step.
            
        return F
# EVOLVE-BLOCK-END
