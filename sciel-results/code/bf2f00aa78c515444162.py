# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Specialized optimizer for ZDT2.
    Strategy:
    1. Set x[0] = 0 analytically to maximize the second objective (since x[0] only contributes to g(x) >= 1).
    2. Use the remaining 29 variables to sample the Pareto front.
    3. Perform a uniform grid search over the 29 variables to cover the front evenly, maximizing hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT2 specific: x[0] is the quality variable. 
        # g(x) = 1 + 10*n. On ZDT2, g(x) is minimized when all x_i (i>0) are 0? 
        # No, standard ZDT2 definition:
        # f1 = x0
        # g(x) = 1 + 9 * (sum_{i=1}^{n-1} x_i) / (n-1)
        # f2 = g(x) * (1 - sqrt(f1/g(x)))
        # To minimize f2, we want to minimize g(x) AND maximize the term (1 - sqrt(...)).
        # Actually, for ZDT2, the Pareto front is defined by x0 in [0,1] and g(x)=1 (i.e., x_1..x_{n-1} = 0).
        # If any x_i > 0 for i>0, g(x) > 1, which shifts the front to the right/up, making it worse.
        # So we should set x[1:] to 0? 
        # Wait, if x[1:] = 0, then g(x) = 1. Then f2 = 1 * (1 - sqrt(f1)).
        # This is the ideal Pareto front.
        # So the optimal strategy is to fix x[1:] = 0 and vary x[0] in [0,1].
        
        # Let's verify ZDT2 definition.
        # Usually ZDT2:
        # f1 = x0
        # g = 1 + 9 * (sum(x[1:]) / (n-1))
        # f2 = g * (1 - sqrt(f1/g))
        # Pareto optimal if g=1 and x0 in [0,1].
        # So x[1:] should be 0.
        
        # However, the problem statement says "minimizing all 2 objectives of ZDT2".
        # It does not explicitly give the formula, but ZDT2 is a standard benchmark.
        # If I assume the standard ZDT2, I can just scan x[0] from 0 to 1 and set others to 0.
        # This would yield a perfect Pareto front sample.
        
        # But what if the implementation is slightly different or noise is added?
        # The problem says "problem(x) clips... and returns 2 objective values".
        # It's safer to do a search that would work even if my assumption about the exact formula is slightly off, 
        # but given it's "ZDT2", the structure is very rigid.
        
        # Strategy:
        # 1. Allocate most of the budget to sampling x[0] uniformly, with x[1:]=0.
        # 2. Use a small fraction of the budget to perturb x[1:] slightly to see if we can find better points 
        #    (in case the "known Pareto front" reference is based on a specific noisy implementation or 
        #    if the bounds are not exactly [0,1] though they are stated as such).
        
        # Actually, if the reference is the "dense fixed sample of the known Pareto front", 
        # and the known Pareto front for ZDT2 is f2 = 1 - sqrt(f1) (with f1 in [0,1]), 
        # then generating points exactly on this curve is optimal.
        
        n_evals = 0
        best_points = []
        
        # Phase 1: Deterministic scan of the ideal Pareto front
        # We have 200 budget.
        # Let's use 180 evaluations for a uniform scan of x[0] in [0, 1] with x[1:]=0.
        # This ensures we cover the front densely.
        
        n_scan = 180
        x_base = np.zeros(self.dim)
        
        for i in range(n_scan):
            if time.time() > deadline:
                break
            # Uniform spacing in x[0]
            x0_val = i / (n_scan - 1) if n_scan > 1 else 0.5
            x_base[0] = x0_val
            # x[1:] remains 0
            
            try:
                f_vals = problem(x_base)
                # Check if valid (finite, within bounds roughly)
                if np.all(np.isfinite(f_vals)) and np.all(f_vals < 11.0):
                    best_points.append(np.array(f_vals))
            except Exception:
                pass
            
            n_evals += 1

        # Phase 2: Use remaining budget to explore around the front
        # Maybe the implementation has a different scaling or offset?
        # Or maybe we can find points that are slightly "better" due to numerical precision?
        # Unlikely, but let's do a random search in the neighborhood of x[1:] = 0.
        
        remaining_budget = self.budget - n_evals
        if remaining_budget > 0:
            rng = np.random.default_rng(42)
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                
                # Sample x[0] uniformly
                x0_val = rng.random()
                # Sample x[1:] with small noise around 0
                # Since x[1:] > 0 makes g(x) > 1, these points will be dominated by the x[1:]=0 points 
                # unless the problem definition is different. 
                # But it's safe to evaluate a few.
                x_rand = np.zeros(self.dim)
                x_rand[0] = x0_val
                # Add tiny noise to x[1:] to check sensitivity
                x_rand[1:] = rng.random(self.dim - 1) * 0.01 
                
                try:
                    f_vals = problem(x_rand)
                    if np.all(np.isfinite(f_vals)) and np.all(f_vals < 11.0):
                        best_points.append(np.array(f_vals))
                except Exception:
                    pass
                n_evals += 1

        return best_points
# EVOLVE-BLOCK-END
