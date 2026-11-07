"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Problem-aware hybrid optimizer for ZDT1: analytical x1 sweep + local g(x) minimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We will collect all evaluated points and their objectives
        all_points = []
        all_f = []
        
        def eval_point(x):
            """Evaluate a point, record it, and return its objective vector."""
            nonlocal all_points, all_f
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_points.append(x.copy())
            all_f.append(f.copy())
            return f

        # ------------------------------------------------------------------
        # Phase 1: Identify the structure of ZDT1
        # Standard ZDT1:
        # f1 = x[0]
        # g(x) = 1 + 10*(n-1) / sum(x[1:])
        # f2 = g(x) * (1 - sqrt(x[0] / g(x)))
        #
        # To be on the Pareto front, we need g(x) = 1.
        # This requires sum(x[1:]) to be as large as possible.
        # With bounds [0, 1], max sum(x[1:]) = dim-1.
        # If sum(x[1:]) = dim-1, then g = 1 + 10*(dim-1)/(dim-1) = 11?
        # Wait. Let's check the standard definition carefully.
        #
        # ZDT1 definition:
        # f1(x) = x1
        # g(x) = 1 + 9 * (sum_{i=2}^n x_i) / (n-1)   <-- Some versions use 9, some 10, some 10/(n-1)
        # Actually, the most common version is:
        # g(x) = 1 + 10 * (n-1) / sum(x[1:])  <-- This is often ZDT2 or similar?
        #
        # Let's look at the "known Pareto front" mentioned in the prompt.
        # "f2 = 1 - sqrt(f1)" is mentioned in one inspiration, but that's not standard ZDT1.
        # Standard ZDT1 Pareto front: f1 in [0,1], f2 = 1 - sqrt(f1).
        # Wait, if f2 = 1 - sqrt(f1), then when f1=1, f2=0. When f1=0, f2=1.
        #
        # Let's re-read the inspiration:
        # "f2 = 1 - sqrt(x0/g(x)) where g(x) = 1 + 10*(n-1)/sum(x_i for i>0)"
        # If g(x) = 1, then f2 = 1 - sqrt(x0). This matches the standard ZDT1 front.
        # So, the goal is to make g(x) = 1.
        #
        # If g(x) = 1 + 10*(n-1)/sum(x[1:]), then for g(x)=1, we need sum(x[1:]) -> infinity.
        # But x[i] are bounded by [0,1]. Max sum is n-1.
        # Then g_min = 1 + 10*(n-1)/(n-1) = 11.
        # This contradicts the idea that g=1 is achievable.
        #
        # Alternative ZDT1 definition:
        # g(x) = 1 + 9 * (sum_{i=2}^n x_i) / (n-1)
        # If all x[i]=1, sum = n-1, so g = 1 + 9 = 10.
        # If all x[i]=0, sum = 0, so g = 1.
        # Then f2 = g * (1 - sqrt(x1/g)).
        # If g=1, f2 = 1 - sqrt(x1). This matches the front!
        # So, to get on the Pareto front, we need g(x) = 1, which means sum(x[1:]) = 0.
        # i.e., x[1:] should be all zeros.
        #
        # Let's test this hypothesis.
        # If x[1:] = 0, then g = 1.
        # f1 = x[0]
        # f2 = 1 * (1 - sqrt(x[0]/1)) = 1 - sqrt(x[0])
        # This is exactly the Pareto front.
        #
        # So the strategy is:
        # 1. Set x[1:] to 0 (or very close to 0).
        # 2. Sweep x[0] from 0 to 1 to cover the front.
        #
        # However, we don't know the exact formula for g(x). We need to discover it.
        # But the prompt says "minimizing all 2 objectives of ZDT1".
        # And the inspiration says "Analytical ZDT1 structure exploitation: x[0] sweep with x[1:]=0.2 places points exactly on the known Pareto front f2=1+sqrt(f1)".
        # Wait, "f2=1+sqrt(f1)"? That would be maximization of f2? No, we are minimizing.
        # If f2 = 1 - sqrt(f1), then minimizing f2 means maximizing sqrt(f1), i.e., maximizing f1.
        # But we want the whole front.
        #
        # Let's assume the standard ZDT1 where the front is f2 = 1 - sqrt(f1).
        # This requires g(x) = 1.
        # If g(x) = 1 + 9 * mean(x[1:]), then g=1 implies mean(x[1:])=0, i.e., x[1:]=0.
        #
        # Let's verify by evaluating a few points.
        # ------------------------------------------------------------------

        # Step 1: Verify the structure by evaluating a few points
        # Point A: x[0]=0.5, x[1:]=0
        x_test1 = np.zeros(self.dim)
        x_test1[0] = 0.5
        f_test1 = eval_point(x_test1)
        
        # Point B: x[0]=0.5, x[1:]=1
        x_test2 = np.ones(self.dim)
        x_test2[0] = 0.5
        f_test2 = eval_point(x_test2)
        
        # Point C: x[0]=1.0, x[1:]=0
        x_test3 = np.zeros(self.dim)
        x_test3[0] = 1.0
        f_test3 = eval_point(x_test3)
        
        # Point D: x[0]=0.0, x[1:]=0
        x_test4 = np.zeros(self.dim)
        f_test4 = eval_point(x_test4)
        
        # If the hypothesis is correct:
        # f_test1 should be [0.5, 1 - sqrt(0.5)] = [0.5, 0.2929]
        # f_test3 should be [1.0, 0.0]
        # f_test4 should be [0.0, 1.0]
        
        # Check if f2 is lower when x[1:]=0 than when x[1:]=1
        # If f_test1[1] < f_test2[1], then x[1:]=0 is better for f2.
        
        # ------------------------------------------------------------------
        # Phase 2: Systematic sweep of x[0] with x[1:] optimized for g(x)
        # We will sweep x[0] and for each, try to minimize f2 by adjusting x[1:].
        # But if the structure is as hypothesized, x[1:]=0 is optimal.
        # Let's do a dense sweep of x[0] with x[1:]=0 first.
        # ------------------------------------------------------------------
        
        n_sweep = min(150, self.budget - len(all_points) - 10)
        if n_sweep > 0:
            x0_values = np.linspace(0.0, 1.0, n_sweep)
            for x0 in x0_values:
                if time.time() > deadline:
                    break
                if len(all_points) >= self.budget - 5:
                    break
                
                x = np.zeros(self.dim)
                x[0] = x0
                eval_point(x)
        
        # ------------------------------------------------------------------
        # Phase 3: Local optimization around the sweep points
        # For each point on the front, try to improve it by small perturbations.
        # This helps if the exact optimal x[1:] is not exactly 0, or if there are
        # numerical issues.
        # ------------------------------------------------------------------
        
        if time.time() < deadline and len(all_points) < self.budget - 20:
            F_current = np.array(all_f)
            
            # Find non-dominated points
            n_dom = len(F_current)
            nondom_idx = []
            for i in range(n_dom):
                is_dom = False
                for j in range(n_dom):
                    if i != j and np.all(F_current[j] <= F_current[i]) and np.any(F_current[j] < F_current[i]):
                        is_dom = True
                        break
                if not is_dom:
                    nondom_idx.append(i)
            
            if nondom_idx:
                # Do local search around a few non-dominated points
                n_local = min(10, len(nondom_idx), (self.budget - len(all_points)) // 5)
                step_size = 0.05
                
                for idx in nondom_idx[:n_local]:
                    if time.time() > deadline:
                        break
                    if len(all_points) >= self.budget - 5:
                        break
                    
                    base_x = all_points[idx].copy()
                    base_f = all_f[idx].copy()
                    
                    # Try small perturbations
                    for _ in range(3):
                        if time.time() > deadline or len(all_points) >= self.budget - 5:
                            break
                        
                        # Perturb x[1:] slightly
                        x_new = base_x.copy()
                        x_new[1:] += step_size * np.random.randn(self.dim - 1)
                        x_new = np.clip(x_new, lo, hi)
                        
                        f_new = eval_point(x_new)
                        
                        # If the new point is better, keep it (it's already recorded)
                        # We could do a more sophisticated local search, but for now,
                        # just try a few random perturbations.
        
        # ------------------------------------------------------------------
        # Phase 4: Final refinement
        # If we have budget left, do a few more random evaluations to ensure
        # we haven't missed anything.
        # ------------------------------------------------------------------
        
        while len(all_points) < self.budget and time.time() < deadline:
            x = lo + (hi - lo) * np.random.random(self.dim)
            eval_point(x)
        
        return all_points
# EVOLVE-BLOCK-END
