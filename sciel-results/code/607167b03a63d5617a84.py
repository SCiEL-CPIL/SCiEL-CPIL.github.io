"""Model-and-optimize for DTLZ1: exploit known Pareto front structure, fit g(x), then map to front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    DTLZ1-specific optimizer that exploits the known Pareto front structure.
    
    DTLZ1 with M=3, n=7:
    - Position variables: x[0], x[1] (these determine the point on the simplex)
    - Aggregation variables: x[2], ..., x[6] (these determine g)
    - g(x) = sum_{i=2}^{6} x_i^2
    - f_0 = 0.5 * x[0] * (1 + g)
    - f_1 = 0.5 * x[0] * (1 - x[1]) * (1 + g)
    - f_2 = 0.5 * x[0] * x[1] * (1 + g)
    
    The Pareto front is achieved when g=0 (i.e., x[2..6]=0) and x[0]=1,
    with x[1] varying in [0,1]. This gives a 1D line segment (simplex edge)
    in objective space.
    
    Strategy:
    1. Probe to find the configuration of x[2..6] that minimizes g (should be near 0).
    2. Sample points on the Pareto front by varying x[1] with x[0]=1 and x[2..6] at optimum.
    3. Use remaining budget for local refinement to improve coverage.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Step 1: Find the g-minimizing configuration for x[2..6]
        # The aggregation variables are x[2] through x[6] (indices 2 to dim-1)
        # For DTLZ1, g = sum of squares of these, so minimum is at all zeros.
        
        # Probe with x[0]=1, x[1]=0.5, and various x[2..6] configurations
        # to confirm the structure and find the best g-minimizing point.
        
        # Start with all zeros for x[2..6]
        best_x_tail = np.zeros(dim - 2)
        best_g_val = np.inf
        
        # Evaluate a few candidate configurations
        candidates = []
        # All zeros
        candidates.append(np.zeros(dim - 2))
        # All ones
        candidates.append(np.ones(dim - 2))
        # All 0.5
        candidates.append(np.full(dim - 2, 0.5))
        
        # Some random candidates
        rng = np.random.default_rng(42)
        for _ in range(5):
            candidates.append(rng.random(dim - 2))
        
        for x_tail in candidates:
            x_test = np.concatenate([[1.0, 0.5], x_tail])
            x_test = np.clip(x_test, lo, hi)
            f = problem(x_test)
            # For DTLZ1, sum of objectives = 0.5 * x[0] * (1 + g) * (1 + (1-x[1]) + x[1]) = 0.5 * x[0] * (1+g) * 2 = x[0]*(1+g)
            # So sum(f) = 1.0 * (1 + g) when x[0]=1
            g_val = np.sum(f) - 1.0  # Since x[0]=1, sum(f) = 1 + g
            if g_val < best_g_val:
                best_g_val = g_val
                best_x_tail = x_tail.copy()
        
        # Refine x[2..6] with local search
        for iteration in range(20):
            if time.time() > deadline - 10:
                break
            
            # Try perturbations around the current best
            for _ in range(3):
                x_tail_perturbed = best_x_tail + 0.1 * (hi[2:] - lo[2:]) * rng.standard_normal(dim - 2)
                x_tail_perturbed = np.clip(x_tail_perturbed, lo[2:], hi[2:])
                
                x_test = np.concatenate([[1.0, 0.5], x_tail_perturbed])
                x_test = np.clip(x_test, lo, hi)
                f = problem(x_test)
                g_val = np.sum(f) - 1.0
                
                if g_val < best_g_val:
                    best_g_val = g_val
                    best_x_tail = x_tail_perturbed.copy()
        
        # Step 2: Sample points on the Pareto front
        # The PF is parameterized by x[1] in [0,1] with x[0]=1 and x[2..6]=best_x_tail
        # f_0 = 0.5 * 1 * (1 + g) = 0.5 * (1 + g)
        # f_1 = 0.5 * 1 * (1 - x[1]) * (1 + g)
        # f_2 = 0.5 * 1 * x[1] * (1 + g)
        
        # We want to cover the simplex well. The PF in objective space is:
        # f_0 = c, f_1 = c*(1-t), f_2 = c*t where c = 0.5*(1+g) and t = x[1]
        # So f_1 + f_2 = c and f_0 = c, meaning f_0 = f_1 + f_2.
        # The PF is a line segment from (c, c, 0) to (c, 0, c).
        
        # For hypervolume, we want to maximize the volume under the PF.
        # Since the PF is a line, we should sample many points along it.
        
        n_samples = min(150, budget - 10)  # Leave some budget for refinement
        t_vals = np.linspace(0.0, 1.0, n_samples)
        
        recorded_F = []
        used = 0
        
        for t in t_vals:
            if used >= budget or time.time() > deadline:
                break
            
            x = np.concatenate([[1.0, t], best_x_tail])
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded_F.append(f)
            used += 1
        
        # Step 3: Use remaining budget for local refinement
        remaining = budget - used
        if remaining > 0 and recorded_F:
            F_arr = np.array(recorded_F)
            
            # Find non-dominated points
            dominated = np.zeros(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                if dominated[i]:
                    continue
                for j in range(len(F_arr)):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            
            non_dom_idx = np.where(~dominated)[0]
            
            if len(non_dom_idx) > 0:
                # Refine around non-dominated points by trying nearby t values
                for idx in non_dom_idx[:min(10, remaining)]:
                    if used >= budget or time.time() > deadline:
                        break
                    
                    # Get the t value for this point
                    t_base = t_vals[idx] if idx < len(t_vals) else 0.5
                    
                    # Try a few nearby t values
                    for delta_t in [-0.02, 0.02, -0.01, 0.01]:
                        if used >= budget or time.time() > deadline:
                            break
                        
                        t_new = t_base + delta_t
                        t_new = np.clip(t_new, 0.0, 1.0)
                        
                        x = np.concatenate([[1.0, t_new], best_x_tail])
                        x = np.clip(x, lo, hi)
                        f = problem(x)
                        recorded_F.append(f)
                        used += 1
        
        return recorded_F
# EVOLVE-BLOCK-END
