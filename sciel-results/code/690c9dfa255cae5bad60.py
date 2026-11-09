# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 exploiting separability and Pareto front geometry."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Track all evaluations
        all_x = []
        all_f = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_x.append(x)
            all_f.append(f)
            eval_count += 1
            return x, f

        def is_deadline():
            return time.time() > deadline

        # Phase 1: Grid-based exploration of the WFG4 Pareto front
        # WFG4 has a curved Pareto front. We sample along the front using
        # the known structure: the front lies in a specific region.
        
        # First, do a coarse random search to understand the landscape
        n_random = 30
        for i in range(n_random):
            if is_deadline() or eval_count >= budget:
                break
            x = lo + (hi - lo) * np.random.rand(dim)
            evaluate(x)

        # Phase 2: Exploit WFG4 structure
        # WFG4: f1 = x1, f2 = (1+0.1*(f1+f3))*g, f3 = (1+0.1*(f1+f2))*g
        # where g = 1 + 10*(n-1) + sum(x_i^2 - 2*cos(20*pi*x_i + 2*pi*f1/(n-1)))
        # The Pareto front has g=1, so x_i^2 - 2*cos(...) = 0 for all i>0
        # This means x_i is near 0 or specific values.
        
        # Key insight: on the Pareto front, x1 varies and determines the trade-off.
        # For each x1, the other variables should minimize g.
        
        # Strategy: Sweep x1, and for each x1, optimize the remaining variables
        # to minimize g (which minimizes both f2 and f3 simultaneously).
        
        # Get best f values so far
        if all_f:
            F = np.array(all_f)
        else:
            F = np.zeros((0, 3))

        # Phase 2a: Sweep x1 with local optimization of other variables
        n_sweep = 40
        x1_range = np.linspace(lo[0], hi[0], n_sweep)
        
        for x1_val in x1_range:
            if is_deadline() or eval_count >= budget:
                break
            
            # Initialize: x1 fixed, other vars at 0 (common Pareto front location)
            x = np.zeros(dim)
            x[0] = x1_val
            # Set other variables to 0 first
            for i in range(1, dim):
                x[i] = 0.0
            
            x, f = evaluate(x)
            
            # Local refinement: perturb each other variable slightly
            for _ in range(3):  # A few refinement steps
                if is_deadline() or eval_count >= budget:
                    break
                for i in range(1, dim):
                    if is_deadline() or eval_count >= budget:
                        break
                    # Try small perturbations
                    for delta in [0.01, -0.01, 0.05, -0.05]:
                        if is_deadline() or eval_count >= budget:
                            break
                        x_try = x.copy()
                        x_try[i] = np.clip(x_try[i] + delta * (hi[i] - lo[i]), lo[i], hi[i])
                        x_try, f_try = evaluate(x_try)
                        # If better in all objectives, keep it
                        if np.all(f_try <= f):
                            x, f = x_try, f_try
                            break
                        # If better in at least one and not worse in others, consider
                        elif np.any(f_try < f) and np.all(f_try <= f + 1e-6):
                            # Keep as candidate but don't replace
                            pass

        # Phase 3: Non-dominated sorting and targeted refinement
        if all_f:
            F = np.array(all_f)
            X = np.array(all_x)
            
            # Find non-dominated points
            n_points = len(F)
            dominated = np.zeros(n_points, dtype=bool)
            for i in range(n_points):
                if dominated[i]:
                    continue
                for j in range(n_points):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            nd_indices = np.where(~dominated)[0]
            
            # Refine around non-dominated points
            for idx in nd_indices[:15]:  # Limit to 15 best
                if is_deadline() or eval_count >= budget:
                    break
                
                x_center = X[idx].copy()
                f_center = F[idx].copy()
                
                # CMA-ES-like local search
                sigma = 0.05
                for gen in range(5):
                    if is_deadline() or eval_count >= budget:
                        break
                    # Generate perturbations
                    for _ in range(4):
                        if is_deadline() or eval_count >= budget:
                            break
                        perturbation = np.random.randn(dim) * sigma
                        x_try = np.clip(x_center + perturbation, lo, hi)
                        x_try, f_try = evaluate(x_try)
                        
                        # If non-dominated, update center
                        if np.all(f_try <= f_center) and np.any(f_try < f_center):
                            x_center = x_try
                            f_center = f_try
                        elif np.all(f_center <= f_try) and np.any(f_center < f_try):
                            # Current is better, shrink sigma
                            sigma *= 0.8
                        else:
                            # Incomparable, keep current
                            pass
                    
                    sigma *= 0.7

        # Phase 4: Final targeted search in promising regions
        # Sample along the expected Pareto front more densely
        if not is_deadline() and eval_count < budget:
            # Based on WFG4 structure, the Pareto front has f1 in [0,1] roughly
            # and f2, f3 increasing as f1 increases
            n_final = min(30, budget - eval_count)
            for i in range(n_final):
                if is_deadline() or eval_count >= budget:
                    break
                
                # Interpolate along the front
                t = i / max(n_final - 1, 1)
                x1_val = lo[0] + t * (hi[0] - lo[0])
                x = np.zeros(dim)
                x[0] = x1_val
                
                # Try different configurations for other variables
                configs = [
                    np.zeros(dim - 1),
                    np.full(dim - 1, 0.1),
                    np.full(dim - 1, -0.1),
                ]
                best_x = x.copy()
                best_f = None
                
                for config in configs:
                    x_try = x.copy()
                    x_try[1:] = config
                    x_try, f_try = evaluate(x_try)
                    if best_f is None or np.all(f_try <= best_f):
                        best_x = x_try
                        best_f = f_try
                
                # One more refinement
                if not is_deadline() and eval_count < budget:
                    for _ in range(2):
                        perturbation = np.random.randn(dim) * 0.02
                        x_try = np.clip(best_x + perturbation, lo, hi)
                        x_try, f_try = evaluate(x_try)
                        if best_f is None or np.all(f_try <= best_f):
                            best_x = x_try
                            best_f = f_try

        return all_x
# EVOLVE-BLOCK-END
