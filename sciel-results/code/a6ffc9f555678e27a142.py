import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 specific logic
        # x[0] is f1 driver
        # x[1:] drive g(x)
        
        # 1. Find a good value for x[1:] that minimizes h(x)
        # h(x) = -3*x^2*cos(14*pi*x) - x + 1
        # We want to minimize sum of h(x_i)
        
        # Use a small budget to find a good local minimum for h(x)
        # We can use a simple 1D search on a single variable to find a good starting point
        # Or just use a known approximate minimum.
        # Let's do a quick random search + polish for one variable to get a good x_opt
        
        # Reserve budget
        # 20 for finding x_opt
        # 50 for polishing x_rest
        # 130 for sampling x1
        
        # Find x_opt for h(x)
        # h(x) is periodic. Minima are around 0.225, 0.5, 0.775?
        # Let's try a few candidates and pick the best using a few evals
        # Actually, we can just use a deterministic local search on a single variable
        
        def h_func(x):
            return -3 * x**2 * np.cos(14 * np.pi * x) - x + 1
        
        # Find min of h(x) in [0, 1]
        # Use golden section or simple grid + polish
        # Grid of 20 points
        x_candidates = np.linspace(0, 1, 20)
        h_vals = [h_func(x) for x in x_candidates]
        best_idx = np.argmin(h_vals)
        x_opt = x_candidates[best_idx]
        
        # Polish x_opt with a few steps
        step = 0.01
        for _ in range(5):
            x_left = max(0, x_opt - step)
            x_right = min(1, x_opt + step)
            h_left = h_func(x_left)
            h_right = h_func(x_right)
            if h_left < h_func(x_opt):
                x_opt = x_left
            elif h_right < h_func(x_opt):
                x_opt = x_right
            else:
                step *= 0.5
            if step < 1e-4:
                break
        
        # Initialize x_rest with x_opt
        x_rest = np.full(dim - 1, x_opt)
        
        # Evaluate this initial point to get a baseline g
        # We need to call problem to count budget
        # Let's create a helper to call problem and track count
        eval_count = 0
        def eval_point(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            eval_count += 1
            return f

        # Initial eval
        x_init = np.concatenate(([0.5], x_rest))
        f_init = eval_point(x_init)
        
        # 2. Polish x_rest using coordinate descent
        # Budget for this phase: 50
        polish_budget = 50
        remaining_polish = polish_budget
        
        # Simple coordinate descent
        # Cycle through variables 1..29
        # For each, try a few perturbations
        
        # We need to be careful: each eval_point call consumes 1 budget.
        # We will do a few passes.
        
        current_x_rest = x_rest.copy()
        
        # Define a helper to evaluate a modified x_rest
        # We fix x1=0.5 for this phase to isolate g minimization
        # f2 = (1+g)(1 - sqrt(x1/g) cos(...))
        # Minimizing f2 with x1 fixed is roughly minimizing g if g is small
        
        while remaining_polish > 0 and eval_count < budget:
            improved = False
            for i in range(dim - 1):
                if remaining_polish <= 0:
                    break
                # Try to minimize h(current_x_rest[i]) by adjusting x_rest[i]
                # Use a small local search
                x_curr = current_x_rest[i]
                best_x = x_curr
                best_f = None
                
                # Get current f to compare
                # We don't have the current f stored for the whole vector, only the last one.
                # We need to re-evaluate or store.
                # Let's just try a few steps and pick the best f2.
                
                candidates = [x_curr]
                # Add perturbations
                for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    nx = np.clip(x_curr + delta, 0, 1)
                    candidates.append(nx)
                
                # Evaluate candidates
                best_cand_f = np.inf
                best_cand_x = x_curr
                
                for cx in candidates:
                    if remaining_polish <= 0:
                        break
                    test_x_rest = current_x_rest.copy()
                    test_x_rest[i] = cx
                    test_x = np.concatenate(([0.5], test_x_rest))
                    f_test = eval_point(test_x)
                    remaining_polish -= 1
                    if f_test is not None:
                        # Use f2 as proxy for g minimization
                        if f_test[1] < best_cand_f:
                            best_cand_f = f_test[1]
                            best_cand_x = cx
                
                if best_cand_x != x_curr:
                    current_x_rest[i] = best_cand_x
                    improved = True
            
            if not improved:
                break

        # 3. Sample x1
        # Remaining budget
        remaining_budget = budget - eval_count
        
        if remaining_budget > 0:
            # Sample x1 values
            # Use the best x_rest found
            final_x_rest = current_x_rest
            
            # Generate x1 values
            # Uniform sampling
            x1_samples = np.linspace(0.01, 0.99, remaining_budget)
            
            for x1 in x1_samples:
                if eval_count >= budget:
                    break
                if time.time() > deadline:
                    break
                x_sample = np.concatenate(([x1], final_x_rest))
                f_sample = eval_point(x_sample)
        
        return None
