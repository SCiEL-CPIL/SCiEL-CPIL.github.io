import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: SGD-based gradient/Hessian estimation
        # For sphere f(x) = ||x - c||^2 + offset, we can estimate:
        # - Gradient at origin: grad_i = (f(e_i*h) - f(-e_i*h)) / (2*h)
        # - Hessian diagonal: hess_ii = (f(e_i*h) - 2*f(0) + f(-e_i*h)) / h^2
        # For a sphere, Hessian should be ~2*I, so we can use gradient to find center
        # c_i = -grad_i / (2 * hess_mean)
        
        # Use multiple random perturbations for robustness
        rng_init = np.random.default_rng(42)
        h = 0.5  # Perturbation size
        
        # Evaluate at origin first
        f0 = evaluate(np.zeros(dim))
        
        # Estimate gradient using central differences at several random points
        # and average them (for sphere, gradient is linear: grad(x) = 2*(x-c))
        # So we can estimate c from grad(0) = -2*c => c = -grad(0)/2
        
        # Central difference for gradient at origin
        grad_est = np.zeros(dim)
        hess_diag = np.zeros(dim)
        
        for i in range(dim):
            e_pos = np.zeros(dim); e_pos[i] = h
            e_neg = np.zeros(dim); e_neg[i] = -h
            fp = evaluate(e_pos)
            fn = evaluate(e_neg)
            grad_est[i] = (fp - fn) / (2 * h)
            hess_diag[i] = (fp - 2 * f0 + fn) / (h * h)
        
        # For sphere, Hessian should be approximately constant (2*I for standard)
        # Use median of Hessian diagonal as robust estimate
        hess_mean = np.median(hess_diag)
        
        if hess_mean > 1e-8:
            # Center estimate: c = -grad / (2 * hess_mean)
            x_center = -grad_est / (2.0 * hess_mean)
            x_center = np.clip(x_center, -5, 5)
            f_center = evaluate(x_center)
            
            if f_center < best_f:
                best_x = x_center
                best_f = f_center
                # Refine with a few more steps if we have budget
                if budget - calls_made > 10:
                    # Do a few Newton-like steps using local gradient
                    for _ in range(3):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        # Estimate gradient at current best
                        h_refine = 0.1
                        grad_local = np.zeros(dim)
                        for j in range(dim):
                            ep = best_x.copy(); ep[j] += h_refine
                            en = best_x.copy(); en[j] -= h_refine
                            fp = evaluate(ep)
                            fn = evaluate(en)
                            grad_local[j] = (fp - fn) / (2 * h_refine)
                        
                        # Step in negative gradient direction with damping
                        step_size = 0.5
                        x_new = np.clip(best_x - step_size * grad_local / hess_mean, -5, 5)
                        f_new = evaluate(x_new)
                        
                        if f_new < best_f:
                            best_x = x_new
                            best_f = f_new
                        else:
                            break

        # Phase 2: Polish with (1+1)-ES
        # Start from best_x
        x = best_x.copy()
        fx = best_f
        
        # Adaptive sigma
        # Start moderate since we may still be far from optimum
        sigma = 0.3
        rng = np.random.default_rng(0)
        
        # Track success for sigma adaptation
        window_size = 15
        successes = []
        
        while calls_made < budget and time.time() < deadline:
            # Generate random direction
            z = rng.standard_normal(dim)
            
            # Step
            y = np.clip(x + sigma * z, -5, 5)
            fy = evaluate(y)
            
            if fy < fx:
                x = y
                fx = fy
                successes.append(1)
                # Increase sigma moderately
                sigma *= 1.15
            else:
                successes.append(0)
                # Decrease sigma
                sigma *= 0.85
                
            # Adapt sigma based on recent success rate
            if len(successes) >= window_size:
                recent_success_rate = np.mean(successes[-window_size:])
                # If success rate is too high, increase sigma more
                # If too low, decrease more
                if recent_success_rate > 0.65:
                    sigma *= 1.1
                elif recent_success_rate < 0.25:
                    sigma *= 0.9
                successes = [] # Reset window
                
            # If sigma gets too small, we might be stuck or converged
            if sigma < 1e-12:
                break

        return best_x, best_f

    def _simple_es(self, func, best_x, best_f):
        # Fallback for very small budgets
        calls_made = 0
        rng = np.random.default_rng(0)
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        
        start_time = time.time()
        deadline = start_time + 290
        
        while calls_made < self.budget and time.time() < deadline:
            z = rng.standard_normal(self.dim)
            y = np.clip(x + sigma * z, -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.15
            else:
                sigma *= 0.85
                
        return x, fx
