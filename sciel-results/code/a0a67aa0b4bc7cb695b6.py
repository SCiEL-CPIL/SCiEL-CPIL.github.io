import numpy as np
import time
from scipy.special import comb

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit
        
        n = self.dim
        budget = self.budget
        
        # 1. Global Initialization Phase: Use Halton sequence for low-discrepancy sampling
        # This provides better coverage of the [-5, 5]^n space than pure random sampling
        # and is crucial for finding good basins in multimodal functions like Rastrigin/Gallagher.
        num_init = 20
        if budget < num_init:
            num_init = budget
            
        # Generate Halton sequence points
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        points = []
        for i in range(1, num_init + 1):
            p = np.array([halton(i, b) for b in range(2, n + 2)])
            # Map [0, 1]^n to [-5, 5]^n
            points.append(20 * p - 5)
        
        # Evaluate initial points
        best_x = None
        best_f = np.inf
        used_budget = 0
        
        for p in points:
            if used_budget >= budget:
                break
            if time.time() > deadline:
                break
            f_val = func(p)
            used_budget += 1
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()
        
        if best_x is None:
            # Fallback if no budget left or error
            return np.zeros(n), 0.0

        # 2. SPSA Phase: Simultaneous Perturbation Stochastic Approximation
        # SPSA estimates the gradient using two function evaluations per step:
        # g_k = (f(x + c*d) - f(x - c*d)) / (2*c) * d
        # where d is a random vector with entries +/-1.
        # This is much more sample-efficient than finite differences (which need 2*n evals)
        # and often faster than CMA-ES for smooth functions in low dimensions when combined with good init.
        
        # SPSA Parameters
        # Standard SPSA gain sequences: a_k = a / (k + A)^alpha, c_k = c / k^gamma
        # Tuned for n=5 and budget ~280 remaining
        a = 0.5  # Step size gain
        A = 10.0 # Offset to prevent large initial steps
        alpha = 0.602
        c = 0.2   # Perturbation size
        gamma = 0.101
        
        x_current = best_x.copy()
        f_current = best_f
        k = 0 # SPSA iteration counter
        
        # We will run SPSA until budget or time is exhausted
        # SPSA is a local search method. To handle multimodality better, 
        # we could restart SPSA from the best point found so far if it stagnates,
        # but given the short budget, a single strong SPSA run from the best Halton point 
        # is a solid structural change from the population-based CMA-ES.
        # However, to be safer against local minima, we can do a few "re-simulated" 
        # SPSA runs from the top 2-3 Halton points if budget allows, but 
        # the directive asks for a structural change. SPSA itself is the change.
        
        # Let's stick to one robust SPSA run from the global best found in init.
        # If it stagnates, we might just stop. 
        
        while used_budget < budget:
            if time.time() > deadline:
                break
            
            # SPSA requires 2 evaluations per step
            if used_budget + 2 > budget:
                break
                
            k += 1
            
            # Calculate current gains
            a_k = a / (k + A) ** alpha
            c_k = c / k ** gamma
            
            # Generate random direction d with entries in {-1, 1}
            d = np.where(np.random.rand(n) < 0.5, -1.0, 1.0)
            
            # Calculate perturbed points
            x_plus = x_current + c_k * d
            x_minus = x_current - c_k * d
            
            # Clip to box
            x_plus = np.clip(x_plus, -5.0, 5.0)
            x_minus = np.clip(x_minus, -5.0, 5.0)
            
            # Evaluate
            f_plus = func(x_plus)
            f_minus = func(x_minus)
            used_budget += 2
            
            # Gradient estimate
            # Note: If clipping occurred, the effective perturbation size changes.
            # For simplicity and robustness, we use the standard SPSA update.
            # The clipping handles boundary constraints.
            gradient = (f_plus - f_minus) / (2 * c_k) * d
            
            # Update position
            x_new = x_current - a_k * gradient
            
            # Clip to box
            x_new = np.clip(x_new, -5.0, 5.0)
            
            # Evaluate new point to decide if it's an improvement
            # Wait, standard SPSA updates x without evaluating x_new immediately in the loop logic?
            # No, SPSA is a stochastic approximation. The next iteration uses x_new.
            # However, we need to track the best value found for the return value.
            # And we need to know if we are making progress.
            # Usually, SPSA doesn't require evaluating x_new separately if we just move there.
            # But to update best_x, we need f(x_new).
            # This would cost 1 more eval per step (3 total).
            # Alternative: Use the fact that f_plus and f_minus are close to f(x_new).
            # Or, just accept that we don't know f(x_new) exactly until we evaluate it.
            # Since we have a tight budget, let's use a (1+1) style check:
            # If we evaluate x_new, we spend 3 evals/step.
            # If we don't, we might move to a worse point and lose track of the best.
            # Let's evaluate x_new to maintain a valid "current best" and allow for restarts if needed.
            
            f_new = func(x_new)
            used_budget += 1
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                x_current = x_new.copy()
                f_current = f_new
            else:
                # If it didn't improve, we still move to x_new in standard SPSA?
                # Actually, SPSA is a gradient descent method. It moves in the direction of the estimated gradient.
                # It doesn't necessarily improve every step.
                # So we update x_current to x_new regardless, to continue the stochastic process.
                x_current = x_new.copy()
                f_current = f_new # We know this value now
            
            # Optional: Stagnation check and restart?
            # If we haven't improved in, say, 20 steps, we could restart SPSA from best_x with a fresh k=0?
            # This adds complexity. Given the budget, let's keep it simple.
            
        return best_x, best_f
