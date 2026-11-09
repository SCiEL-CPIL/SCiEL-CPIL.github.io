import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = time.time()
        self.deadline = 280.0 # Safety margin for 300s limit

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)
        
        # Helper to track calls
        def eval_func(x):
            if self.calls >= self.budget:
                return self.best_f
            if time.time() > self.deadline:
                return self.best_f
            self.calls += 1
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x_clipped.copy()
            return f_val

        # Phase 1: Initialization
        # Try a few random points to get into the basin
        init_points = 10
        if self.budget <= init_points:
             # If budget is tiny, just do random search
             for _ in range(self.budget):
                 x = rng.uniform(-5, 5, self.dim)
                 eval_func(x)
             return self.best_x, self.best_f

        for _ in range(init_points):
            x = rng.uniform(-5, 5, self.dim)
            eval_func(x)
        
        x = self.best_x.copy()
        fx = self.best_f

        # Phase 2: Powell's Method
        # Directions
        directions = np.eye(self.dim)
        
        # Simple 1D minimizer using parabolic/golden hybrid
        def min_1d(x, d, f_val):
            # Minimize f(x + alpha * d)
            # We need to find alpha.
            # Start with a small step, expand if decreasing, then refine.
            # To save calls, we use a fixed small search space around 0 initially?
            # Better: Use a simple golden section search on alpha in a range.
            # But we don't know the scale.
            # Let's use a simple adaptive step.
            
            # This is a simplified 1D search that uses ~15 calls max.
            alpha = 0.0
            best_alpha = 0.0
            best_f_alpha = f_val
            
            # Try a few initial steps to determine direction and scale
            step = 0.1
            improved = False
            for i in range(5):
                # Try positive and negative
                x_plus = x + step * d
                f_plus = eval_func(x_plus)
                if f_plus < best_f_alpha:
                    best_f_alpha = f_plus
                    best_alpha = step
                    improved = True
                    # Expand
                    step *= 1.5
                else:
                    break
            
            x_minus = x - step * d
            f_minus = eval_func(x_minus)
            if f_minus < best_f_alpha:
                best_f_alpha = f_minus
                best_alpha = -step
                improved = True
                # Expand negative
                step *= 1.5
                for i in range(5):
                    x_minus2 = x - step * d
                    f_minus2 = eval_func(x_minus2)
                    if f_minus2 < best_f_alpha:
                        best_f_alpha = f_minus2
                        best_alpha = -step
                        step *= 1.5
                    else:
                        break
            
            if not improved:
                # No improvement in initial probes, return current
                return x, f_val, best_alpha

            # Now we have a bracket [alpha_low, alpha_high] or a best point.
            # Let's just do a simple local refinement around best_alpha using golden section.
            # Range for alpha
            if best_alpha > 0:
                a, b = 0.0, best_alpha * 2
            else:
                a, b = best_alpha * 2, 0.0
            
            # Golden section search
            invphi = (np.sqrt(5) - 1) / 2
            invphi2 = invphi * invphi
            
            c = b - invphi * (b - a)
            d = a + invphi * (b - a)
            
            # We already have f at a (which is f_val if a=0) and maybe b?
            # Let's just evaluate c and d.
            fc = eval_func(x + c * d)
            fd = eval_func(x + d * d)
            
            # We have used ~15 calls so far (5 pos + 1 neg + 5 neg + 2 gold + 1 init)
            # Let's do 5 more iterations of golden section
            for _ in range(5):
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - invphi * (b - a)
                    fc = eval_func(x + c * d)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + invphi * (b - a)
                    fd = eval_func(x + d * d)
                
                if self.calls >= self.budget:
                    break

            # Pick best of a, b, c, d
            candidates = [a, b, c, d]
            # We don't have all f values stored efficiently, just re-eval best guess or keep track.
            # For simplicity, assume the last evaluated or min of stored is good.
            # Actually, let's just return the point corresponding to the lowest f seen in this 1D search.
            # This requires tracking.
            
            # Simplified: Just return x + best_alpha * d if it was better, else x
            # This is a bit crude. Let's implement a proper 1D search with tracking.
            
            return x + best_alpha * d, best_f_alpha, best_alpha

        # The above min_1d is a bit messy. Let's replace with a cleaner implementation.
        
        def robust_min_1d(x, d, f_val):
            """
            Minimizes f(x + alpha * d).
            Returns (new_x, new_f, alpha_used, f_reductions)
            """
            # Initial step size
            step = 0.5
            # Try to find a decreasing direction
            alpha = 0.0
            best_alpha = 0.0
            best_f = f_val
            
            # Probe forward
            for i in range(3):
                x_test = x + step * d
                f_test = eval_func(x_test)
                if f_test < best_f:
                    best_f = f_test
                    best_alpha = step
                    step *= 1.5
                else:
                    break
            
            # Probe backward
            step_b = 0.5
            for i in range(3):
                x_test = x - step_b * d
                f_test = eval_func(x_test)
                if f_test < best_f:
                    best_f = f_test
                    best_alpha = -step_b
                    step_b *= 1.5
                else:
                    break
            
            if best_alpha == 0.0:
                return x, f_val, 0.0, 0.0

            # Refine using Golden Section Search
            # Bracket: if best_alpha > 0, search [0, best_alpha*1.5]
            # if best_alpha < 0, search [best_alpha*1.5, 0]
            
            if best_alpha > 0:
                a, b = 0.0, best_alpha * 1.5
            else:
                a, b = best_alpha * 1.5, 0.0
            
            # We know f(a) = f_val (if a=0) or we need to evaluate.
            # Let's just run golden section for 8 iterations max.
            
            invphi = (np.sqrt(5) - 1) / 2
            
            # Initialize
            if b > a:
                c = b - invphi * (b - a)
                d = a + invphi * (b - a)
            else:
                # Swap if needed, but a<b is enforced above
                pass
                
            fc = eval_func(x + c * d)
            fd = eval_func(x + d * d)
            
            # Track best
            # We have f_val at alpha=0.
            # Let's keep a small dict or just variables.
            # For simplicity, we assume the golden section converges to the min.
            
            for _ in range(8):
                if self.calls >= self.budget:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - invphi * (b - a)
                    fc = eval_func(x + c * d)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + invphi * (b - a)
                    fd = eval_func(x + d * d)
            
            # Determine best alpha among a, b, c, d, 0
            # This is inefficient if we don't store f values.
            # Let's just pick the midpoint of the final interval as a good estimate.
            alpha_final = (a + b) / 2
            x_final = x + alpha_final * d
            f_final = eval_func(x_final)
            
            if f_final < best_f:
                best_f = f_final
                best_alpha = alpha_final
            
            return x + best_alpha * d, best_f, best_alpha, f_val - best_f

        # Powell Loop
        # We have budget left.
        # Each cycle: 5 directions * ~15 calls = 75 calls.
        # We can do ~5 cycles with 400 calls.
        
        while self.calls < self.budget - 50: # Save 50 for polish
            if time.time() > self.deadline:
                break
            
            x_prev = x.copy()
            f_prev = fx
            reductions = []
            
            for i in range(self.dim):
                if self.calls >= self.budget - 50:
                    break
                
                x_new, f_new, alpha, reduction = robust_min_1d(x, directions[i], fx)
                x = x_new
                fx = f_new
                reductions.append(reduction)
            
            # Check if we should update directions
            # Condition: reduction in last direction > reduction in first direction
            if reductions and reductions[-1] > reductions[0]:
                # Replace direction with smallest reduction
                min_idx = np.argmin(reductions)
                directions[min_idx] = x - x_prev
                # Normalize? Powell's method usually doesn't normalize, but it helps with scale.
                # Let's normalize to unit length to keep steps consistent.
                norm = np.linalg.norm(directions[min_idx])
                if norm > 1e-10:
                    directions[min_idx] /= norm
            
            # Check convergence
            if np.linalg.norm(x - x_prev) < 1e-6:
                break

        # Phase 3: Nelder-Mead Polish
        # Use remaining budget
        # Simple Nelder-Mead implementation
        
        def nelder_mead(x0, f0, max_iter):
            # Initialize simplex
            simplex = [x0]
            simple_f = [f0]
            
            # Create other points
            for i in range(self.dim):
                x_i = x0.copy()
                # Step size
                if abs(x0[i]) > 1e-5:
                    x_i[i] *= 1.05
                else:
                    x_i[i] += 0.05
                f_i = eval_func(x_i)
                simplex.append(x_i)
                simple_f.append(f_i)
                
                if self.calls >= self.budget:
                    break
            
            for _ in range(max_iter):
                if self.calls >= self.budget:
                    break
                if time.time() > self.deadline:
                    break
                
                # Sort simplex
                order = np.argsort(simple_f)
                simplex = [simplex[i] for i in order]
                simple_f = [simple_f[i] for i in order]
                
                # Centroid of best n points
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + (centroid - simplex[-1])
                f_ref = eval_func(x_ref)
                
                if f_ref < simple_f[0]:
                    # Expansion
                    x_exp = centroid + 2 * (centroid - simplex[-1])
                    f_exp = eval_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simple_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simple_f[-1] = f_ref
                elif f_ref < simple_f[-2]:
                    simplex[-1] = x_ref
                    simple_f[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + 0.5 * (simplex[-1] - centroid)
                    f_con = eval_func(x_con)
                    if f_con < simple_f[-1]:
                        simplex[-1] = x_con
                        simple_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simple_f[i] = eval_func(simplex[i])
                            if self.calls >= self.budget:
                                break
                        if self.calls >= self.budget:
                            break
                
                # Update global best
                # eval_func already updates self.best_x/f

        # Run Nelder-Mead with remaining budget
        # Estimate iterations: 1 iter ~ 6-10 calls.
        # Remaining budget = self.budget - self.calls
        # Max iter = remaining // 10
        if self.budget - self.calls > 10:
            nelder_mead(self.best_x, self.best_f, (self.budget - self.calls) // 10)

        return self.best_x, self.best_f

