import numpy as np
import time

class BudgetExceeded(Exception):
    pass

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        dim = self.dim
        budget = self.budget
        
        # Track state
        calls = 0
        best_x = None
        best_fx = np.inf
        
        # Wrapper to count calls and enforce budget
        def wrapped_func(x):
            nonlocal calls, best_x, best_fx
            if time.time() > deadline:
                raise BudgetExceeded("Time limit exceeded")
            
            # Clip to domain
            x_clipped = np.clip(x, -5.0, 5.0)
            
            # Call the actual function
            fx = func(x_clipped)
            calls += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
                
            if calls >= budget:
                raise BudgetExceeded("Budget exceeded")
                
            return fx

        # Bent Cigar is highly anisotropic: f(x) = (x1-3)^2 + sum_{i=2}^{n} i^2*(xi-1)^2
        # The narrow valley makes gradient methods slow. Nelder-Mead with adaptive
        # scaling handles this better by exploring along principal axes.
        
        # Start at origin (reasonable for shifted/rotated instances)
        x0 = np.zeros(dim)
        wrapped_func(x0)
        
        # Nelder-Mead simplex method with adaptive step sizes
        # Initialize simplex with scaled initial step
        simplex = [x0.copy()]
        for i in range(dim):
            v = x0.copy()
            # Use different initial step sizes per dimension to handle anisotropy
            v[i] += 1.0 if i == 0 else 0.5
            simplex.append(v)
        
        # Evaluate initial simplex
        for i, v in enumerate(simplex):
            try:
                wrapped_func(v)
            except BudgetExceeded:
                break
        
        # Nelder-Mead iterations
        max_iter = 200
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            if calls >= budget - 5:
                break
            
            # Sort simplex by function value
            # We need to track which point has which value
            # Since we can't easily get all values back, we re-evaluate
            # Actually, let's track values separately
            
            # This approach is getting complex. Let's use a simpler strategy:
            # Coordinate descent with line search, which is very effective for
            # separable/anisotropic functions like Bent Cigar.
            break
        
        # Better approach: Alternating coordinate descent with golden-section line search
        # This is extremely effective for Bent Cigar because it's nearly separable
        # after rotation, and even without knowing the rotation, cycling through
        # coordinates with good line searches works well.
        
        # First, do a coarse coordinate descent to get into the valley
        x = best_x.copy()
        
        for coord_pass in range(30):
            if time.time() > deadline - 20:
                break
            if calls >= budget - 10:
                break
            
            for i in range(dim):
                if time.time() > deadline - 20:
                    break
                if calls >= budget - 10:
                    break
                
                # Golden-section search along dimension i
                a = max(-5.0, x[i] - 2.0)
                b = min(5.0, x[i] + 2.0)
                
                # Golden ratio
                gr = (1.0 + np.sqrt(5.0)) / 2.0
                invphi = 1.0 / gr
                invphi2 = invphi * invphi
                
                c = b - invphi * (b - a)
                d = a + invphi * (b - a)
                
                x_c = x.copy()
                x_c[i] = c
                x_d = x.copy()
                x_d[i] = d
                
                try:
                    fc = wrapped_func(x_c)
                    fd = wrapped_func(x_d)
                except BudgetExceeded:
                    break
                
                for _ in range(15):
                    if time.time() > deadline - 20:
                        break
                    if calls >= budget - 10:
                        break
                    
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - invphi * (b - a)
                        x_c[i] = c
                        try:
                            fc = wrapped_func(x_c)
                        except BudgetExceeded:
                            break
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + invphi * (b - a)
                        x_d[i] = d
                        try:
                            fd = wrapped_func(x_d)
                        except BudgetExceeded:
                            break
                
                # Best point in this coordinate
                if fc < fd:
                    x[i] = c
                else:
                    x[i] = d
        
        # Refine with smaller steps
        for refine_pass in range(20):
            if time.time() > deadline - 10:
                break
            if calls >= budget - 5:
                break
            
            step_scale = 0.5 ** refine_pass
            
            for i in range(dim):
                if time.time() > deadline - 10:
                    break
                if calls >= budget - 5:
                    break
                
                # Try small perturbations
                for delta in [step_scale, -step_scale]:
                    if time.time() > deadline - 10:
                        break
                    if calls >= budget - 5:
                        break
                    
                    x_test = x.copy()
                    x_test[i] += delta
                    x_test = np.clip(x_test, -5.0, 5.0)
                    
                    try:
                        fx_test = wrapped_func(x_test)
                    except BudgetExceeded:
                        break
                    
                    if fx_test < best_fx:
                        x = x_test.copy()
        
        # Final polish with very small steps
        for final_pass in range(10):
            if time.time() > deadline - 5:
                break
            if calls >= budget - 2:
                break
            
            step_scale = 0.1 ** (final_pass + 1)
            
            for i in range(dim):
                if time.time() > deadline - 5:
                    break
                if calls >= budget - 2:
                    break
                
                for delta in [step_scale, -step_scale]:
                    if time.time() > deadline - 5:
                        break
                    if calls >= budget - 2:
                        break
                    
                    x_test = x.copy()
                    x_test[i] += delta
                    x_test = np.clip(x_test, -5.0, 5.0)
                    
                    try:
                        fx_test = wrapped_func(x_test)
                    except BudgetExceeded:
                        break
                    
                    if fx_test < best_fx:
                        x = x_test.copy()
        
        # If we didn't update best_x (shouldn't happen), use x0
        if best_x is None:
            best_x = x0
            try:
                best_fx = wrapped_func(x0)
            except BudgetExceeded:
                best_fx = np.inf

        return best_x, best_fx
