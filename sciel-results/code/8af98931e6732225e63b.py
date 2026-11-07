"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope) using Deterministic Coordinate Descent.
    
    Strategy:
    1. The function is effectively linear: f(x) = a^T x + b.
    2. For a linear function, the minimum on a box [-5, 5]^d occurs at a vertex 
       (corner) where each coordinate is either -5 or 5, determined by the sign 
       of the corresponding component of the gradient vector a.
    3. Instead of estimating the full gradient via finite differences (which uses 
       2*d calls) and then guessing the corner, we can determine the optimal 
       direction for each coordinate independently using a small number of 
       evaluations per coordinate.
    4. Specifically, for each dimension i, we evaluate f at x with x_i = -5 and 
       x_i = 5 (keeping other coordinates at a reference point, e.g., 0). 
       This tells us if the function decreases as x_i goes to -5 or +5.
    5. Actually, a more robust approach for linear functions is to simply find 
       the sign of the slope for each dimension. 
       f(x) - f(x - h*e_i) approx h * a_i.
       If we pick a large h (e.g., from 0 to 5), we can directly see the trend.
       
    Let's use a "Boundary Probe" strategy:
    1. Start at center x=0.
    2. For each dimension i, compare f(x with x_i=5) vs f(x with x_i=-5).
       This requires 2*d evaluations total (if we reuse the center call for one side? 
       No, center is 0. We need to evaluate at boundaries).
       Actually, we can do:
       - Evaluate f(0).
       - For each i, evaluate f(e_i * 5) and f(-e_i * 5) relative to 0? 
         No, we need to evaluate the full vector.
       
    Better Strategy: "Sign Detection via Boundary Evaluation"
    1. Evaluate f(0).
    2. For each dimension i:
       - Construct x_plus: x_i = 5, others = 0.
       - Construct x_minus: x_i = -5, others = 0.
       - Evaluate f(x_plus) and f(x_minus).
       - If f(x_plus) < f(x_minus), the slope in direction i is likely negative 
         (function decreases as x_i increases), so optimal x_i is 5.
       - If f(x_minus) < f(x_plus), the slope is positive, so optimal x_i is -5.
    3. This uses 1 (center) + 2*d (boundary probes) = 11 calls for d=5.
    4. Construct the candidate corner x_opt based on these signs.
    5. Evaluate f(x_opt). This is likely the global minimum.
    6. If budget allows, perform a final verification or minor adjustment.
       Since the function is linear, the corner found by sign detection should 
       be exact. Any "noise" or non-linearity is minimal or absent.
       
    Wait, what if the rotation mixes the coordinates?
    The task says "hidden seeded shift, rotation and offset".
    If there is a rotation, f(x) = a^T (R(x-s)) + b = (R^T a)^T x + const.
    It is still a linear function in x.
    The gradient is constant: g = R^T a.
    The minimum is at x_i = -5 if g_i > 0, else 5.
    
    My sign detection logic:
    f(0) = b'
    f(5*e_i) = 5*g_i + b'
    f(-5*e_i) = -5*g_i + b'
    
    If g_i > 0:
    f(5*e_i) = 5g_i + b' > b'
    f(-5*e_i) = -5g_i + b' < b'
    So f(-5*e_i) < f(5*e_i). We should set x_i = -5.
    
    If g_i < 0:
    f(5*e_i) = 5g_i + b' < b'
    f(-5*e_i) = -5g_i + b' > b'
    So f(5*e_i) < f(-5*e_i). We should set x_i = 5.
    
    This logic holds perfectly for any linear function.
    
    Total calls:
    1 for f(0)
    2 * 5 = 10 for the probes.
    1 for the final corner evaluation.
    Total = 12 calls.
    
    We have 500 calls. We can use the remaining budget to verify or handle 
    edge cases (e.g., if the function is not perfectly linear, or if there's 
    numerical precision issues). But for BBOB f5, it is linear.
    
    To be safe against the "minimizer lies in [-4, 4]^5" constraint which might 
    imply the function is NOT purely linear but has a quadratic term or is 
    constrained such that the minimum is interior?
    "BBOB-style function f5 (Linear slope)".
    Standard BBOB f5 is: f(x) = sum(x_i) + 10 * sum(x_i^2) ? No, that's f2 or similar.
    Let's check BBOB f5 definition specifically.
    BBOB 2019 f5: "Linear slope".
    f(x) = sum_{i=1}^d x_i.
    This is strictly linear.
    
    However, the prompt says: "Instances apply a hidden seeded shift, rotation and offset".
    If f(x) = sum(x), then f(R(x-s)+o) is still linear.
    
    What if the "minimizer lies in [-4, 4]^5" refers to the *shift* vector s?
    "the minimizer lies in [-4, 4]^5".
    If the function is linear, the unconstrained minimizer is at infinity.
    The constrained minimizer is at a corner of the box.
    Unless... the function is actually f(x) = ||x - z||^2 or something?
    But it says "Linear slope".
    
    Maybe the "minimizer" text is a generic template text for the task suite 
    that doesn't perfectly fit f5, OR it implies that the *effective* minimum 
    after transformation is inside?
    If the function is linear, the minimum on the box is ALWAYS at a corner.
    The only way the minimizer is inside is if the gradient is 0 (constant function).
    In that case, any point is a minimizer.
    
    Let's stick to the Corner Strategy. It is optimal for linear functions.
    
    Implementation:
    1. Evaluate f(0).
    2. For i in 0..4:
       - x_p = 0; x_p[i] = 5.
       - x_m = 0; x_m[i] = -5.
       - fp = func(x_p)
       - fm = func(x_m)
       - if fp < fm: opt_dir[i] = 1 (go to 5)
       - else: opt_dir[i] = -1 (go to -5)
    3. x_opt = opt_dir * 5.
    4. f_opt = func(x_opt).
    5. Return x_opt, f_opt.
    
    This uses 1 + 10 + 1 = 12 calls.
    We have 488 calls left.
    We can use these to do a sanity check.
    If f_opt is not better than f(0), something is wrong.
    But for linear, it must be better (unless gradient is 0).
    
    To utilize the extra budget for robustness (e.g. if the function is not 
    perfectly linear due to some hidden non-linearity I'm missing, or if 
    "Linear slope" implies a specific BBOB function that has a slight curvature?
    No, f5 is the simplest one.
    
    Let's add a Nelder-Mead step from the corner just in case, using the 
    remaining budget. It will converge instantly if the corner is correct.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        def timed_out():
            return time.time() > deadline

        dim = self.dim
        budget = self.budget
        calls = 0
        
        # 1. Center evaluation
        x_center = np.zeros(dim)
        f_center = func(x_center)
        calls += 1
        best_x = x_center.copy()
        best_f = f_center
        
        # 2. Determine optimal direction for each dimension
        # We probe the boundaries of the box along each axis from the center.
        # For a linear function f(x) = g^T x + b:
        # f(5*e_i) - f(0) = 5*g_i
        # f(-5*e_i) - f(0) = -5*g_i
        # If g_i > 0, f(-5*e_i) < f(0) < f(5*e_i). Optimal is -5.
        # If g_i < 0, f(5*e_i) < f(0) < f(-5*e_i). Optimal is 5.
        
        opt_dirs = np.zeros(dim)
        
        for i in range(dim):
            if timed_out() or calls >= budget - 5:
                break
                
            x_plus = np.zeros(dim)
            x_plus[i] = 5.0
            x_minus = np.zeros(dim)
            x_minus[i] = -5.0
            
            f_plus = func(x_plus)
            calls += 1
            f_minus = func(x_minus)
            calls += 1
            
            # Determine sign
            if f_plus < f_minus:
                opt_dirs[i] = 5.0
            else:
                opt_dirs[i] = -5.0
                
            # Update best if the probe point itself is better (unlikely for linear 
            # unless the other coords matter, but we are at 0 for others, so this 
            # is just a partial evaluation. The full corner will be better).
            # Actually, f(x_plus) is not directly comparable to f_center in a way 
            # that updates the global best for the full vector, because x_plus 
            # is not the candidate for the final answer, just a probe.
            # However, if the function is non-linear, maybe x_plus is a local min?
            # Unlikely for f5. We stick to the corner hypothesis.
            
            if f_plus < best_f:
                best_f = f_plus
                best_x = x_plus.copy()
            if f_minus < best_f:
                best_f = f_minus
                best_x = x_minus.copy()

        # 3. Construct the optimal corner
        x_opt = opt_dirs.copy()
        
        # 4. Evaluate the corner
        if calls < budget and not timed_out():
            f_opt = func(x_opt)
            calls += 1
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        
        # 5. If we have significant budget left, run a quick Nelder-Mead 
        #    from the best found point to handle any non-linearity or 
        #    precision issues.
        
        if calls < budget - 10 and not timed_out():
            # Simple Nelder-Mead implementation
            # Initialize simplex
            x0 = best_x.copy()
            f0 = best_f
            
            # Create simplex
            simplex = [x0]
            for i in range(dim):
                xi = x0.copy()
                # Step size 1.0 is reasonable for box [-5,5]
                step = 1.0 if x0[i] < 4.0 else -1.0
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
            
            # Evaluate simplex
            fvals = [f0]
            for i in range(1, dim + 1):
                if timed_out() or calls >= budget:
                    break
                fx = func(simplex[i])
                calls += 1
                fvals.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = simplex[i].copy()
            
            # Nelder-Mead iterations
            max_iter = 100
            for _ in range(max_iter):
                if timed_out() or calls >= budget:
                    break
                    
                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Check convergence
                if abs(fvals[0] - fvals[-1]) < 1e-8:
                    break
                    
                # Centroid of best dim points
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                    
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Inner contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                    else:
                        # Outer contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                        
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fs = func(simplex[i])
                            calls += 1
                            fvals[i] = fs
                            if fs < best_f:
                                best_f = fs
                                best_x = simplex[i].copy()
                            
                            if timed_out() or calls >= budget:
                                break
                        if timed_out() or calls >= budget:
                            break

        return best_x, best_f

# EVOLVE-BLOCK-END
