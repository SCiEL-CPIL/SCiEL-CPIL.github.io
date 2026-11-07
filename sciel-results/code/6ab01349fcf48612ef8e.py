"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer: 
    1. Global search using Nelder-Mead simplex (fast for low-dim).
    2. Local refinement using Quadratic Fit (exploits Sphere structure).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        # 1. Global Search Phase: Nelder-Mead
        # Nelder-Mead is very fast for n=5 and convex functions.
        # We use it to find the neighborhood of the minimum.
        
        # Initialize simplex
        # Start at 0,0,0,0,0 as the shift is in [-4,4]
        x0 = np.zeros(n)
        
        # Initial step size for simplex
        # Since box is [-5,5] and min in [-4,4], a step of 1.0 is reasonable
        step = 1.0
        
        simplex = [x0.copy()]
        for i in range(n):
            x = x0.copy()
            x[i] += step
            simplex.append(x)
        
        simplex = np.array(simplex)
        fvals = np.array([func(x) for x in simplex])
        calls_made = n + 1
        
        # Track best
        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma_nm = 0.5 # Shrink
        
        # Max iterations for NM phase: use ~30% of budget
        nm_max_iter = budget // 3
        
        for _ in range(nm_max_iter):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
                
            # Sort simplex
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            best_idx = 0
            worst_idx = n
            second_worst_idx = n - 1
            
            # Check convergence
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[worst_idx])
            f_r = func(x_r)
            calls_made += 1
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[worst_idx])
                f_e = func(x_e)
                calls_made += 1
                if f_e < f_r:
                    simplex[worst_idx] = x_e
                    fvals[worst_idx] = f_e
                else:
                    simplex[worst_idx] = x_r
                    fvals[worst_idx] = f_r
            elif f_r < fvals[second_worst_idx]:
                simplex[worst_idx] = x_r
                fvals[worst_idx] = f_r
            else:
                # Contraction
                if f_r < fvals[worst_idx]:
                    x_c = centroid + rho * (x_r - centroid)
                    f_c = func(x_c)
                    calls_made += 1
                else:
                    x_c = centroid + rho * (simplex[worst_idx] - centroid)
                    f_c = func(x_c)
                    calls_made += 1
                    
                if f_c < min(f_r, fvals[worst_idx]):
                    simplex[worst_idx] = x_c
                    fvals[worst_idx] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls_made += 1
                        if calls_made >= budget:
                            break
                    if calls_made >= budget:
                        break
                        
            # Update best
            current_best_idx = np.argmin(fvals)
            if fvals[current_best_idx] < best_f:
                best_f = fvals[current_best_idx]
                best_x = simplex[current_best_idx].copy()

        # 2. Local Refinement Phase: Quadratic Fit
        # We have used ~1/3 budget. We have best_x.
        # Sphere function: f(x) = ||x - x*||^2 + c
        # We can estimate x* by fitting a quadratic to points around best_x.
        # Strategy:
        # 1. Generate 2*n points: best_x +/- delta * e_i
        # 2. Solve for x* using the gradient approximation or direct quadratic fit.
        # For Sphere: f(x_i + h e_j) = f(x_i) + 2 h (x_ij - x*_j) + h^2
        # So: (f(x_i + h e_j) - f(x_i - h e_j)) / (2h) approx (x_ij - x*_j)
        # Thus: x*_j approx x_ij - (f(x_i + h e_j) - f(x_i - h e_j)) / (2h)
        # This is a finite difference gradient estimate.
        # Since we know it's a sphere, we can do better:
        # Let f0 = f(x0).
        # f(x0 + h e_j) = f0 + 2 h (x0_j - x*_j) + h^2
        # f(x0 - h e_j) = f0 + 2 h (x0_j - x*_j) (-1) + h^2 = f0 - 2 h (x0_j - x*_j) + h^2
        # Sum: f(+h) + f(-h) = 2 f0 + 2 h^2 => f0 = (f(+h) + f(-h))/2 - h^2
        # Diff: f(+h) - f(-h) = 4 h (x0_j - x*_j) => x*_j = x0_j - (f(+h) - f(-h))/(4h)
        
        # This requires 2n calls per iteration to estimate the full x*.
        # We can iterate this: estimate x*, jump to x*, repeat with smaller h.
        
        # We have budget remaining. Let's use a specific number of refinement steps.
        # Each step costs 2*n calls.
        # n=5, so 10 calls per step.
        # Remaining budget ~ 2/3 * 500 = 333.
        # 333 // 10 = 33 steps.
        # This is plenty.
        
        current_x = best_x.copy()
        current_f = best_f
        
        # Start with a moderate step size for the finite difference
        # It should be large enough to overcome noise but small enough to be in the quadratic regime.
        # Since NM converged, we are close.
        h = 0.5
        
        # We will perform refinement until budget or time runs out.
        # We stop when improvement is negligible or h becomes too small.
        
        while calls_made + 2 * n <= budget:
            if time.time() > deadline:
                break
            
            # Evaluate central point if not already known (it is known as best_f)
            # But we need f(x0) to be accurate. best_f is accurate.
            
            x_new = current_x.copy()
            
            # Estimate x* components
            for j in range(n):
                x_plus = current_x.copy()
                x_plus[j] += h
                x_minus = current_x.copy()
                x_minus[j] -= h
                
                # Ensure within bounds? The function clips, but for the math of the fit,
                # we assume we are not hitting the boundary. 
                # Given min in [-4,4] and we started at 0, NM likely kept us away from [-5,5].
                # If we are close to boundary, this estimate might be biased.
                # However, Sphere is smooth.
                
                fp = func(x_plus)
                fm = func(x_minus)
                calls_made += 2
                
                # x*_j = x_j - (fp - fm) / (4h)
                # Derivation:
                # fp = f0 + 2h(x_j - x*_j) + h^2
                # fm = f0 - 2h(x_j - x*_j) + h^2
                # fp - fm = 4h(x_j - x*_j)
                # x_j - x*_j = (fp - fm) / 4h
                # x*_j = x_j - (fp - fm) / 4h
                
                x_new[j] = current_x[j] - (fp - fm) / (4 * h)
                
            # Clip to valid box to be safe, though x* should be in [-4,4]
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate the new estimated center
            f_new = func(x_new)
            calls_made += 1
            
            if f_new < current_f:
                current_f = f_new
                current_x = x_new
                best_f = current_f
                best_x = current_x
                # Shrink step size for next iteration as we are converging
                h *= 0.5
            else:
                # If it didn't improve, we might have hit noise or boundary.
                # Shrink step size anyway to try to refine more locally.
                h *= 0.5
                if h < 1e-6:
                    break
                    
        return best_x, best_f

# EVOLVE-BLOCK-END
