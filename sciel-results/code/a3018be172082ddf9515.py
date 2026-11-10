import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, no overlaps, positive sides."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _energy_and_grad(x, n=10):
    """
    x: array of shape (3n,) -> [cx_1, cy_1, s_1, ..., cx_n, cy_n, s_n]
    Returns energy (scalar) and gradient.
    Energy = -sum(s_i) + lambda * (boundary_penalty + overlap_penalty)
    """
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    # Objective: Minimize -sum(sides) + penalties
    # We want to maximize sum(sides), so we minimize -sum(sides).
    
    # 1. Boundary Penalties
    # For each square i, if it goes out of bounds, penalize quadratically.
    # x_min = cx - s/2, x_max = cx + s/2
    # Violation if x_min < 0 or x_max > 1
    
    boundary_pen = 0.0
    boundary_grad = np.zeros_like(x)
    
    for i in range(n):
        cx, cy, s = centers[i, 0], centers[i, 1], sides[i]
        
        # X-axis
        x_min = cx - s / 2
        x_max = cx + s / 2
        
        if x_min < 0:
            pen = x_min ** 2
            boundary_pen += pen
            # d(pen)/dx_min = 2 * x_min
            # d(x_min)/dcx = -1, d(x_min)/ds = -0.5
            boundary_grad[2*i] += 2 * x_min * (-1)
            boundary_grad[2*n+i] += 2 * x_min * (-0.5)
            
        if x_max > 1:
            pen = (x_max - 1) ** 2
            boundary_pen += pen
            # d(pen)/dx_max = 2 * (x_max - 1)
            # d(x_max)/dcx = 1, d(x_max)/ds = 0.5
            boundary_grad[2*i] += 2 * (x_max - 1) * 1
            boundary_grad[2*n+i] += 2 * (x_max - 1) * 0.5
            
        # Y-axis
        y_min = cy - s / 2
        y_max = cy + s / 2
        
        if y_min < 0:
            pen = y_min ** 2
            boundary_pen += pen
            boundary_grad[2*i+1] += 2 * y_min * (-1)
            boundary_grad[2*n+i] += 2 * y_min * (-0.5)
            
        if y_max > 1:
            pen = (y_max - 1) ** 2
            boundary_pen += pen
            boundary_grad[2*i+1] += 2 * (y_max - 1) * 1
            boundary_grad[2*n+i] += 2 * (x_max - 1) * 0.5 # Correction: should be y_max
            
    # 2. Overlap Penalties
    # Use log-sum-exp of squared overlap depths for smoothness.
    # Overlap condition: |cx_i - cx_j| < (s_i+s_j)/2 AND |cy_i - cy_j| < (s_i+s_j)/2
    # Let dx = cx_i - cx_j, dy = cy_i - cy_j
    # Let tx = (s_i+s_j)/2, ty = (s_i+s_j)/2
    # Overlap if |dx| < tx and |dy| < ty.
    # We can define a "repulsion" based on how much they are inside each other.
    # A common differentiable proxy for non-overlap is:
    # penalty = sum_{i<j} max(0, tx - |dx|)^2 * max(0, ty - |dy|)^2
    # This is 0 if no overlap. If overlap, it grows quadratically.
    # However, max(0, x) is not differentiable at 0.
    # We can use a smooth approximation or just rely on the fact that L-BFGS-B
    # can handle kinks if we are careful, or use a softplus.
    # Let's use: pen = exp(alpha * (tx - |dx|) * (ty - |dy|)) / alpha - 1/alpha ?
    # Simpler: Just use the squared violation directly, it's C1 continuous except at boundaries of overlap.
    # Actually, (max(0, A))^2 is C1. Its derivative is 2*max(0,A)*dA/dx.
    
    overlap_pen = 0.0
    overlap_grad = np.zeros_like(x)
    
    for i in range(n):
        for j in range(i+1, n):
            cx_i, cy_i, s_i = centers[i, 0], centers[i, 1], sides[i]
            cx_j, cy_j, s_j = centers[j, 0], centers[j, 1], sides[j]
            
            tx = (s_i + s_j) / 2.0
            ty = (s_i + s_j) / 2.0
            
            dx = cx_i - cx_j
            dy = cy_i - cy_j
            
            # Violation amounts
            # We want to penalize if |dx| < tx and |dy| < ty.
            # Let vx = tx - |dx|, vy = ty - |dy|.
            # If vx > 0 and vy > 0, they overlap.
            # Penalty term: (vx * vy)^2 ? Or vx^2 * vy^2?
            # Let's use (max(0, vx))^2 * (max(0, vy))^2.
            
            vx = tx - abs(dx)
            vy = ty - abs(dy)
            
            if vx > 0 and vy > 0:
                # Penalty is (vx * vy)^2
                pen = (vx * vy) ** 2
                overlap_pen += pen
                
                # Gradient of pen w.r.t vx: 2 * vx * vy * vy = 2 * vx * vy^2
                # Gradient of pen w.r.t vy: 2 * vx^2 * vy
                dpen_dvx = 2 * vx * vy ** 2
                dpen_dvy = 2 * vx ** 2 * vy
                
                # d(vx)/dx = -sign(dx) (if dx != 0)
                # d(vx)/ds_i = 0.5, d(vx)/ds_j = 0.5
                # d(vy)/dy = -sign(dy) (if dy != 0)
                # d(vy)/ds_i = 0.5, d(vy)/ds_j = 0.5
                
                if dx > 0:
                    s_dx = 1.0
                elif dx < 0:
                    s_dx = -1.0
                else:
                    s_dx = 0.0 # Subgradient, pick 0 for stability at 0? Or random? 0 is fine.
                    
                if dy > 0:
                    s_dy = 1.0
                elif dy < 0:
                    s_dy = -1.0
                else:
                    s_dy = 0.0
                    
                # Update gradients for i
                # d(pen)/dcx_i = dpen_dvx * d(vx)/dcx_i + dpen_dvy * d(vy)/dcx_i (0)
                overlap_grad[2*i] += dpen_dvx * (-s_dx)
                # d(pen)/dcy_i = dpen_dvx * 0 + dpen_dvy * (-s_dy)
                overlap_grad[2*i+1] += dpen_dvy * (-s_dy)
                # d(pen)/ds_i = dpen_dvx * 0.5 + dpen_dvy * 0.5
                overlap_grad[2*n+i] += 0.5 * (dpen_dvx + dpen_dvy)
                
                # Update gradients for j
                # d(pen)/dcx_j = dpen_dvx * d(vx)/dcx_j = dpen_dvx * s_dx
                overlap_grad[2*j] += dpen_dvx * s_dx
                # d(pen)/dcy_j = dpen_dvy * s_dy
                overlap_grad[2*j+1] += dpen_dvy * s_dy
                # d(pen)/ds_j = 0.5 * (dpen_dvx + dpen_dvy)
                overlap_grad[2*n+j] += 0.5 * (dpen_dvx + dpen_dvy)
                
    # Total Energy
    lambda_bound = 1000.0
    lambda_overlap = 1000.0
    
    energy = -np.sum(sides) + lambda_bound * boundary_pen + lambda_overlap * overlap_pen
    
    # Total Gradient
    grad = np.zeros_like(x)
    # d(-sum(sides))/ds_i = -1
    grad[2*n:] = -1.0
    grad += lambda_bound * boundary_grad
    grad += lambda_overlap * overlap_grad
    
    return energy, grad

def _solve_continuous(seeds, time_limit):
    """
    Run L-BFGS-B from multiple seeds.
    """
    n = 10
    best_val = -np.inf
    best_x = None
    start_time = time.time()
    
    for seed_x in seeds:
        if time.time() - start_time > time_limit:
            break
            
        # Define bounds
        # cx, cy in [0.01, 0.99] (approx, will be penalized if outside)
        # s in [0.01, 1.0]
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0)) # cx
            bounds.append((0.0, 1.0)) # cy
            bounds.append((0.01, 1.0)) # s
        
        try:
            res = minimize(
                _energy_and_grad,
                seed_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
            
            x_opt = res.x
            centers = x_opt[:2*n].reshape(n, 2)
            sides = x_opt[2*n:]
            
            # Post-process: ensure strict validity
            # Clip to bounds
            for i in range(n):
                # Clip center so square is inside [0,1]
                s_i = sides[i]
                min_cx = s_i / 2
                max_cx = 1 - s_i / 2
                if min_cx > max_cx:
                    # Square too big? Should not happen if s<=1
                    sides[i] = 0.01
                    centers[i] = [0.5, 0.5]
                    continue
                centers[i, 0] = np.clip(centers[i, 0], min_cx, max_cx)
                centers[i, 1] = np.clip(centers[i, 1], min_cx, max_cx) # Same for y
                
            # Check validity
            if _is_valid(centers, sides):
                val = np.sum(sides)
                if val > best_val:
                    best_val = val
                    best_x = (centers, sides)
        except Exception as e:
            pass
            
    return best_x, best_val

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 570
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Grid-like seeds
    for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
        x = np.zeros(3*n)
        # Place in a loose grid
        for i in range(n):
            row = i // 5
            col = i % 5
            x[2*i] = 0.1 + col * 0.2
            x[2*i+1] = 0.1 + row * 0.2
            x[2*n+i] = s
        seeds.append(x)
        
    # 2. Random seeds
    np.random.seed(42)
    for _ in range(20):
        x = np.zeros(3*n)
        x[:2*n] = np.random.rand(2*n) * 0.8 + 0.1
        x[2*n:] = np.random.rand(n) * 0.3 + 0.1
        seeds.append(x)
        
    # 3. "Shelf" seeds (good starting points from previous knowledge)
    # Example: 5 squares in bottom row, 5 in top row
    x = np.zeros(3*n)
    for i in range(5):
        x[2*i] = 0.1 + i * 0.2
        x[2*i+1] = 0.1
        x[2*n+i] = 0.2
    for i in range(5):
        x[2*(i+5)] = 0.1 + i * 0.2
        x[2*(i+5)+1] = 0.3
        x[2*n+(i+5)] = 0.2
    seeds.append(x)
    
    # Run optimization
    best_result, best_val = _solve_continuous(seeds, time_limit=deadline - start_time - 10)
    
    if best_result is None:
        # Fallback to a valid simple layout
        centers = np.array([[0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
                            [0.1, 0.3], [0.3, 0.3], [0.5, 0.3], [0.7, 0.3], [0.9, 0.3]])
        sides = np.full(10, 0.1)
    else:
        centers, sides = best_result
        
    # Final safety check and correction
    if not _is_valid(centers, sides):
        # Try to shrink and center
        for i in range(10):
            s = sides[i]
            # Ensure inside
            min_c = s / 2
            max_c = 1 - s / 2
            if min_c > max_c:
                sides[i] = 0.01
                centers[i] = [0.5, 0.5]
            else:
                centers[i, 0] = np.clip(centers[i, 0], min_c, max_c)
                centers[i, 1] = np.clip(centers[i, 1], min_c, max_c)
                
        if not _is_valid(centers, sides):
            # Last resort: tiny squares in corners
            centers = np.array([[0.05, 0.05], [0.15, 0.05], [0.25, 0.05], [0.35, 0.05], [0.45, 0.05],
                                [0.05, 0.15], [0.15, 0.15], [0.25, 0.15], [0.35, 0.15], [0.45, 0.15]])
            sides = np.full(10, 0.09)
            
    return centers, sides

if __name__ == "__main__":
    c, s = squares_in_square10()
    print(f"Sum of sides: {np.sum(s)}")
    print(f"Valid: {_is_valid(c, s)}")
