"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 21
DEADLINE = 115  # seconds, safety margin before 120s limit


def _initial_hex_config():
    """Generate a hexagonal lattice configuration for 21 circles."""
    # Hexagonal packing: rows with alternating offsets
    # For 21 circles, try arrangements like 5-4-5-4-3 or similar
    # Let's use a more systematic approach
    
    # Try different row arrangements
    best_config = None
    best_score = -1
    
    # Row arrangements that sum to 21
    arrangements = [
        [5, 4, 5, 4, 3],
        [4, 5, 4, 5, 3],
        [5, 5, 5, 3, 3],
        [4, 4, 4, 4, 5],
        [3, 4, 5, 4, 5],
        [5, 4, 4, 4, 4],
        [4, 5, 4, 4, 4],
        [3, 5, 4, 4, 5],
        [5, 3, 5, 4, 4],
        [4, 4, 5, 4, 4],
    ]
    
    for arr in arrangements:
        if sum(arr) != N:
            continue
        
        # Build hexagonal positions
        pts = []
        row_y = 0
        for row_idx, count in enumerate(arr):
            # Hexagonal spacing
            row_offset = 0.5 if row_idx % 2 == 1 else 0.0
            for i in range(count):
                x = i + row_offset
                y = row_y
                pts.append((x, y))
            row_y += np.sqrt(3) / 2  # hex row height
        
        pts = np.array(pts)
        
        # Normalize to fit in a reasonable box
        x_range = pts[:, 0].max() - pts[:, 0].min() + 2
        y_range = pts[:, 1].max() - pts[:, 1].min() + 2
        
        # Scale to perimeter 4: 2*(w+h) = 4, so w+h = 2
        # We'll scale later during optimization
        
        if best_config is None or len(pts) == N:
            best_config = pts
    
    if best_config is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        best_config = np.array([[(i + 0.5) / side, (j + 0.5) / side] 
                                for i in range(side) for j in range(side)])[:N]
    
    return best_config


def _objective(params, n_circles):
    """
    Objective: maximize sum of radii subject to constraints.
    params: [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    """
    n = n_circles
    centers = params[0:2*n].reshape(n, 2)
    radii = params[2*n:3*n]
    
    # Penalty for negative radii
    neg_r_pen = np.sum(np.maximum(0, -radii)) ** 2
    
    # Penalty for circles going out of bounds
    # We'll determine the bounding box from the configuration
    x_min = np.min(centers[:, 0] - radii)
    x_max = np.max(centers[:, 0] + radii)
    y_min = np.min(centers[:, 1] - radii)
    y_max = np.max(centers[:, 1] + radii)
    
    w = x_max - x_min
    h = y_max - y_min
    
    # Perimeter constraint: 2*(w+h) <= 4, so w+h <= 2
    perim_pen = np.maximum(0, (w + h) - 2.0) ** 2
    
    # Overlap penalty: for each pair, dist >= ri + rj
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            min_d = radii[i] + radii[j]
            if d < min_d:
                overlap_pen += (min_d - d) ** 2
    
    # We want to maximize sum of radii, so minimize negative
    return -(np.sum(radii)) + 100.0 * overlap_pen + 100.0 * perim_pen + 10.0 * neg_r_pen


def _objective_grad(params, n_circles):
    """Gradient of the objective."""
    n = n_circles
    centers = params[0:2*n].reshape(n, 2)
    radii = params[2*n:3*n]
    
    grad = np.zeros_like(params)
    
    # Compute bounding box
    x_min = np.min(centers[:, 0] - radii)
    x_max = np.max(centers[:, 0] + radii)
    y_min = np.min(centers[:, 1] - radii)
    y_max = np.max(centers[:, 1] + radii)
    
    w = x_max - x_min
    h = y_max - y_min
    perim_viol = max(0, (w + h) - 2.0)
    
    # Overlap gradient
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            d = np.sqrt(dx*dx + dy*dy) + 1e-10
            min_d = radii[i] + radii[j]
            if d < min_d:
                diff = min_d - d
                # Gradient w.r.t. centers[i]
                grad[2*i] += 200.0 * diff * dx / d
                grad[2*i+1] += 200.0 * diff * dy / d
                # Gradient w.r.t. centers[j]
                grad[2*j] += 200.0 * diff * (-dx) / d
                grad[2*j+1] += 200.0 * diff * (-dy) / d
                # Gradient w.r.t. radii
                grad[2*n+i] += 200.0 * diff
                grad[2*n+j] += 200.0 * diff
    
    # Negative radii penalty
    for i in range(n):
        if radii[i] < 0:
            grad[2*n+i] += 20.0 * (-radii[i])
    
    # Perimeter penalty - this is complex, use numerical gradient for simplicity
    # For now, just add a small push
    if perim_viol > 0:
        # Push all radii down slightly
        for i in range(n):
            grad[2*n+i] += 100.0 * perim_viol
    
    # Main objective: maximize sum(radii)
    for i in range(n):
        grad[2*n+i] -= 1.0
    
    return grad


def _evaluate_config(centers, radii):
    """Evaluate a configuration: return (valid, score)."""
    n = len(radii)
    if np.any(radii < 0):
        return False, 0.0
    
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            if d < radii[i] + radii[j] - 1e-9:
                return False, 0.0
    
    # Check perimeter
    x_min = np.min(centers[:, 0] - radii)
    x_max = np.max(centers[:, 0] + radii)
    y_min = np.min(centers[:, 1] - radii)
    y_max = np.max(centers[:, 1] + radii)
    
    w = x_max - x_min
    h = y_max - y_min
    
    if 2*(w + h) > 4.0 + 1e-9:
        return False, 0.0
    
    return True, np.sum(radii)


def construct():
    start_time = time.time()
    
    # Initial configuration
    hex_pts = _initial_hex_config()
    n = N
    
    # Initial radii: small
    r0 = 0.05
    centers0 = hex_pts.copy()
    
    # Normalize initial positions to be in [0, 1] x [0, 1]
    x_range = centers0[:, 0].max() - centers0[:, 0].min()
    y_range = centers0[:, 1].max() - centers0[:, 1].min()
    if x_range > 0:
        centers0[:, 0] = (centers0[:, 0] - centers0[:, 0].min()) / x_range
    if y_range > 0:
        centers0[:, 1] = (centers0[:, 1] - centers0[:, 1].min()) / y_range
    
    # Initial params: [x1,y1,r1, x2,y2,r2, ...]
    params0 = np.zeros(3*n)
    params0[0:2*n:2] = centers0[:, 0]
    params0[1:2*n:2] = centers0[:, 1]
    params0[2*n:] = r0
    
    best_score = 0.0
    best_params = params0.copy()
    
    # Try multiple restarts with different initial conditions
    restarts = [
        params0.copy(),
    ]
    
    # Add some perturbed restarts
    for _ in range(3):
        p = params0.copy()
        p[0:2*n] += np.random.normal(0, 0.05, size=2*n)
        p[2*n:] = 0.03 + np.random.uniform(0, 0.02, size=n)
        restarts.append(p)
    
    for restart_idx, p0 in enumerate(restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        try:
            result = minimize(
                _objective,
                p0,
                args=(n,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            p_opt = result.x
            centers_opt = p_opt[0:2*n].reshape(n, 2)
            radii_opt = p_opt[2*n:]
            
            valid, score = _evaluate_config(centers_opt, radii_opt)
            if valid and score > best_score:
                best_score = score
                best_params = p_opt.copy()
                
        except Exception:
            continue
    
    # Polish pass with tighter tolerances
    if time.time() - start_time < DEADLINE - 5:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(n,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
            )
            
            p_opt = result.x
            centers_opt = p_opt[0:2*n].reshape(n, 2)
            radii_opt = p_opt[2*n:]
            
            valid, score = _evaluate_config(centers_opt, radii_opt)
            if valid and score > best_score:
                best_score = score
                best_params = p_opt.copy()
        except Exception:
            pass
    
    # Final result
    centers = best_params[0:2*n].reshape(n, 2)
    radii = best_params[2*n:]
    
    # Ensure validity - if not valid, fall back to safe grid
    valid, score = _evaluate_config(centers, radii)
    if not valid:
        # Safe fallback: grid packing
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            r = 0.5 / side - 1e-3
            result[i] = [(col + 0.5) / side, (row + 0.5) / side, r]
        return result
    
    # Clip tiny negative radii
    radii = np.maximum(radii, 0)
    
    result = np.zeros((N, 3))
    result[:, 0] = centers[:, 0]
    result[:, 1] = centers[:, 1]
    result[:, 2] = radii
    
    return result
# EVOLVE-BLOCK-END
