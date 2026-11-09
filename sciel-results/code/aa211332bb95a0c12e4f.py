"""Circle packing in unit square using L-BFGS-B with smooth penalty."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _is_valid(circles, eps=1e-9):
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(x < -eps) or np.any(x > 1 + eps) or np.any(y < -eps) or np.any(y > 1 + eps):
        return False
    if np.any(r < -eps):
        return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - eps:
                return False
    return True

def _sum_radii(circles):
    return float(np.sum(circles[:, 2]))

def _objective_and_grad(params):
    """
    Objective: Maximize sum(r) - Penalty
    Penalty:
      1. Wall violations: (min(x, 1-x, y, 1-y) - r)^2 for each circle
      2. Overlaps: (r_i + r_j - dist_ij)^2 for each pair where dist < r_i + r_j
    
    We return -objective for minimization.
    """
    x = params[0:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    # Ensure radii are positive for distance calc stability in penalty
    # But in optimization, r can go negative. We penalize r < 0 heavily? 
    # Actually, L-BFGS-B handles bounds. We will use bounds to keep r >= 0.
    # However, the penalty function should be smooth.
    
    obj = -np.sum(r)
    
    # Wall penalties
    # dist to walls: min(x, 1-x, y, 1-y)
    # If r > dist_to_wall, penalty.
    # Smooth approximation: max(0, r - dist)^2
    # Let d_w = min(x, 1-x, y, 1-y)
    # violation = max(0, r - d_w)
    # This is not smooth at r = d_w.
    # Use softplus or just square of max? 
    # L-BFGS-B prefers smooth gradients. 
    # Alternative: Since we have bounds on x,y in [0,1] and r in [0, 0.5],
    # and we want to pack tightly, we can define the penalty as:
    # P_wall = sum( (max(0, r - min(x, 1-x, y, 1-y)))^2 )
    # The gradient of max(0, u)^2 is 2*max(0,u)*du/dx. This is continuous but not C2.
    # L-BFGS-B can handle this, but might be slow.
    
    # Better: Use a smooth approximation for the "distance" constraint.
    # Or, simpler: Just use the hard constraint logic in a penalty that is always active?
    # No, that ruins the packing.
    
    # Let's use a large multiplier method or just a simple penalty.
    # Given the time limit, a simple penalty with L-BFGS-B is standard.
    
    # Wall constraint penalty
    dw = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    wall_viol = np.maximum(0, r - dw)
    obj += np.sum(wall_viol**2)
    
    # Pairwise overlap penalty
    # dist_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
    # overlap = max(0, ri + rj - dist_ij)
    # obj += sum(overlap^2)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2 + 1e-12) # 1e-12 to avoid div by zero in grad
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Calculate overlap for all pairs
    # r_i + r_j
    r_sum = r[:, None] + r[None, :]
    overlap = np.maximum(0, r_sum - dist)
    obj += np.sum(overlap[mask]**2)
    
    # Gradient calculation
    grad = np.zeros(3*N)
    
    # Gradient of -sum(r)
    grad[2*N:3*N] = -1.0
    
    # Gradient of wall penalty
    # d/dx (max(0, r - min(x, 1-x, y, 1-y))^2)
    # Let w = min(x, 1-x, y, 1-y)
    # if r > w, viol = r - w
    # grad_x = 2 * viol * d/dx (r - w) = -2 * viol * d/dx w
    # d/dx w is -1 if x < 0.5, 1 if x > 0.5, 0 if y is the min.
    # This is messy. Let's compute numerically? No, analytic is better if possible.
    # Or just use finite differences for the penalty part? 
    # L-BFGS-B with analytic gradient is much faster.
    
    # Let's refine the wall gradient.
    # For each circle i:
    # w_i = min(x_i, 1-x_i, y_i, 1-y_i)
    # viol_i = max(0, r_i - w_i)
    # If viol_i > 0:
    #   grad_x_i += -2 * viol_i * d(w_i)/dx_i
    #   grad_y_i += -2 * viol_i * d(w_i)/dy_i
    #   grad_r_i +=  2 * viol_i * d(w_i)/dr_i  (d(w_i)/dr_i is 0, so no change to grad_r from wall)
    # Wait, obj += viol^2. d(obj)/dr = 2*viol * d(viol)/dr = 2*viol * 1 = 2*viol.
    # d(obj)/dx = 2*viol * d(viol)/dx = 2*viol * (-d(w)/dx).
    
    for i in range(N):
        if wall_viol[i] > 0:
            # Determine which wall is limiting
            if x[i] <= 1-x[i] and x[i] <= y[i] and x[i] <= 1-y[i]:
                # Left wall limiting
                dw_dx = 1.0
                dw_dy = 0.0
            elif 1-x[i] <= x[i] and 1-x[i] <= y[i] and 1-x[i] <= 1-y[i]:
                # Right wall limiting
                dw_dx = -1.0
                dw_dy = 0.0
            elif y[i] <= x[i] and y[i] <= 1-x[i] and y[i] <= 1-y[i]:
                # Bottom wall limiting
                dw_dx = 0.0
                dw_dy = 1.0
            else:
                # Top wall limiting
                dw_dx = 0.0
                dw_dy = -1.0
            
            grad[i] += -2.0 * wall_viol[i] * dw_dx
            grad[N+i] += -2.0 * wall_viol[i] * dw_dy
            grad[2*N+i] += 2.0 * wall_viol[i]
    
    # Gradient of overlap penalty
    # For pair (i,j) with i<j:
    # viol_ij = max(0, r_i + r_j - dist_ij)
    # If viol_ij > 0:
    #   d(obj)/dx_i = 2*viol_ij * (-d(dist)/dx_i) = 2*viol_ij * (-(x_i-x_j)/dist)
    #   d(obj)/dx_j = 2*viol_ij * (-d(dist)/dx_j) = 2*viol_ij * (-(x_j-x_i)/dist)
    #   d(obj)/dr_i = 2*viol_ij * 1
    #   d(obj)/dr_j = 2*viol_ij * 1
    
    # We can vectorize this partially or loop. Loop is fine for N=26.
    for i in range(N):
        for j in range(i+1, N):
            if overlap[i, j] > 0:
                d = dist[i, j]
                # grad for x_i
                grad[i] += 2.0 * overlap[i, j] * (-(x[i] - x[j]) / d)
                # grad for x_j
                grad[j] += 2.0 * overlap[i, j] * (-(x[j] - x[i]) / d)
                
                # grad for y_i
                grad[N+i] += 2.0 * overlap[i, j] * (-(y[i] - y[j]) / d)
                # grad for y_j
                grad[N+j] += 2.0 * overlap[i, j] * (-(y[j] - y[i]) / d)
                
                # grad for r_i
                grad[2*N+i] += 2.0 * overlap[i, j]
                # grad for r_j
                grad[2*N+j] += 2.0 * overlap[i, j]
                
    return obj, grad

def _init_random_valid(seed):
    """Generate a random valid initial configuration."""
    rng = np.random.RandomState(seed)
    # Try to place circles randomly with small radius first
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.05)
    return np.column_stack([x, y, r])

def _init_grid():
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([cx, cy, r])
    return np.array(cells)

def construct():
    deadline = time.time() + 110
    best_circles = None
    best_score = -np.inf
    
    # Bounds for L-BFGS-B
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius in unit square)
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    
    # Try multiple initializations
    inits = []
    inits.append(_init_grid())
    for s in [1, 7, 42, 99, 2024, 555]:
        inits.append(_init_random_valid(s))
    
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        # Flatten params
        x0 = np.concatenate([init[:, 0], init[:, 1], init[:, 2]])
        
        # Run L-BFGS-B
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
        except Exception:
            continue
            
        if time.time() > deadline - 5:
            break
            
        if res.success or res.status == 0:
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            circles = np.column_stack([x_opt, y_opt, r_opt])
            
            # Post-processing: Clip to ensure validity
            # Sometimes L-BFGS-B might leave tiny violations.
            # We can do a final "projection" or just check validity.
            # If invalid, we can try to shrink circles slightly.
            
            # Check validity
            if _is_valid(circles, eps=1e-6):
                score = _sum_radii(circles)
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
            else:
                # Try to repair: shrink radii to fix overlaps/walls
                # Simple repair: for each circle, reduce r to satisfy constraints
                # This is essentially the greedy expansion but in reverse (shrinking)
                # Or just skip this init if it fails validity.
                # Let's try a quick local repair.
                repaired = circles.copy()
                for _ in range(10):
                    changed = False
                    for i in range(N):
                        # Max allowed r at current pos
                        max_r = min(repaired[i,0], 1-repaired[i,0], repaired[i,1], 1-repaired[i,1])
                        for j in range(N):
                            if i == j: continue
                            d = np.sqrt((repaired[i,0]-repaired[j,0])**2 + (repaired[i,1]-repaired[j,1])**2)
                            max_r = min(max_r, d - repaired[j,2])
                        if repaired[i,2] > max_r + 1e-9:
                            repaired[i,2] = max(0, max_r)
                            changed = True
                    if not changed: break
                    
                if _is_valid(repaired, eps=1e-6):
                    score = _sum_radii(repaired)
                    if score > best_score:
                        best_score = score
                        best_circles = repaired.copy()

    if best_circles is None:
        # Fallback
        best_circles = _init_grid()
        
    return best_circles
