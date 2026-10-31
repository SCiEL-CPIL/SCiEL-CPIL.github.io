"""Circle packing in perimeter-4 rectangle using hexagonal initialization and SLSQP refinement."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START

def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check validity of the packing."""
    if W <= 0 or H <= 0:
        return False
    # Wall constraints
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    # Pairwise non-overlap
    n = len(cx)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _force_directed_relax(cx, cy, r, W, H, max_iter=500, dt=0.005):
    """
    Relax positions to satisfy constraints.
    Radii are fixed during this phase to ensure we find a valid layout for the given radii.
    """
    cx = cx.copy()
    cy = cy.copy()
    n = len(cx)
    
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 50
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 50
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 50
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 50
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d_sq = dx * dx + dy * dy
                min_d = r[i] + r[j]
                if d_sq < min_d * min_d:
                    d = np.sqrt(d_sq) if d_sq > 1e-12 else 1e-12
                    if d < 1e-12:
                        # Random nudge if perfectly overlapping
                        fx[i] += np.random.uniform(-1, 1)
                        fy[i] += np.random.uniform(-1, 1)
                        fx[j] -= np.random.uniform(-1, 1)
                        fy[j] -= np.random.uniform(-1, 1)
                    else:
                        overlap = min_d - d
                        fx[i] += (dx / d) * overlap * 10
                        fy[i] += (dy / d) * overlap * 10
                        fx[j] -= (dx / d) * overlap * 10
                        fy[j] -= (dy / d) * overlap * 10
        
        cx += dt * fx
        cy += dt * fy
        
        # Clamp to rectangle to prevent flying out
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
        
        # Check if converged
        if np.max(np.abs(fx)) < 1e-6 and np.max(np.abs(fy)) < 1e-6:
            break
            
    return cx, cy

def _hexagonal_init(W, H):
    """
    Generate a hexagonal packing of N circles scaled to fit roughly in W x H.
    """
    # Hexagonal packing: rows are offset.
    # For N=21, a 5x5 grid is 25, so we take 21.
    # Or 4x6 = 24, take 21.
    # Let's use a 5x5 grid pattern but only place 21.
    
    cols = 5
    rows = 5
    
    # Hexagonal geometry
    # Radius r. Width of row = cols * 2r. Height of row = sqrt(3) * r.
    # Total width approx 2r * cols. Total height approx sqrt(3) * r * (rows - 0.5) + r.
    
    # We want to estimate a base radius that fits.
    # Let's just place them in a normalized 1x1 space first, then scale.
    # Or better, pick a radius, compute bounding box, scale to W, H.
    
    r_est = 0.1
    cx = []
    cy = []
    
    for row in range(rows):
        for col in range(cols):
            if len(cx) >= N:
                break
            # Hexagonal offset
            x = (col + 0.5) * 2 * r_est + (0.5 * 2 * r_est if row % 2 == 1 else 0)
            y = (row + 0.5) * np.sqrt(3) * r_est
            
            cx.append(x)
            cy.append(y)
            
    cx = np.array(cx[:N])
    cy = np.array(cy[:N])
    
    # Compute bounding box of this configuration
    min_x = np.min(cx) - r_est
    max_x = np.max(cx) + r_est
    min_y = np.min(cy) - r_est
    max_y = np.max(cy) + r_est
    
    w_box = max_x - min_x
    h_box = max_y - min_y
    
    # Scale to fit W and H
    # We want to fill the space, so scale by min(W/w_box, H/h_box)
    scale = min(W / w_box, H / h_box)
    
    cx_scaled = (cx - min_x) * scale
    cy_scaled = (cy - min_y) * scale
    r_scaled = np.full(N, r_est * scale)
    
    # Center in the rectangle
    cx_scaled += (W - w_box * scale) / 2
    cy_scaled += (H - h_box * scale) / 2
    
    return cx_scaled, cy_scaled, r_scaled

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_val = -float('inf')
    best_result = None
    
    # Try a range of aspect ratios W (since H = 2 - W)
    # W ranges from 0.5 to 1.5 (symmetric around 1.0)
    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0:
            continue
        
        # 1. Hexagonal Initialization
        cx, cy, r = _hexagonal_init(W, H)
        
        # 2. Force Directed Relaxation to ensure validity
        # We relax positions for the initial radii.
        cx, cy = _force_directed_relax(cx, cy, r, W, H, max_iter=200)
        
        # Check if valid
        if not _is_valid(cx, cy, r, W, H):
            # Try a smaller radius if relaxation failed? 
            # Or just skip this aspect ratio for now.
            # Actually, if FDR fails, the geometry is likely too tight.
            # Let's try a smaller initial radius.
            r = r * 0.8
            cx, cy = _force_directed_relax(cx, cy, r, W, H, max_iter=200)
            if not _is_valid(cx, cy, r, W, H):
                continue
        
        # 3. SLSQP Optimization
        # Variables: cx (N), cy (N), r (N)
        # Objective: -sum(r)
        # Constraints:
        # 1. cx[i] - r[i] >= 0
        # 2. cy[i] - r[i] >= 0
        # 3. cx[i] + r[i] <= W
        # 4. cy[i] + r[i] <= H
        # 5. (cx[i]-cx[j])^2 + (cy[i]-cy[j])^2 >= (r[i]+r[j])^2
        
        x0 = np.concatenate([cx, cy, r])
        
        def objective(x):
            cx = x[:N]
            cy = x[N:2*N]
            r = x[2*N:3*N]
            return -np.sum(r)
            
        def constraint_wall_bottom(x):
            cx, cy, r = x[:N], x[N:2*N], x[2*N:3*N]
            return cx - r
            
        def constraint_wall_left(x):
            cx, cy, r = x[:N], x[N:2*N], x[2*N:3*N]
            return cy - r
            
        def constraint_wall_top(x):
            cx, cy, r = x[:N], x[N:2*N], x[2*N:3*N]
            return W - (cx + r)
            
        def constraint_wall_right(x):
            cx, cy, r = x[:N], x[N:2*N], x[2*N:3*N]
            return H - (cy + r)
            
        constraints = []
        # Wall constraints
        constraints.append({'type': 'ineq', 'fun': constraint_wall_bottom})
        constraints.append({'type': 'ineq', 'fun': constraint_wall_left})
        constraints.append({'type': 'ineq', 'fun': constraint_wall_top})
        constraints.append({'type': 'ineq', 'fun': constraint_wall_right})
        
        # Non-overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                def make_non_overlap(i, j):
                    def non_overlap(x):
                        cx, cy, r = x[:N], x[N:2*N], x[2*N:3*N]
                        dx = cx[i] - cx[j]
                        dy = cy[i] - cy[j]
                        dist_sq = dx*dx + dy*dy
                        min_dist = r[i] + r[j]
                        return dist_sq - min_dist*min_dist
                    return non_overlap
                constraints.append({'type': 'ineq', 'fun': make_non_overlap(i, j)})
                
        bounds = []
        for i in range(N):
            bounds.append((0, W)) # cx
        for i in range(N):
            bounds.append((0, H)) # cy
        for i in range(N):
            bounds.append((1e-4, min(W, H)/2)) # r
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, 
                           options={'maxiter': 300, 'ftol': 1e-8})
            
            if res.success or res.fun < objective(x0):
                cx_opt = res.x[:N]
                cy_opt = res.x[N:2*N]
                r_opt = res.x[2*N:3*N]
                
                # Final validation
                if _is_valid(cx_opt, cy_opt, r_opt, W, H):
                    val = float(np.sum(r_opt))
                    if val > best_val:
                        best_val = val
                        best_result = np.column_stack([cx_opt, cy_opt, r_opt])
                        
        except Exception:
            pass

    # Fallback to seed if no valid result found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        # Simple grid in 1x1 square (perimeter 4)
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result

# EVOLVE-BLOCK-END
