"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _eval_packing(xy, r):
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True

def _repulsion_relax(xy, r, steps=300, repulse=0.05):
    """Push overlapping circles apart to create a better initial structure."""
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    for _ in range(steps):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                dx = xy[j, 0] - xy[i, 0]
                dy = xy[j, 1] - xy[i, 1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-9:
                    dx, dy = 1e-9, 1e-9
                    d = 1e-9
                min_dist = r[i] + r[j]
                if d < min_dist + 1e-6:
                    overlap = (min_dist + 1e-6 - d)
                    # Push apart
                    nx, ny = dx/d, dy/d
                    shift = 0.5 * overlap * 0.1 # Damped
                    xy[i, 0] -= nx * shift
                    xy[i, 1] -= ny * shift
                    xy[j, 0] += nx * shift
                    xy[j, 1] += ny * shift
                    moved = True
            # Wall repulsion
            # Left
            if xy[i, 0] < r[i]:
                xy[i, 0] += (r[i] - xy[i, 0]) * 0.1
                moved = True
            # Right
            if 1.0 - xy[i, 0] < r[i]:
                xy[i, 0] -= (r[i] - (1.0 - xy[i, 0])) * 0.1
                moved = True
            # Bottom
            if xy[i, 1] < r[i]:
                xy[i, 1] += (r[i] - xy[i, 1]) * 0.1
                moved = True
            # Top
            if 1.0 - xy[i, 1] < r[i]:
                xy[i, 1] -= (r[i] - (1.0 - xy[i, 1])) * 0.1
                moved = True
        
        if not moved:
            break
            
    # Clamp
    for i in range(n):
        xy[i, 0] = np.clip(xy[i, 0], r[i], 1.0-r[i])
        xy[i, 1] = np.clip(xy[i, 1], r[i], 1.0-r[i])
        
    return xy, r

def _tension_energy(v, n):
    """
    Objective function for the tension network.
    v: [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    Minimizes overlap and wall violations, while maximizing radii.
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]
    
    # 1. Wall penalties (hard constraint push)
    # dist to walls
    d_left = x
    d_right = 1.0 - x
    d_bot = y
    d_top = 1.0 - y
    
    wall_viol = np.maximum(0, r - d_left) + np.maximum(0, r - d_right) + \
                np.maximum(0, r - d_bot) + np.maximum(0, r - d_top)
    wall_energy = np.sum(wall_viol ** 2)
    
    # 2. Pairwise overlap penalties (tension springs)
    # We want dist >= r_i + r_j
    overlap_energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            min_d = r[i] + r[j]
            if d < min_d:
                overlap_energy += (min_d - d) ** 2
                
    # 3. Maximize radii (negative sum)
    # We use a penalty formulation: minimize (wall_energy + overlap_energy) - sum(r) * lambda
    # If we just do -sum(r), the optimizer might shrink radii to avoid overlap.
    # We need a balance. A large penalty for overlap ensures validity, 
    # but we want to push radii up.
    # Let's use: Energy = wall_energy + overlap_energy - sum(r) * 10.0
    # Wait, if we maximize r, we want to minimize -r.
    # So objective = wall_energy + overlap_energy - 10.0 * np.sum(r)
    
    return wall_energy + overlap_energy - 10.0 * np.sum(r)

def _tension_grad(v, n):
    """
    Gradient of the tension energy.
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]
    
    grad = np.zeros(3*n)
    
    # Wall gradients
    # d/dx (max(0, r-x)^2) = -2(r-x) if r>x else 0
    # d/dx (max(0, r-(1-x))^2) = 2(r-(1-x)) if r>1-x else 0
    # d/dr (max(0, r-x)^2) = 2(r-x) if r>x else 0
    
    for i in range(n):
        # Left
        viol_l = r[i] - x[i]
        if viol_l > 0:
            grad[i] -= 2 * viol_l
            grad[2*n+i] += 2 * viol_l
        # Right
        viol_r = r[i] - (1.0 - x[i])
        if viol_r > 0:
            grad[i] += 2 * viol_r
            grad[2*n+i] += 2 * viol_r
        # Bottom
        viol_b = r[i] - y[i]
        if viol_b > 0:
            grad[n+i] -= 2 * viol_b
            grad[2*n+i] += 2 * viol_b
        # Top
        viol_t = r[i] - (1.0 - y[i])
        if viol_t > 0:
            grad[n+i] += 2 * viol_t
            grad[2*n+i] += 2 * viol_t
            
    # Pairwise gradients
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            min_d = r[i] + r[j]
            if d < min_d and d > 1e-9:
                viol = min_d - d
                # d/dx_i (viol^2) = 2*viol * d/dx_i (min_d - d)
                # d/dx_i d = dx/d
                # so d/dx_i (viol^2) = 2*viol * (-dx/d)
                grad[i] += -2 * viol * dx / d
                grad[n+i] += -2 * viol * dy / d
                grad[j] += 2 * viol * dx / d
                grad[n+j] += 2 * viol * dy / d
                # d/dr_i (viol^2) = 2*viol * 1
                grad[2*n+i] += 2 * viol
                grad[2*n+j] += 2 * viol
                
    # Objective gradient: -10 * sum(r)
    grad[2*n:3*n] -= 10.0
    
    return grad

def _tension_optimize(xy, r, deadline, max_iter=2000):
    """
    Run L-BFGS-B on the tension network model.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])
    
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.01, 0.5)) # Radii bounds
        
    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _tension_energy(v, n)
        
    def grad(v):
        if time.time() > deadline:
            return np.zeros(3*n)
        return _tension_grad(v, n)
        
    try:
        res = minimize(obj, v0, jac=grad, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 20000})
        v_opt = res.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]
        
        # Validate
        total, valid = _eval_packing(new_xy, new_r)
        if valid:
            return new_xy, new_r, total
    except Exception:
        pass
        
    return xy, r, _eval_packing(xy, r)[0]

def _local_search(xy, r, deadline, grid_size=50):
    """
    Standard local search to fix any remaining issues or fine-tune.
    """
    n = len(xy)
    best_xy = xy.copy()
    best_r = r.copy()
    best_total = _eval_packing(xy, r)[0]
    
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)
    
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))
                            
    for sweep in range(10):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]
            
            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()
                
            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]
            
            if new_r > r[i] + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True
                
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_xy = xy.copy()
            best_r = r.copy()
            
        if not improved:
            break
            
    return best_xy, best_r, best_total

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_xy = None
    best_r = None
    best_total = 0.0
    
    # Phase 1: Repulsion Relaxed Initialization + Tension Optimization
    # Try a few different random seeds for initialization
    for seed in range(20):
        if time.time() > deadline - 10:
            break
            
        rng = np.random.RandomState(seed)
        # Initial positions
        xy = rng.uniform(0.1, 0.9, size=(N, 2))
        # Initial radii: small enough to not overlap too much initially, but large enough to grow
        # 26 circles in unit square. Area per circle ~ 1/26. Radius ~ sqrt(1/(4*pi*26)) ~ 0.055
        r = np.full(N, 0.05)
        
        # Relax to remove initial overlaps
        xy_rel, r_rel = _repulsion_relax(xy, r, steps=200)
        
        # Optimize with Tension Network
        out_xy, out_r, total = _tension_optimize(xy_rel, r_rel, deadline, max_iter=1000)
        
        if total > best_total:
            best_total = total
            best_xy = out_xy
            best_r = out_r
            
        # If we have time, do a local search polish on the best result so far
        if time.time() < deadline - 5 and best_xy is not None:
             ls_xy, ls_r, ls_total = _local_search(best_xy, best_r, deadline, grid_size=40)
             if ls_total > best_total:
                 best_total = ls_total
                 best_xy = ls_xy
                 best_r = ls_r

    # Fallback if tension optimization failed completely
    if best_xy is None:
        side = int(np.ceil(np.sqrt(N)))
        best_xy = np.zeros((N, 2))
        best_r = np.zeros(N)
        for i in range(N):
            gx = i % side
            gy = i // side
            best_xy[i, 0] = (gx + 0.5) / side
            best_xy[i, 1] = (gy + 0.5) / side
            best_r[i] = 0.4 / side
        best_total = _eval_packing(best_xy, best_r)[0]
        
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        # Emergency fallback
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
