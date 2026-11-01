"""
Circle packing in perimeter-4 rectangle: Multi-Start SLSQP + Force Relax + Polish.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
PERIMETER = 4.0
DEADLINE_MARGIN = 10.0  # seconds

def _force_relax(positions, radii, W, H, steps=500, dt=0.01):
    """
    Simple force-directed relaxation to untangle overlapping circles.
    Circles repel each other and are attracted to the center of the rectangle.
    """
    pos = positions.copy()
    r = radii.copy()
    n = len(r)
    
    for step in range(steps):
        forces = np.zeros((n, 2))
        
        # Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                
                if d < 1e-6:
                    # Random nudge if exactly overlapping
                    dx = np.random.uniform(-0.1, 0.1)
                    dy = np.random.uniform(-0.1, 0.1)
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < 1e-6:
                        continue
                        
                if d < min_d:
                    # Repulsive force proportional to overlap
                    overlap = min_d - d
                    force_mag = 10.0 * overlap
                    fx = force_mag * dx / d
                    fy = force_mag * dy / d
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Attraction to center
        cx, cy = W / 2.0, H / 2.0
        for i in range(n):
            forces[i, 0] += 0.1 * (cx - pos[i, 0])
            forces[i, 1] += 0.1 * (cy - pos[i, 1])
            
        # Update positions
        for i in range(n):
            pos[i, 0] += forces[i, 0] * dt
            pos[i, 1] += forces[i, 1] * dt
            
            # Keep within bounds (soft)
            if pos[i, 0] < r[i]: pos[i, 0] = r[i]
            if pos[i, 0] > W - r[i]: pos[i, 0] = W - r[i]
            if pos[i, 1] < r[i]: pos[i, 1] = r[i]
            if pos[i, 1] > H - r[i]: pos[i, 1] = H - r[i]
            
    return pos

def _objective(z, W, H, penalty=1000.0):
    """
    Objective function for SLSQP.
    z = [x_1..x_n, y_1..y_n, r_1..r_n]
    We maximize sum(r_i) by minimizing -sum(r_i) + penalties.
    """
    x = z[0:N]
    y = z[N:2*N]
    r = z[2*N:3*N]
    
    # Basic bounds check
    if np.any(r < 0):
        return 1e6
        
    obj = -np.sum(r)
    
    # Boundary penalties
    # Left: x >= r
    viol_left = np.maximum(r - x, 0.0)
    # Right: x + r <= W
    viol_right = np.maximum(x + r - W, 0.0)
    # Bottom: y >= r
    viol_bottom = np.maximum(r - y, 0.0)
    # Top: y + r <= H
    viol_top = np.maximum(y + r - H, 0.0)
    
    obj += penalty * (np.sum(viol_left**2) + np.sum(viol_right**2) + 
                      np.sum(viol_bottom**2) + np.sum(viol_top**2))
    
    # Pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            min_d = r[i] + r[j]
            if d < min_d:
                overlap = min_d - d
                obj += penalty * overlap * overlap
                
    return obj

def _slsqp_optimize(x, y, r, W, H, deadline):
    """
    Run SLSQP to optimize positions and radii.
    """
    z0 = np.concatenate([x, y, r])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, W))  # x
        bounds.append((0.0, H))  # y
        bounds.append((0.01, min(W, H)/2.0)) # r
    
    def obj(z):
        return _objective(z, W, H)
    
    # Constraints: none explicitly, handled by penalty + bounds
    # But we can add simple linear constraints if needed, penalty is usually enough for SLSQP
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-8})
        if res.success:
            return res.x
        else:
            return z0
    except Exception:
        return z0

def _polish(x, y, r, W, H, deadline):
    """
    Local polishing:
    1. Fix positions, expand radii to max feasible.
    2. Fix radii, adjust positions slightly to allow further expansion.
    """
    r = r.copy()
    x = x.copy()
    y = y.copy()
    
    # Pass 1: Expand radii
    for _ in range(5):
        changed = False
        for i in range(N):
            max_r = min(x[i], W - x[i], y[i], H - y[i])
            for j in range(N):
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, d - r[j])
            if max_r > r[i] + 1e-9:
                r[i] = max_r * 0.9999
                changed = True
        if not changed:
            break
            
    # Pass 2: Nudge positions to create space
    for _ in range(20):
        if time.time() > deadline:
            break
        # Try small random nudges
        best_val = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        for _ in range(10):
            # Random nudge to a random circle
            idx = np.random.randint(N)
            dx = np.random.uniform(-0.05, 0.05)
            dy = np.random.uniform(-0.05, 0.05)
            
            new_x = x.copy()
            new_y = y.copy()
            new_r = r.copy()
            
            new_x[idx] += dx
            new_y[idx] += dy
            
            # Check validity and expand radii
            # Simple check: is it valid?
            valid = True
            # Bounds
            if new_x[idx] < 0 or new_x[idx] > W or new_y[idx] < 0 or new_y[idx] > H:
                valid = False
            
            if valid:
                # Try to expand radii
                for i in range(N):
                    max_r = min(new_x[i], W - new_x[i], new_y[i], H - new_y[i])
                    for j in range(N):
                        if i == j: continue
                        dx_ij = new_x[i] - new_x[j]
                        dy_ij = new_y[i] - new_y[j]
                        d = np.sqrt(dx_ij*dx_ij + dy_ij*dy_ij)
                        max_r = min(max_r, d - new_r[j])
                    if max_r > new_r[i]:
                        new_r[i] = max_r * 0.9999
                        
                val = np.sum(new_r)
                if val > best_val:
                    best_val = val
                    best_x, best_y, best_r = new_x, new_y, new_r
                    
        x, y, r = best_x, best_y, best_r
        
        # Re-expand radii to max
        for i in range(N):
            max_r = min(x[i], W - x[i], y[i], H - y[i])
            for j in range(N):
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, d - r[j])
            if max_r > r[i]:
                r[i] = max_r * 0.9999
                
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_result = None
    
    # Try different aspect ratios
    # Perimeter = 4 => w + h = 2
    aspect_ratios = [1.0, 1.2, 0.83, 1.5, 0.67, 1.8, 0.56, 2.0, 0.5]
    
    # Random seeds for reproducibility but diversity
    np.random.seed(42)
    
    for w in aspect_ratios:
        if time.time() > deadline - 10:
            break
        h = 2.0 - w
        if h < 0.3 or w < 0.3:
            continue
            
        # Multiple initializations per aspect ratio
        for init_idx in range(3):
            if time.time() > deadline - 5:
                break
                
            # Initialize positions
            # Simple grid-like init with jitter
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.full(N, 0.05)
            
            for i in range(N):
                row = i // side
                col = i % side
                # Add jitter to break symmetry
                x[i] = (col + 0.5) * w / side + np.random.uniform(-0.1, 0.1) * w
                y[i] = (row + 0.5) * h / side + np.random.uniform(-0.1, 0.1) * h
                
                # Ensure within bounds
                x[i] = np.clip(x[i], 0.05, w - 0.05)
                y[i] = np.clip(y[i], 0.05, h - 0.05)
                
            # Force relaxation to untangle
            x_relaxed = _force_relax(x, r, w, h, steps=200)
            
            # SLSQP Optimization
            z_opt = _slsqp_optimize(x_relaxed, y, r, w, h, deadline)
            x_opt = z_opt[0:N]
            y_opt = z_opt[N:2*N]
            r_opt = z_opt[2*N:3*N]
            
            # Polish
            x_pol, y_pol, r_pol = _polish(x_opt, y_opt, r_opt, w, h, deadline)
            
            # Validate and calculate score
            # Check validity
            valid = True
            for i in range(N):
                if x_pol[i] < -1e-9 or x_pol[i] > w + 1e-9 or y_pol[i] < -1e-9 or y_pol[i] > h + 1e-9:
                    valid = False
                if r_pol[i] <= 0:
                    valid = False
                    
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x_pol[i] - x_pol[j]
                        dy = y_pol[i] - y_pol[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < r_pol[i] + r_pol[j] - 1e-9:
                            valid = False
                            break
                    if not valid: break
                    
            if valid:
                score = np.sum(r_pol)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([x_pol, y_pol, r_pol])
    
    if best_result is None:
        # Fallback
        W, H = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        pos = np.array([[(i + 0.5) * W / side, (j + 0.5) * H / side] 
                        for i in range(side) for j in range(side)])[:N]
        r = np.full(N, 0.1)
        best_result = np.column_stack([pos, r])
        
    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Score: {np.sum(res[:, 2])}")
