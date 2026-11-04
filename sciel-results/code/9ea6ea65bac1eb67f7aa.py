import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate_global(circles):
    """
    Validate the solution:
    1. 21 circles
    2. Positive radii
    3. Perimeter <= 4
    4. No overlaps
    """
    if len(circles) != N:
        return False
    
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    if np.any(rs <= 0):
        return False
        
    # Perimeter check
    left = np.min(xs - rs)
    right = np.max(xs + rs)
    bottom = np.min(ys - rs)
    top = np.max(ys + rs)
    width = right - left
    height = top - bottom
    perimeter = 2 * (width + height)
    
    if perimeter > 4 + 1e-9:
        return False
        
    # Disjoint check
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return False
    return True

def _score(circles):
    if not _validate_global(circles):
        return 0.0
    return np.sum(circles[:, 2])

def _make_hexagonal_grid(w, h, n_circles):
    """
    Generate a hexagonal packing initialization.
    """
    centers = []
    # Try to fit as many rows as possible
    # Approximate density of hex packing is ~0.9
    # Let's just generate a grid and keep the first N
    # Row 0: x, x+2r, ...
    # Row 1: x+r, x+3r, ... at y + sqrt(3)r
    
    # Estimate r
    area = w * h
    # 21 circles, area ~ 21 * pi * r^2
    # r ~ sqrt(area / (21 * pi))
    r_est = np.sqrt(area / (N * np.pi * 1.2)) # 1.2 for packing inefficiency
    
    row_y = 0
    row_idx = 0
    while len(centers) < n_circles:
        offset = 0
        if row_idx % 2 == 1:
            offset = r_est
            
        x = r_est + offset
        while x < w - r_est and len(centers) < n_circles:
            centers.append([x, row_y])
            x += 2 * r_est
        row_y += np.sqrt(3) * r_est
        row_idx += 1
        
    if len(centers) < n_circles:
        # Fallback to grid if hex fails to fill
        cols = int(np.ceil(np.sqrt(n_circles * w/h)))
        rows = int(np.ceil(n_circles / cols))
        for i in range(rows):
            for j in range(cols):
                if len(centers) >= n_circles:
                    break
                x = (j + 0.5) * w / cols
                y = (i + 0.5) * h / rows
                centers.append([x, y])
                r_est = min(w/cols, h/rows) * 0.45
                
    centers = np.array(centers[:n_circles])
    # Scale to fit
    min_x, max_x = np.min(centers[:, 0]), np.max(centers[:, 0])
    min_y, max_y = np.min(centers[:, 1]), np.max(centers[:, 1])
    
    # If it overflows, scale down
    span_x = max_x - min_x
    span_y = max_y - min_y
    
    # Target span is w and h (roughly)
    # We will let the optimizer adjust positions, but initial r should be small enough
    # to fit.
    # Let's just place them in [0, w] x [0, h]
    
    circles = np.zeros((n_circles, 3))
    circles[:, :2] = centers
    # Estimate r based on min distance
    min_dist = np.inf
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                min_dist = d
    r_init = min_dist * 0.45
    circles[:, 2] = r_init
    
    return circles

def _make_grid(w, h, n_circles):
    """
    Generate a grid initialization.
    """
    cols = int(np.ceil(np.sqrt(n_circles * w/h)))
    rows = int(np.ceil(n_circles / cols))
    
    circles = np.zeros((n_circles, 3))
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n_circles:
                break
            x = (j + 0.5) * w / cols
            y = (i + 0.5) * h / rows
            circles[count, 0] = x
            circles[count, 1] = y
            r = 0.45 * min(w/cols, h/rows)
            circles[count, 2] = r
            count += 1
            
    return circles

def _solve_aspect(w, h, deadline):
    """
    Solve for a specific aspect ratio.
    """
    if time.time() > deadline - 5:
        return None

    # Initializations
    inits = [
        _make_hexagonal_grid(w, h, N),
        _make_grid(w, h, N)
    ]
    
    # Add random init
    rand_circles = np.zeros((N, 3))
    rand_circles[:, 0] = np.random.uniform(0, w, N)
    rand_circles[:, 1] = np.random.uniform(0, h, N)
    rand_circles[:, 2] = 0.1 * min(w, h)
    inits.append(rand_circles)

    best_circles = None
    best_score = -np.inf

    for init in inits:
        if time.time() > deadline - 2:
            break

        # Relaxation to fix initial overlaps
        # Simple relaxation
        pos = init[:, :2].copy()
        rad = init[:, 2].copy()
        
        for _ in range(50):
            moved = False
            # Boundary
            for i in range(N):
                x, y, r = pos[i, 0], pos[i, 1], rad[i]
                if x - r < 0: pos[i, 0] = r; moved = True
                if x + r > w: pos[i, 0] = w - r; moved = True
                if y - r < 0: pos[i, 1] = r; moved = True
                if y + r > h: pos[i, 1] = h - r; moved = True
            # Overlap
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rad[i] + rad[j]
                    if dist < min_dist:
                        if dist < 1e-9:
                            angle = np.random.uniform(0, 2*np.pi)
                            push = (min_dist - dist) * 0.5
                            pos[i, 0] -= np.cos(angle) * push
                            pos[i, 1] -= np.sin(angle) * push
                            pos[j, 0] += np.cos(angle) * push
                            pos[j, 1] += np.sin(angle) * push
                        else:
                            overlap = min_dist - dist
                            nx = dx / dist
                            ny = dy / dist
                            pos[i, 0] -= nx * overlap * 0.5
                            pos[i, 1] -= ny * overlap * 0.5
                            pos[j, 0] += nx * overlap * 0.5
                            pos[j, 1] += ny * overlap * 0.5
                        moved = True
            if not moved:
                break
        
        relaxed = np.column_stack((pos, rad))
        
        # Optimization
        x0 = relaxed.flatten()
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        
        def objective(x):
            pos = x.reshape(-1, 3)
            xs = pos[:, 0]
            ys = pos[:, 1]
            rs = pos[:, 2]
            
            # Soft penalty for violations
            # Boundary
            bound_viol = np.concatenate([
                rs - xs,
                xs + rs - w,
                rs - ys,
                ys + rs - h
            ])
            # Overlap
            dx = xs[i_idx] - xs[j_idx]
            dy = ys[i_idx] - ys[j_idx]
            dists = np.sqrt(dx**2 + dy**2)
            min_dists = rs[i_idx] + rs[j_idx]
            overlap_viol = min_dists - dists
            
            # Penalty: sum of squares of positive violations
            penalty = np.sum(np.maximum(bound_viol, 0)**2) + np.sum(np.maximum(overlap_viol, 0)**2)
            
            # Maximize sum of radii -> Minimize -sum(r) + penalty
            return -np.sum(rs) + 100.0 * penalty

        def gradient(x):
            pos = x.reshape(-1, 3)
            xs = pos[:, 0]
            ys = pos[:, 1]
            rs = pos[:, 2]
            
            grad = np.zeros_like(x)
            
            # -sum(r) -> -1 for r components
            grad[2::3] = -1.0
            
            # Boundary Penalty Gradient
            # L = sum(max(v, 0)^2)
            # dL/dv = 2*max(v,0)
            # v1 = r - x -> dv/dx = -1, dv/dr = 1
            # v2 = x + r - w -> dv/dx = 1, dv/dr = 1
            # v3 = r - y -> dv/dy = -1, dv/dr = 1
            # v4 = y + r - h -> dv/dy = 1, dv/dr = 1
            
            v1 = rs - xs
            v2 = xs + rs - w
            v3 = rs - ys
            v4 = ys + rs - h
            
            dv1 = np.maximum(v1, 0)
            dv2 = np.maximum(v2, 0)
            dv3 = np.maximum(v3, 0)
            dv4 = np.maximum(v4, 0)
            
            # dL/dx_i = -2*dv1_i + 2*dv2_i
            grad[0::3] += -2*dv1 + 2*dv2
            # dL/dy_i = -2*dv3_i + 2*dv4_i
            grad[1::3] += -2*dv3 + 2*dv4
            # dL/dr_i = 2*dv1_i + 2*dv2_i + 2*dv3_i + 2*dv4_i
            grad[2::3] += 2*(dv1 + dv2 + dv3 + dv4)
            
            # Overlap Penalty Gradient
            # v_ij = r_i + r_j - dist_ij
            # dL/dv = 2*max(v,0)
            # dv/dr_i = 1
            # dv/dr_j = 1
            # dv/dx_i = (x_i - x_j)/dist = -nx
            # dv/dy_i = (y_i - y_j)/dist = -ny
            # dv/dx_j = (x_j - x_i)/dist = nx
            # dv/dy_j = (y_j - y_i)/dist = ny
            
            dx = xs[i_idx] - xs[j_idx]
            dy = ys[i_idx] - ys[j_idx]
            dists = np.sqrt(dx**2 + dy**2)
            min_dists = rs[i_idx] + rs[j_idx]
            viol = min_dists - dists
            dv = np.maximum(viol, 0)
            
            # Only compute for non-zero distances
            mask = dists > 1e-8
            if np.any(mask):
                nx = dx[mask] / dists[mask]
                ny = dy[mask] / dists[mask]
                
                # Accumulate gradients using np.add.at
                # For pair (i, j):
                # grad_x[i] += -2*dv*nx
                # grad_y[i] += -2*dv*ny
                # grad_r[i] += 2*dv
                # grad_x[j] += 2*dv*nx
                # grad_y[j] += 2*dv*ny
                # grad_r[j] += 2*dv
                
                i_idx_m = i_idx[mask]
                j_idx_m = j_idx[mask]
                dv_m = dv[mask]
                
                np.add.at(grad, i_idx_m*3, -2*dv_m*nx)
                np.add.at(grad, i_idx_m*3+1, -2*dv_m*ny)
                np.add.at(grad, i_idx_m*3+2, 2*dv_m)
                
                np.add.at(grad, j_idx_m*3, 2*dv_m*nx)
                np.add.at(grad, j_idx_m*3+1, 2*dv_m*ny)
                np.add.at(grad, j_idx_m*3+2, 2*dv_m)
            
            return 100.0 * grad # Scale by penalty weight

        bounds = []
        for i in range(N):
            bounds.append((1e-6, w - 1e-6)) # x
            bounds.append((1e-6, h - 1e-6)) # y
            bounds.append((1e-6, min(w, h)/2)) # r

        try:
            res = minimize(
                objective,
                x0,
                jac=gradient,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            
            new_circles = res.x.reshape(-1, 3)
            
            # Post-processing: Clip to bounds
            new_circles[:, 0] = np.clip(new_circles[:, 0], new_circles[:, 2], w - new_circles[:, 2])
            new_circles[:, 1] = np.clip(new_circles[:, 1], new_circles[:, 2], h - new_circles[:, 2])
            
            # Check validity
            if _validate_global(new_circles):
                score = _score(new_circles)
                if score > best_score:
                    best_score = score
                    best_circles = new_circles.copy()
                    
        except Exception:
            continue

    return best_circles

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_circles = None
    best_score = 0.0
    
    # Aspect ratios W/H. W+H=2.
    # ar = W/H => W = 2*ar/(1+ar), H = 2/(1+ar)
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        W = 2 * ar / (1 + ar)
        H = 2 / (1 + ar)
        
        try:
            result = _solve_aspect(W, H, deadline)
            
            if result is not None:
                if _validate_global(result):
                    score = _score(result)
                    if score > best_score:
                        best_score = score
                        best_circles = result.copy()
                        
        except Exception:
            continue

    # Fallback
    if best_circles is None:
        # Simple grid in 1x1
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-5
                cells.append([x, y, r])
        best_circles = np.array(cells[:N])
        
    if not _validate_global(best_circles):
        # Minimal valid
        r = 0.01
        best_circles = np.zeros((N, 3))
        for i in range(N):
            best_circles[i, 0] = r + i * (2 * r + 1e-6)
            best_circles[i, 1] = r
            best_circles[i, 2] = r
            
    return best_circles

# EVOLVE-BLOCK-END
