import numpy as np

N = 21

def construct():
    best_value = -np.inf
    best_circles = None
    
    # Try different aspect ratios
    aspect_ratios = np.linspace(0.3, 3.0, 20)
    
    for w_init in aspect_ratios:
        h_init = 2.0 - w_init
        if h_init <= 0.1:
            continue
            
        # Initialize circles in a grid
        aspect = w_init / h_init
        c = max(1, int(np.sqrt(N * aspect)))
        r_grid = max(1, int(np.ceil(N / c)))
        
        cw = w_init / c
        ch = h_init / r_grid
        
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        idx = 0
        for i in range(c):
            for j in range(r_grid):
                if idx >= N:
                    break
                xs[idx] = (i + 0.5) * cw
                ys[idx] = (j + 0.5) * ch
                rs[idx] = 0.48 * min(cw, ch)
                idx += 1
        
        if idx < N:
            for k in range(idx, N):
                xs[k] = w_init / 2
                ys[k] = h_init / 2
                rs[k] = 0.1
        
        # Force-directed relaxation
        w = w_init
        h = h_init
        
        # Parameters
        repulsion_strength = 10.0
        wall_attraction = 5.0
        lr = 0.01
        decay = 0.999
        max_iters = 5000
        tol = 1e-10
        
        for it in range(max_iters):
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = rs[i] + rs[j]
                    
                    if dist < min_dist and dist > 1e-10:
                        # Overlapping: push apart
                        overlap = min_dist - dist
                        force = repulsion_strength * overlap
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
                        # Also push radii down slightly
                        fr[i] -= 0.1 * force
                        fr[j] -= 0.1 * force
                    elif dist < min_dist * 1.5 and dist > 1e-10:
                        # Near overlapping: mild repulsion
                        overlap = min_dist - dist
                        if overlap > 0:
                            force = repulsion_strength * 0.1 * overlap
                            fx[i] += force * dx / dist
                            fy[i] += force * dy / dist
                            fx[j] -= force * dx / dist
                            fy[j] -= force * dy / dist
            
            # Wall constraints: push circles inward if outside
            for i in range(N):
                # Left wall
                if xs[i] < rs[i]:
                    fx[i] += wall_attraction * (rs[i] - xs[i])
                # Right wall
                if xs[i] + rs[i] > w:
                    fx[i] -= wall_attraction * (xs[i] + rs[i] - w)
                # Bottom wall
                if ys[i] < rs[i]:
                    fy[i] += wall_attraction * (rs[i] - ys[i])
                # Top wall
                if ys[i] + rs[i] > h:
                    fy[i] -= wall_attraction * (xs[i] + rs[i] - h)
            
            # Radius growth: try to increase radii if not constrained
            for i in range(N):
                # Check if circle can grow
                can_grow = True
                for j in range(N):
                    if i == j:
                        continue
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < rs[i] + rs[j] + 0.01:
                        can_grow = False
                        break
                # Check walls
                if xs[i] - rs[i] < 0.01 or xs[i] + rs[i] > w - 0.01:
                    can_grow = False
                if ys[i] - rs[i] < 0.01 or ys[i] + rs[i] > h - 0.01:
                    can_grow = False
                
                if can_grow:
                    fr[i] += 0.5
                else:
                    fr[i] -= 0.1
            
            # Update positions and radii
            for i in range(N):
                xs[i] += lr * fx[i]
                ys[i] += lr * fy[i]
                rs[i] += lr * fr[i]
                
                # Clamp to valid range
                xs[i] = np.clip(xs[i], 0, w)
                ys[i] = np.clip(ys[i], 0, h)
                rs[i] = np.clip(rs[i], 0.01, min(w, h) / 2)
            
            # Decay learning rate
            lr *= decay
            
            # Check convergence
            if it % 100 == 0:
                # Check for violations
                valid = True
                for i in range(N):
                    if xs[i] < rs[i] - 1e-6 or xs[i] + rs[i] > w + 1e-6:
                        valid = False
                        break
                    if ys[i] < rs[i] - 1e-6 or ys[i] + rs[i] > h + 1e-6:
                        valid = False
                        break
                    for j in range(i + 1, N):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < rs[i] + rs[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    val = np.sum(rs)
                    if val > best_value:
                        best_value = val
                        best_circles = np.column_stack([xs.copy(), ys.copy(), rs.copy()])
                    
                    if val > 0 and abs(val - best_value) < tol:
                        break
        
        # Try to adjust aspect ratio slightly
        if best_circles is not None:
            # Try small perturbations of w
            for dw in [-0.1, -0.05, 0.05, 0.1]:
                w_new = w_init + dw
                h_new = 2.0 - w_new
                if h_new <= 0.1:
                    continue
                
                # Scale circles to fit new rectangle
                scale_x = w_new / w_init
                scale_y = h_new / h_init
                xs_test = xs.copy() * scale_x
                ys_test = ys.copy() * scale_y
                rs_test = rs.copy() * min(scale_x, scale_y)
                
                # Quick check if valid
                valid = True
                for i in range(N):
                    if xs_test[i] < rs_test[i] - 1e-6 or xs_test[i] + rs_test[i] > w_new + 1e-6:
                        valid = False
                        break
                    if ys_test[i] < rs_test[i] - 1e-6 or ys_test[i] + rs_test[i] > h_new + 1e-6:
                        valid = False
                        break
                    for j in range(i + 1, N):
                        dx = xs_test[i] - xs_test[j]
                        dy = ys_test[i] - ys_test[j]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < rs_test[i] + rs_test[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    val = np.sum(rs_test)
                    if val > best_value:
                        best_value = val
                        best_circles = np.column_stack([xs_test, ys_test, rs_test])
    
    if best_circles is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_circles
