import numpy as np

N = 21

def construct():
    best_sum_r = -np.inf
    best_config = None
    
    widths = np.linspace(0.6, 1.4, 9)
    np.random.seed(42)
    
    for w in widths:
        h = 2.0 - w
        
        side = 5
        xs = np.zeros(N)
        ys = np.zeros(N)
        
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                xs[idx] = (i + 0.5) * (w / side)
                ys[idx] = (j + 0.5) * (h / side)
                xs[idx] += np.random.uniform(-0.01, 0.01)
                ys[idx] += np.random.uniform(-0.01, 0.01)
                xs[idx] = np.clip(xs[idx], 0.01, w - 0.01)
                ys[idx] = np.clip(ys[idx], 0.01, h - 0.01)
                idx += 1
        
        r_current = 0.05
        rs = np.full(N, r_current)
        
        delta_r = 0.01
        max_outer_iters = 50
        max_inner_iters = 500
        tol = 1e-5
        
        for outer_iter in range(max_outer_iters):
            r_new = r_current + delta_r
            if r_new > min(w, h) / 2.0:
                break
            
            rs[:] = r_new
            
            converged = False
            for inner_iter in range(max_inner_iters):
                # Vectorized overlap check and force calculation
                # Distances
                dx = xs[:, None] - xs[None, :]
                dy = ys[:, None] - ys[None, :]
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Min distances
                min_dist = rs[:, None] + rs[None, :]
                
                # Overlaps
                overlaps = min_dist - dist
                # Only consider i < j to avoid double counting and self-overlap
                # Mask for upper triangle
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                
                # Get indices of overlaps
                # We need to apply forces.
                # Instead of iterating, we can sum up forces.
                # Force on i from j is proportional to overlap * unit_vector_ij
                
                # Initialize forces
                fx = np.zeros(N)
                fy = np.zeros(N)
                
                # We need to handle the case where dist is 0
                # dist_safe = np.where(dist > 1e-9, dist, 1e-9)
                
                # Calculate unit vectors
                # ux = dx / dist_safe
                # uy = dy / dist_safe
                
                # Force magnitude
                # mag = overlaps * 0.5
                # But we only want to apply if overlap > 0
                
                # This is tricky to do purely vectorized for force accumulation
                # because we need to add to i and subtract from j.
                # We can do:
                # force_mag = np.where(overlaps > 0, overlaps * 0.5, 0)
                # ux = dx / np.where(dist > 1e-9, dist, 1)
                # uy = dy / np.where(dist > 1e-9, dist, 1)
                
                # fx_i += force_mag * ux
                # fy_i += force_mag * uy
                
                # For j, it's -force_mag * ux
                # So fx = sum(force_mag * ux, axis=1) - sum(force_mag * ux, axis=0) ??
                # No, dx[i,j] = x[i] - x[j].
                # Force on i is in direction (x[i]-x[j])/dist.
                # So fx[i] += sum_j (force_mag[i,j] * ux[i,j])
                # fx[j] -= sum_i (force_mag[i,j] * ux[i,j])
                
                # Let's stick to a loop for clarity and correctness, but optimize it.
                # N=21 is small. A Python loop of 21*20/2 = 210 pairs is fast.
                # 500 iterations * 210 pairs = 105,000 operations.
                # 50 outer * 9 widths = 450 * 105,000 = 47M operations.
                # This is acceptable in Python if the inner loop is tight.
                
                max_overlap = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        dx_ij = xs[i] - xs[j]
                        dy_ij = ys[i] - ys[j]
                        dist_ij = np.sqrt(dx_ij*dx_ij + dy_ij*dy_ij)
                        min_dist_ij = rs[i] + rs[j]
                        
                        if dist_ij < min_dist_ij:
                            overlap = min_dist_ij - dist_ij
                            if overlap > max_overlap:
                                max_overlap = overlap
                            
                            if dist_ij > 1e-9:
                                ux = dx_ij / dist_ij
                                uy = dy_ij / dist_ij
                                move = overlap * 0.5
                                xs[i] += ux * move
                                xs[j] -= ux * move
                                ys[i] += uy * move
                                ys[j] -= uy * move
                            else:
                                angle = np.random.uniform(0, 2*np.pi)
                                move = overlap * 0.5
                                xs[i] += np.cos(angle) * move
                                xs[j] -= np.cos(angle) * move
                                ys[i] += np.sin(angle) * move
                                ys[j] -= np.sin(angle) * move
                                
                # Boundary
                for i in range(N):
                    r = rs[i]
                    if xs[i] < r: xs[i] = r
                    elif xs[i] > w - r: xs[i] = w - r
                    if ys[i] < r: ys[i] = r
                    elif ys[i] > h - r: ys[i] = h - r
                    
                if max_overlap < tol:
                    converged = True
                    break
            
            if converged:
                r_current = r_new
            else:
                rs[:] = r_current
                delta_r *= 0.5
                if delta_r < 1e-5:
                    break
        
        # Final check
        valid = True
        for i in range(N):
            if xs[i] < rs[i] - 1e-6 or xs[i] > w - rs[i] + 1e-6:
                valid = False
                break
            if ys[i] < rs[i] - 1e-6 or ys[i] > h - rs[i] + 1e-6:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < rs[i] + rs[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            current_sum = np.sum(rs)
            if current_sum > best_sum_r:
                best_sum_r = current_sum
                best_config = np.column_stack((xs, ys, rs))
                
    if best_config is None:
        side = 5
        cells = []
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
                idx += 1
        return np.array(cells)
    
    return best_config
