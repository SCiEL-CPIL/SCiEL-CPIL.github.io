"""Circle packing in perimeter-4 rectangle: Deterministic Staggered Construction."""
import numpy as np
import time
import math

N = 21

def _check_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if the configuration is valid."""
    if W <= 0 or H <= 0:
        return False
    if np.any(cx < r - tol) or np.any(cx > W - r + tol):
        return False
    if np.any(cy < r - tol) or np.any(cy > H - r + tol):
        return False
    # Vectorized overlap check
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    d2 = dx**2 + dy**2
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    # Check upper triangle to avoid double counting and self-interaction
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d2[mask] < (rs[mask] - tol)**2):
        return False
    return True

def _optimize_radius(cx, cy, r, W, H, max_iter=1000, time_limit=None, start_time=None):
    """
    Optimize the sum of radii by uniformly scaling and nudging.
    This is a local search around a good initial configuration.
    """
    if start_time is None:
        start_time = time.time()
    
    best_val = np.sum(r)
    best_state = (cx.copy(), cy.copy(), r.copy(), W, H)
    
    # We will try to increase all radii by a factor.
    # If it fails, we try to nudge centers to make space.
    
    lr_nudge = 0.01
    growth_step = 0.001
    
    for it in range(max_iter):
        if time_limit and time.time() - start_time > time_limit:
            break
            
        # Try to grow radii
        r_new = r * (1.0 + growth_step)
        
        if _check_valid(cx, cy, r_new, W, H):
            r = r_new
            val = np.sum(r)
            if val > best_val:
                best_val = val
                best_state = (cx.copy(), cy.copy(), r.copy(), W, H)
        else:
            # If growth fails, try to nudge circles to create space
            # Identify the most "constrained" circles (those touching others or walls)
            # Simple heuristic: nudge all circles slightly randomly
            # But to be deterministic and efficient, we can try specific directions
            
            # Calculate forces based on overlaps
            forces_x = np.zeros(N)
            forces_y = np.zeros(N)
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    d2 = dx*dx + dy*dy
                    d = np.sqrt(d2) if d2 > 1e-12 else 1e-12
                    rs = r[i] + r[j]
                    
                    if d < rs:
                        overlap = rs - d
                        f = overlap * 0.5
                        if d > 1e-12:
                            fx = f * dx / d
                            fy = f * dy / d
                            forces_x[i] += fx
                            forces_y[i] += fy
                            forces_x[j] -= fx
                            forces_y[j] -= fy
                        else:
                            angle = np.pi / 4
                            forces_x[i] += f * np.cos(angle)
                            forces_y[i] += f * np.sin(angle)
                            forces_x[j] -= f * np.cos(angle)
                            forces_y[j] -= f * np.sin(angle)
                
                # Wall forces
                if cx[i] - r[i] < 0:
                    forces_x[i] += 0.5
                if W - cx[i] - r[i] < 0:
                    forces_x[i] -= 0.5
                if cy[i] - r[i] < 0:
                    forces_y[i] += 0.5
                if H - cy[i] - r[i] < 0:
                    forces_y[i] -= 0.5
            
            # Apply nudge
            cx += forces_x * lr_nudge
            cy += forces_y * lr_nudge
            
            # Project to bounds
            cx = np.clip(cx, r, W - r)
            cy = np.clip(cy, r, H - r)
            
            # Decay nudge strength
            lr_nudge *= 0.95

    return best_state

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_val = 0.0
    best_result = None
    
    # Define a set of "staggered" grid configurations
    # These are known to be efficient packings.
    # We will try different aspect ratios and stagger patterns.
    
    # Pattern 1: Standard staggered (hexagonal-like)
    # Rows of circles, offset by half a diameter.
    
    # We will iterate over possible numbers of rows and columns
    # and aspect ratios.
    
    for rows in range(3, 8):
        for cols in range(3, 8):
            if rows * cols < N:
                continue
                
            # Try different aspect ratios W/H
            # Perimeter W + H = 2 => H = 2 - W
            for W in np.linspace(0.6, 1.4, 17):
                H = 2.0 - W
                
                # Construct the grid
                cx = np.zeros(N)
                cy = np.zeros(N)
                r = np.zeros(N)
                
                # Cell dimensions
                # For staggered, horizontal spacing is 2*r, vertical is sqrt(3)*r
                # But we fit into W x H.
                
                # Let's estimate r from area or simple division
                # Simple division:
                cell_w = W / cols
                cell_h = H / rows
                
                # Initial radius estimate
                r_est = min(cell_w, cell_h) * 0.5 * 0.9 # 90% of half-cell to allow stagger
                
                # Place circles
                idx = 0
                for i in range(rows):
                    for j in range(cols):
                        if idx >= N:
                            break
                        # Stagger offset
                        offset = 0.5 * cell_w if i % 2 == 1 else 0.0
                        
                        x = (j + 0.5) * cell_w + offset
                        y = (i + 0.5) * cell_h
                        
                        # Clip to bounds
                        x = np.clip(x, r_est, W - r_est)
                        y = np.clip(y, r_est, H - r_est)
                        
                        cx[idx] = x
                        cy[idx] = y
                        r[idx] = r_est
                        idx += 1
                        
                if idx < N:
                    # Fill remaining spots if any (should not happen if rows*cols >= N)
                    # Just place them in the center or something
                    for k in range(idx, N):
                        cx[k] = W / 2
                        cy[k] = H / 2
                        r[k] = r_est * 0.5
                        
                # Check if initial config is valid
                if _check_valid(cx, cy, r, W, H):
                    # Optimize
                    res_cx, res_cy, res_r, res_W, res_H = _optimize_radius(
                        cx, cy, r, W, H, 
                        max_iter=500, 
                        time_limit=2.0, 
                        start_time=time.time()
                    )
                    
                    if _check_valid(res_cx, res_cy, res_r, res_W, res_H):
                        val = np.sum(res_r)
                        if val > best_val:
                            best_val = val
                            best_result = (res_cx, res_cy, res_r, res_W, res_H)
                            
                # Early exit if time is running out
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    if best_result is None:
        # Fallback: Simple grid in square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    cx, cy, r, W, H = best_result
    
    # Final Polish: Try to adjust aspect ratio slightly and re-optimize
    # This can squeeze out a bit more value.
    for _ in range(20):
        if time.time() > deadline - 1:
            break
            
        # Try small aspect ratio changes
        W_trial = W + np.random.uniform(-0.05, 0.05)
        W_trial = np.clip(W_trial, 0.5, 1.5)
        H_trial = 2.0 - W_trial
        
        # Check if current circles fit in new box
        if np.all(cx >= r - 1e-9) and np.all(cx <= W_trial - r + 1e-9) and \
           np.all(cy >= r - 1e-9) and np.all(cy <= H_trial - r + 1e-9):
            # Re-optimize in new box
            res_cx, res_cy, res_r, res_W, res_H = _optimize_radius(
                cx, cy, r, W_trial, H_trial,
                max_iter=200,
                time_limit=1.0,
                start_time=time.time()
            )
            if _check_valid(res_cx, res_cy, res_r, res_W, res_H):
                val = np.sum(res_r)
                if val > best_val:
                    best_val = val
                    best_result = (res_cx, res_cy, res_r, res_W, res_H)
                    cx, cy, r, W, H = best_result

    # Final safety check
    if not _check_valid(cx, cy, r, W, H):
        # Shrink until valid
        scale = 1.0
        while not _check_valid(cx, cy, r * scale, W, H) and scale > 0.1:
            scale *= 0.95
        r = r * scale
        if not _check_valid(cx, cy, r, W, H):
             # Extreme fallback
             side = int(np.ceil(np.sqrt(N)))
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                      for i in range(side) for j in range(side)]
             return np.array(cells[:N])

    return np.column_stack((cx, cy, r))
