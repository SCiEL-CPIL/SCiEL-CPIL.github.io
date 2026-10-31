"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = np.random.default_rng(42)
    
    best_value = -np.inf
    best_result = None
    
    # Try multiple aspect ratios and initial configurations
    # Perimeter = 4 => w + h = 2, so h = 2 - w
    # Optimal aspect ratio for circle packing is often near 1:1 or specific ratios
    
    for trial in range(8):
        if time.time() > deadline:
            break
        
        # Vary aspect ratio
        if trial == 0:
            w = 1.0  # square
        elif trial == 1:
            w = 1.2
        elif trial == 2:
            w = 0.85
        elif trial == 3:
            w = 1.1
        elif trial == 4:
            w = 0.9
        elif trial == 5:
            w = 1.05
        elif trial == 6:
            w = 0.95
        else:
            w = 1.0
        
        h = 2.0 - w
        
        # Initialize circles in a grid pattern with some randomness
        side = 5  # 5x5 grid = 25 positions, use 21
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        for i in range(N):
            gx = i % side
            gy = i // side
            xs[i] = (gx + 0.5) * w / side + rng.uniform(-0.1, 0.1)
            ys[i] = (gy + 0.5) * h / side + rng.uniform(-0.1, 0.1)
            rs[i] = 0.5 * min(w, h) / side * 0.9
        
        # Clamp to bounds
        xs = np.clip(xs, 0, w)
        ys = np.clip(ys, 0, h)
        rs = np.clip(rs, 0.01, min(w, h) * 0.4)
        
        # Simulated annealing
        T = 0.05
        T_min = 0.0001
        alpha = 0.999
        n_steps = 500000
        
        cur_x = xs.copy()
        cur_y = ys.copy()
        cur_r = rs.copy()
        cur_val = -np.inf
        cur_valid = False
        
        for step in range(n_steps):
            if time.time() > deadline:
                break
            
            # Random perturbation
            idx = rng.integers(0, N)
            dx = rng.normal(0, 0.05)
            dy = rng.normal(0, 0.05)
            dr = rng.normal(0, 0.02)
            
            new_x = cur_x.copy()
            new_y = cur_y.copy()
            new_r = cur_r.copy()
            
            new_x[idx] = np.clip(cur_x[idx] + dx, 0, w)
            new_y[idx] = np.clip(cur_y[idx] + dy, 0, h)
            new_r[idx] = np.clip(cur_r[idx] + dr, 0.01, min(w, h) * 0.4)
            
            # Check validity: no overlaps, within bounds
            valid = True
            # Check bounds
            if np.any(new_x < 0) or np.any(new_x > w) or np.any(new_y < 0) or np.any(new_y > h):
                valid = False
            # Check overlaps (vectorized)
            if valid:
                dx_all = new_x[:, None] - new_x[None, :]
                dy_all = new_y[:, None] - new_y[None, :]
                dist = np.sqrt(dx_all**2 + dy_all**2)
                min_dist = new_r[:, None] + new_r[None, :]
                np.fill_diagonal(dist, np.inf)
                if np.any(dist < min_dist - 1e-9):
                    valid = False
            
            if valid:
                new_val = np.sum(new_r)
                if new_val > cur_val or not cur_valid:
                    cur_x = new_x
                    cur_y = new_y
                    cur_r = new_r
                    cur_val = new_val
                    cur_valid = True
                    if cur_val > best_value:
                        best_value = cur_val
                        best_result = np.column_stack([cur_x, cur_y, cur_r])
            else:
                # Reject if invalid
                pass
            
            T *= alpha
            if T < T_min:
                T = T_min
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback: simple grid
        side = 5
        w = 1.0
        h = 1.0
        cells = []
        for i in range(N):
            gx = i % side
            gy = i // side
            cells.append(((gx + 0.5) * w / side, (gy + 0.5) * h / side, 0.5 * min(w, h) / side * 0.85))
        best_result = np.array(cells)
    
    return best_result
# EVOLVE-BLOCK-END
