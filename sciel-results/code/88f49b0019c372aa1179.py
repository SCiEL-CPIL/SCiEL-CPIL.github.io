"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Initialization
    side = int(np.ceil(np.sqrt(N)))
    xs = []
    ys = []
    count = 0
    # Use a fixed seed for reproducibility of the "random" jitter if needed, 
    # but np.random is fine.
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            jitter = 0.05 / side
            cx += np.random.uniform(-jitter, jitter)
            cy += np.random.uniform(-jitter, jitter)
            xs.append(cx)
            ys.append(cy)
            count += 1
    
    x = np.array(xs)
    y = np.array(ys)
    r = np.full(N, 0.01)
    
    best_score = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    def relax(x, y, r, iters=200):
        # x, y, r are arrays
        # Returns True if valid configuration found
        for _ in range(iters):
            if time.time() > deadline:
                return False
            # Enforce boundaries
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            
            # Resolve overlaps
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_d = r[i] + r[j]
                    if d < min_d:
                        if d < 1e-9:
                            angle = np.random.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            d = 1.0
                        push = (min_d - d) / 2.0
                        nx = dx / d
                        ny = dy / d
                        x[i] -= push * nx
                        y[i] -= push * ny
                        x[j] += push * nx
                        y[j] += push * ny
                        moved = True
            if not moved:
                return True
        return False

    step = 0.005
    stagnation = 0
    
    while time.time() < deadline:
        r_new = r + step
        x_new = x.copy()
        y_new = y.copy()
        
        if relax(x_new, y_new, r_new, iters=100):
            r = r_new
            x = x_new
            y = y_new
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            stagnation = 0
            # Slightly increase step to speed up
            step *= 1.02 
        else:
            stagnation += 1
            if stagnation > 5:
                # Perturb positions to escape local minimum
                # Small random walk
                x += np.random.normal(0, 0.005, N)
                y += np.random.normal(0, 0.005, N)
                # Re-clamp to avoid extreme invalid states for next relax
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                # Reduce step size
                step *= 0.7
                if step < 1e-5:
                    break
                stagnation = 0
                
    return np.column_stack((best_x, best_y, best_r))
# EVOLVE-BLOCK-END
