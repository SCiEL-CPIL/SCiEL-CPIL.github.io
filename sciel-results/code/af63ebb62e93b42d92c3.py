import numpy as np
import time
import random
from scipy.spatial import Delaunay

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # State
    w = 1.4
    h = 2.0 - w
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Initialization: 7x3 Grid
    a, b = 7, 3
    for i in range(N):
        col = i % a
        row = i // a
        x[i] = (col + 0.5) * w / a
        y[i] = (row + 0.5) * h / b
        r[i] = 0.09

    if not _full_valid(w, x, y, r):
        r[:] = 0.08
        if not _full_valid(w, x, y, r):
            r[:] = 0.07

    best_value = np.sum(r)
    best_state = (w, x.copy(), y.copy(), r.copy())
    
    centers = np.column_stack((x, y))
    neighbors = _get_neighbors(centers)
    
    T = 0.05
    T_min = 1e-6
    alpha = 0.999
    step_size = 0.02
    
    iterations = 0
    max_iterations = 500000
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        trial_w = w
        trial_x = x.copy()
        trial_y = y.copy()
        trial_r = r.copy()
        
        move_type = random.random()
        
        if move_type < 0.5:
            i = random.randint(0, N-1)
            trial_x[i] += random.gauss(0, step_size)
            trial_y[i] += random.gauss(0, step_size)
            
        elif move_type < 0.7:
            i = random.randint(0, N-1)
            delta_r = random.gauss(0, 0.005)
            trial_r[i] += delta_r
            if trial_r[i] < 0.01: trial_r[i] = 0.01
            if trial_r[i] > 0.5: trial_r[i] = 0.5
            
        else:
            trial_w += random.gauss(0, 0.01)
            if trial_w < 0.5: trial_w = 0.5
            if trial_w > 1.5: trial_w = 1.5

        # Fast validity check using neighbors
        if _fast_valid(trial_w, trial_x, trial_y, trial_r, neighbors):
            trial_value = np.sum(trial_r)
            delta_val = trial_value - best_value
            
            if delta_val > 0:
                w, x, y, r = trial_w, trial_x, trial_y, trial_r
                best_value = trial_value
                best_state = (w, x.copy(), y.copy(), r.copy())
            else:
                if random.random() < np.exp(delta_val / T):
                    w, x, y, r = trial_w, trial_x, trial_y, trial_r
                    
        iterations += 1
        
        if iterations % 50 == 0:
            centers = np.column_stack((x, y))
            neighbors = _get_neighbors(centers)
            
        T *= alpha

    # Force-Directed Polish
    # We try to expand the packing by applying forces.
    # This is a simplified version of "jiggle" and "expand".
    bw, bx, by, br = best_state
    # Copy state for polish
    pw = bw
    px = bx.copy()
    py = by.copy()
    pr = br.copy()
    ph = 2.0 - pw
    
    polish_steps = 200
    dt = 0.001
    for step in range(polish_steps):
        # Calculate forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Repulsion from walls
        for i in range(N):
            if px[i] - pr[i] < 0:
                fx[i] += (px[i] - pr[i]) # Push right
            if px[i] + pr[i] > pw:
                fx[i] -= (px[i] + pr[i] - pw) # Push left
            if py[i] - pr[i] < 0:
                fy[i] += (py[i] - pr[i]) # Push up
            if py[i] + pr[i] > ph:
                fy[i] -= (py[i] + pr[i] - ph) # Push down
        
        # Repulsion between circles
        for i in range(N):
            for j in range(i+1, N):
                dx = px[i] - px[j]
                dy = py[i] - py[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq)
                min_dist = pr[i] + pr[j]
                if dist < min_dist:
                    if dist < 1e-6:
                        angle = random.uniform(0, 2*np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    # Force proportional to overlap
                    force = (min_dist - dist) * 10.0
                    fx[i] += dx * force
                    fy[i] += dy * force
                    fx[j] -= dx * force
                    fy[j] -= dy * force
        
        # Update positions
        px += fx * dt
        py += fy * dt
        
        # Expand radii slightly
        # Check if expansion is valid
        # Simple expansion: increase all r by small amount, if invalid, reduce
        # This is tricky. Let's just keep radii fixed in this force step,
        # and do a separate radius expansion step after.
        
    # Radius Expansion Polish
    # Try to increase radii
    for _ in range(50):
        improved = False
        for i in range(N):
            old_r = pr[i]
            # Binary search for max r
            low = old_r
            high = min(1.0, old_r * 1.5)
            for _ in range(10):
                mid = (low + high) / 2
                pr[i] = mid
                if _fast_valid(pw, px, py, pr, _get_neighbors(np.column_stack((px, py)))):
                    low = mid
                    improved = True
                else:
                    high = mid
            pr[i] = low
        if not improved:
            break
            
    if _full_valid(pw, px, py, pr):
        final_val = np.sum(pr)
        if final_val > best_value:
            best_value = final_val
            best_state = (pw, px.copy(), py.copy(), pr.copy())

    bw, bx, by, br = best_state
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [bx[i], by[i], br[i]]
        
    return result

def _get_neighbors(centers):
    try:
        tri = Delaunay(centers)
        neighbors = [set() for _ in range(N)]
        for simplex in tri.simplices:
            i, j, k = simplex
            neighbors[i].add(j)
            neighbors[i].add(k)
            neighbors[j].add(i)
            neighbors[j].add(k)
            neighbors[k].add(i)
            neighbors[k].add(j)
        return [list(s) for s in neighbors]
    except Exception:
        return [list(range(N)) for _ in range(N)]

def _fast_valid(w, x, y, r, neighbors):
    h = 2.0 - w
    if w <= 0.1 or h <= 0.1:
        return False
    for i in range(N):
        if r[i] < 1e-9:
            return False
        if x[i] - r[i] < -1e-8 or x[i] + r[i] > w + 1e-8:
            return False
        if y[i] - r[i] < -1e-8 or y[i] + r[i] > h + 1e-8:
            return False
    for i in range(N):
        for j in neighbors[i]:
            if j > i:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - 1e-9)**2:
                    return False
    return True

def _full_valid(w, x, y, r):
    h = 2.0 - w
    if w <= 0.1 or h <= 0.1:
        return False
    for i in range(N):
        if r[i] < 1e-9:
            return False
        if x[i] - r[i] < -1e-8 or x[i] + r[i] > w + 1e-8:
            return False
        if y[i] - r[i] < -1e-8 or y[i] + r[i] > h + 1e-8:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < (min_dist - 1e-9)**2:
                return False
    return True
