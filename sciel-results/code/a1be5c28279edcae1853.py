"""
Circle packing in perimeter-4 rectangle: Kinetic Annealing + Force Relaxation + Greedy Expansion.

Strategy:
1. Initialize circles in a grid or random layout.
2. Use a force-directed relaxation method where circles repel each other (inverse distance)
   and are attracted to the center of the rectangle (or constrained by walls).
3. Apply random "jiggles" to escape local minima (simulated annealing style).
4. After relaxation, perform a greedy radius expansion step to maximize the sum of radii
   given the fixed positions.
5. Iterate with decreasing temperature to converge to a good packing.
"""
import numpy as np
import time

N = 21
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check validity of circles in a w x h rectangle."""
    if w <= 0.05 or h <= 0.05:
        return False
    if np.any(r <= 0):
        return False
    for i in range(N):
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True


def _max_radius_for_circle(i, x, y, r, w, h):
    """Calculate the maximum possible radius for circle i given current positions of others."""
    # Distance to walls
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    
    # Distance to other circles
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx*dx + dy*dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
            
    return max(0.0, r_max)


def _greedy_expand(x, y, r, w, h, max_rounds=100):
    """
    Iteratively expand each circle to its maximum possible radius.
    This is a local improvement step that assumes positions are fixed.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for i in range(N):
            r_new = _max_radius_for_circle(i, x, y, r, w, h)
            if r_new > r[i] + 1e-10:
                r[i] = r_new
                improved = True
    return x, y, r


def _force_relax(x, y, r, w, h, num_steps=1000, damping=0.9, step_size=0.05):
    """
    Force-directed relaxation.
    Forces:
    1. Repulsion between overlapping or close circles.
    2. Confinement to the rectangle (soft walls).
    3. Mild attraction to center to prevent drift (optional, usually handled by symmetry).
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    
    for _ in range(num_steps):
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx*dx + dy*dy
                d = np.sqrt(d_sq) if d_sq > 1e-12 else 1e-6
                r_sum = r[i] + r[j]
                
                # If overlapping or very close, repel strongly
                if d < r_sum + 0.1:
                    # Force magnitude increases as distance decreases
                    # Use a smooth potential: F ~ 1/d^2 or similar
                    # Here we use a simple linear repulsion within a cutoff
                    overlap = r_sum - d
                    if overlap > 0:
                        force_mag = overlap * 10.0 # Strong repulsion
                    else:
                        # Mild repulsion to keep them separated
                        force_mag = (0.1 - (r_sum - d)) * 2.0 if (r_sum - d) < 0.1 else 0.0
                        # Actually, let's just push them apart if they are closer than r_sum + margin
                        margin = 0.05
                        if d < r_sum + margin:
                            force_mag = (r_sum + margin - d) * 5.0
                        else:
                            force_mag = 0.0
                    
                    if d > 1e-6:
                        fx[i] += (dx / d) * force_mag
                        fy[i] += (dy / d) * force_mag
                        fx[j] -= (dx / d) * force_mag
                        fy[j] -= (dy / d) * force_mag
        
        # Wall confinement (soft penalty)
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 10.0
            elif x[i] > w - r[i]:
                fx[i] -= (x[i] - (w - r[i])) * 10.0
            
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 10.0
            elif y[i] > h - r[i]:
                fy[i] -= (y[i] - (h - r[i])) * 10.0
        
        # Update positions
        x += fx * step_size
        y += fy * step_size
        
        # Hard clamp to keep inside (safety)
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)
        
        # Damping isn't directly used in this simple velocity-less update, 
        # but step_size reduction acts similarly.
        
    return x, y


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    best_value = 0.0
    best_result = None
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.4, 0.6, 1.5, 0.5, 1.8]
    
    rng = np.random.default_rng(42)
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 30:
            break
            
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
            
        # Initialize positions
        # Use a grid initialization for stability
        cols = int(np.ceil(np.sqrt(N * w0 / h0)))
        rows = int(np.ceil(N / cols))
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        cw = w0 / cols
        ch = h0 / rows
        
        for i in range(N):
            row, col = divmod(i, cols)
            x[i] = (col + 0.5) * cw
            y[i] = (row + 0.5) * ch
            # Initial radius: half the cell size, slightly smaller to allow movement
            r[i] = min(cw, ch) / 2.0 * 0.8
            
        # Add some random jitter to break symmetry
        x += rng.uniform(-0.01, 0.01, N)
        y += rng.uniform(-0.01, 0.01, N)
        
        # Annealing loop
        temperature = 0.1
        num_annealing_steps = 200
        
        for step in range(num_annealing_steps):
            if time.time() > deadline - 10:
                break
                
            # 1. Force Relaxation
            x_rel, y_rel = _force_relax(x, y, r, w0, h0, num_steps=50, step_size=temperature)
            
            # 2. Jiggle (Random perturbation)
            # The magnitude of jiggle is controlled by temperature
            x_jig = x_rel + rng.normal(0, temperature, N)
            y_jig = y_rel + rng.normal(0, temperature, N)
            
            # 3. Greedy Expansion
            x_exp, y_exp, r_exp = _greedy_expand(x_jig, y_jig, r, w0, h0, max_rounds=20)
            
            # 4. Check Validity and Score
            if _is_valid(x_exp, y_exp, r_exp, w0, h0):
                val = np.sum(r_exp)
                if val > best_value:
                    best_value = val
                    best_result = (x_exp.copy(), y_exp.copy(), r_exp.copy(), w0, h0)
                    
            # Update positions for next iteration
            x, y, r = x_exp, y_exp, r_exp
            
            # Cool down
            temperature *= 0.95
            
            # If temperature is very low, we can stop early if no improvement in last few steps
            # (Simplified: just keep going until time runs out or temp is tiny)
            if temperature < 1e-4:
                break
                
        # Final polish for this aspect ratio
        if best_result is not None:
            # Use the best found so far or the current one? 
            # Let's use the current one if it's valid, otherwise keep best.
            # Actually, let's just run a final high-precision expansion on the current state
            if _is_valid(x, y, r, w0, h0):
                x_final, y_final, r_final = _greedy_expand(x, y, r, w0, h0, max_rounds=100)
                if _is_valid(x_final, y_final, r_final, w0, h0):
                    val_final = np.sum(r_final)
                    if val_final > best_value:
                        best_value = val_final
                        best_result = (x_final.copy(), y_final.copy(), r_final.copy(), w0, h0)

    if best_result is None:
        # Fallback: simple grid in unit square (w=1, h=1)
        w0 = 1.0
        h0 = 1.0
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side * 0.9
        # Expand
        x, y, r = _greedy_expand(x, y, r, w0, h0, max_rounds=100)
        best_result = (x, y, r, w0, h0)
        best_value = np.sum(r)

    x, y, r, w, h = best_result
    
    # Final safety check and shrink if necessary
    # Ensure strict validity
    for _ in range(50):
        if _is_valid(x, y, r, w, h, tol=1e-9):
            break
        r *= 0.999
        if np.any(r < 1e-6):
            break
            
    if not _is_valid(x, y, r, w, h):
        # Last resort fallback
        w0 = 1.0
        h0 = 1.0
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side * 0.8
        x, y, r = _greedy_expand(x, y, r, w0, h0, max_rounds=100)
        
    return np.column_stack((x, y, r))
