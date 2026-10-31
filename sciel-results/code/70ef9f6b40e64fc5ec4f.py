import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin of 10 seconds
    
    best_score = -1.0
    best_result = None
    
    # Try a range of aspect ratios (width/height)
    # Perimeter P = 2(w+h) <= 4 => w+h <= 2
    # Let w = r * h, then h(r+1) <= 2 => h <= 2/(r+1)
    # To maximize radii, we should probably fill the perimeter, so w+h = 2.
    
    # We can parameterize by width w. h = 2 - w.
    # w must be > 0. h must be > 0.
    # Since we want to pack 21 circles, the aspect ratio shouldn't be too extreme.
    # Let's try a few different aspect ratios to see which packs best.
    
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.4, 1.6, 0.6, 2.0, 0.5]
    
    for ratio in aspect_ratios:
        if time.time() > deadline:
            break
            
        # Determine width and height for this ratio
        # w/h = ratio => w = ratio * h
        # 2(w+h) = 4 => w+h = 2
        # h(1+ratio) = 2 => h = 2/(1+ratio), w = 2*ratio/(1+ratio)
        
        h = 2.0 / (1.0 + ratio)
        w = ratio * h
        
        # Ensure positive dimensions
        if w <= 0 or h <= 0:
            continue
            
        result = _relax_pack(w, h, N, start_time, deadline)
        
        if result is not None:
            # Calculate score: sum of radii
            radii = result[:, 2]
            # Check validity: all circles inside and disjoint
            # The relaxation should enforce this, but double check for safety
            if _is_valid(result, w, h):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_result = result
                    
    # If no valid result found (unlikely), return a simple fallback
    if best_result is None:
        # Fallback: simple grid in a square 1x1
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result

def _relax_pack(w, h, n, start_time, deadline):
    """
    Attempt to pack n circles in a w x h rectangle using force-directed relaxation.
    Returns an array of shape (n, 3) [x, y, r] or None if it fails/times out.
    """
    
    # Initial random positions
    # To improve determinism and reproducibility, we could use a fixed seed,
    # but random restarts might help escape local minima if we had more time.
    # Here we do a single robust run per aspect ratio.
    
    np.random.seed(42 + int(w*1000)) # Simple seed based on w
    
    x = np.random.uniform(0.2, w - 0.2, n)
    y = np.random.uniform(0.2, h - 0.2, n)
    
    # Initial radii: estimate based on area
    area = w * h
    r_init = np.sqrt(area / (n * np.pi * 1.5)) # 1.5 is a packing efficiency guess
    r = np.full(n, r_init)
    
    # Force-directed relaxation parameters
    max_iter = 2000
    step_size = 0.01
    cooling = 0.995
    
    # We want to maximize sum of radii.
    # The forces should:
    # 1. Push circles apart if they overlap.
    # 2. Push circles towards walls if they are inside (to allow radii to grow).
    # 3. Push circles inward if they are outside.
    
    # Actually, a better approach for "maximize sum of radii" is to treat radii as variables
    # that grow until constraints are tight.
    
    # Let's iterate:
    # 1. Given current positions and radii, compute forces to resolve overlaps and keep in bounds.
    # 2. Move centers.
    # 3. Grow radii slightly.
    # 4. Repeat.
    
    # Better: Optimize positions for fixed radii that are growing?
    # Or optimize positions and radii together?
    
    # Let's try this heuristic:
    # Fix radii to a target value? No, radii should be different or at least optimized.
    # For this problem, "maximize sum of radii" with "disjoint interiors" usually leads to
    # circles of similar size if the container is not too small, or a few large ones.
    # Given 21 circles in a ~1x1 or 2x1 box, they will likely be somewhat uniform.
    
    # Let's assume uniform radii for simplicity in this first structural change?
    # No, the task asks for an array of circles, implying radii can vary.
    # However, for a first robust attempt, assuming uniform radii is a strong baseline
    # for circle packing problems. If we find the max uniform radius, sum is N * r_max.
    # But non-uniform might be better? In dense packings, uniform is often near optimal.
    # Let's try to optimize uniform radii first, as it simplifies the force calculation.
    
    # Re-initialize for uniform radius optimization
    x = np.random.uniform(0, w, n)
    y = np.random.uniform(0, h, n)
    r = np.full(n, r_init)
    
    # We will try to increase r until no more improvement in positions.
    # Or, we run a relaxation that allows r to grow.
    
    # Let's use a simple "grow and push" algorithm.
    # 1. Start with small r.
    # 2. Relax positions to fit in box without overlap.
    # 3. Increase r slightly.
    # 4. Relax positions again.
    # 5. Stop when we can't fit anymore or time runs out.
    
    r_current = r_init * 0.5
    x = np.random.uniform(r_current, w - r_current, n)
    y = np.random.uniform(r_current, h - r_current, n)
    
    growth_rate = 0.001
    min_growth = 1e-5
    
    while time.time() < deadline:
        # Relax positions for current radius
        x, y = _relax_positions(x, y, r_current, w, h, max_iter=500, deadline=deadline)
        
        # Check if we can grow
        # If after relaxation, circles are still touching or we hit the wall, we can't grow much.
        # A simple heuristic: try to grow. If the next relaxation fails (circles overlap significantly), shrink back.
        # Or just grow by a small amount and rely on the relaxation to fix it.
        # If the relaxation results in high energy, we've gone too far.
        
        # Let's try a step of growth
        r_new = r_current + growth_rate
        
        # Check bounds
        # If r_new is too large to fit even one circle, stop
        if 2 * r_new > min(w, h):
            break
            
        # Try to relax with r_new
        x_new, y_new = _relax_positions(x, y, r_new, w, h, max_iter=500, deadline=deadline)
        
        # Simple energy check: sum of squared overlaps
        # If overlaps are too large, revert
        energy = _calc_overlap_energy(x_new, y_new, r_new, w, h)
        
        # This is a bit heuristic. Let's just accept if we are within bounds.
        # The _relax_positions function should keep them in bounds.
        
        # Update
        x, y = x_new, y_new
        r_current = r_new
        
        # Adaptive growth
        # If we made progress, keep growing rate or increase slightly?
        # If we struggle, decrease.
        # For now, constant growth is slow but safe.
        
    # Final relaxation to ensure validity
    x, y = _relax_positions(x, y, r_current, w, h, max_iter=1000, deadline=deadline)
    
    # Construct result
    radii = np.full(n, r_current)
    centers = np.column_stack((x, y, radii))
    
    return centers

def _relax_positions(x, y, r, w, h, max_iter=1000, deadline=None):
    """
    Relax positions of n circles of radius r in a w x h box.
    Uses a simple force-directed method.
    """
    n = len(x)
    step_size = 0.05
    damping = 0.9
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Pairwise repulsion
        for i_idx in range(n):
            for j_idx in range(i_idx + 1, n):
                dx = x[i_idx] - x[j_idx]
                dy = y[i_idx] - y[j_idx]
                dist = np.sqrt(dx*dx + dy*dy)
                
                min_dist = 2 * r
                if dist < min_dist:
                    # Repulsive force proportional to overlap
                    force = (min_dist - dist) * 1.0
                    if dist == 0:
                        # Randomize to avoid singularity
                        fx[i_idx] += np.random.uniform(-1, 1) * force
                        fy[i_idx] += np.random.uniform(-1, 1) * force
                        fx[j_idx] -= np.random.uniform(-1, 1) * force
                        fy[j_idx] -= np.random.uniform(-1, 1) * force
                    else:
                        norm_x = dx / dist
                        norm_y = dy / dist
                        fx[i_idx] += norm_x * force
                        fy[i_idx] += norm_y * force
                        fx[j_idx] -= norm_x * force
                        fy[j_idx] -= norm_y * force
                        
        # Wall constraints
        # Push inward if touching or outside
        for i_idx in range(n):
            if x[i_idx] < r:
                fx[i_idx] += (r - x[i_idx]) * 2.0
            if x[i_idx] > w - r:
                fx[i_idx] -= (x[i_idx] - (w - r)) * 2.0
            if y[i_idx] < r:
                fy[i_idx] += (r - y[i_idx]) * 2.0
            if y[i_idx] > h - r:
                fy[i_idx] -= (y[i_idx] - (h - r)) * 2.0
                
        # Update positions
        x += fx * step_size
        y += fy * step_size
        
        # Hard clip to box (optional, forces should handle it, but clipping prevents escape)
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)
        
        # Stop if forces are negligible
        if np.max(np.abs(fx)) < 1e-6 and np.max(np.abs(fy)) < 1e-6:
            break
            
    return x, y

def _calc_overlap_energy(x, y, r, w, h):
    n = len(x)
    energy = 0
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            overlap = 2*r - dist
            if overlap > 0:
                energy += overlap**2
    # Wall penalties
    for i in range(n):
        if x[i] < r: energy += (r-x[i])**2
        if x[i] > w-r: energy += (x[i]-(w-r))**2
        if y[i] < r: energy += (r-y[i])**2
        if y[i] > h-r: energy += (y[i]-(h-r))**2
    return energy

def _is_valid(result, w, h):
    n = len(result)
    x = result[:, 0]
    y = result[:, 1]
    r = result[:, 2]
    
    # Check bounds
    if np.any(x < -1e-9) or np.any(y < -1e-9):
        return False
    if np.any(x > w + 1e-9) or np.any(y > h + 1e-9):
        return False
        
    # Check disjoint interiors
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist - 1e-9:
                return False
    return True
