import numpy as np
import time
import random

N = 26

# EVOLVE-BLOCK-START

def _check_validity(pos, radii, tol=1e-7):
    """
    Check if the configuration is valid:
    1. All circles are within [0, 1]^2.
    2. No two circles overlap.
    """
    # Boundary check
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    
    # Overlap check
    # dist^2 >= (r_i + r_j)^2
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx*dx + dy*dy
            sum_r = radii[i] + radii[j]
            if dist_sq < (sum_r - tol)**2:
                return False
    return True

def _max_radius(pos, radii, i):
    """
    Calculate the maximum radius circle i can have at its current position,
    given the positions and radii of other circles.
    """
    x, y = pos[i]
    max_r = min(x, 1.0-x, y, 1.0-y)
    
    for j in range(N):
        if i == j:
            continue
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        # r_i <= dist - r_j
        limit = dist - radii[j]
        if limit < max_r:
            max_r = limit
            
    return max(0.0, max_r)

def _force_directed_anneal(pos, radii, start_time, deadline, seed=42):
    """
    Optimizes the packing using force-directed layout with simulated annealing.
    """
    rng = np.random.RandomState(seed)
    
    # Parameters
    # We use a cooling schedule.
    # Start temp: 0.05, End temp: 0.0001
    T_start = 0.05
    T_end = 0.0001
    
    # Number of iterations depends on time budget
    # Estimate iterations per second. ~1000-5000 simple steps/sec in pure python/numpy mix.
    # Let's aim for 100,000 steps if time allows, but check time frequently.
    
    # Initial setup
    pos = pos.copy()
    radii = radii.copy()
    
    # Ensure initial validity by shrinking if necessary
    for _ in range(100):
        valid = True
        for i in range(N):
            max_r = _max_radius(pos, radii, i)
            if radii[i] > max_r:
                radii[i] = max_r * 0.99
                valid = False
        if valid:
            break
        # If still invalid after shrinking, nudge positions
        if not valid and _ > 10:
             pos += rng.uniform(-0.01, 0.01, pos.shape)
             pos = np.clip(pos, 0.01, 0.99)
    
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    # Annealing loop
    # We divide the time into chunks to allow for dynamic step count adjustment
    total_time_budget = deadline - start_time
    if total_time_budget <= 1:
        return best_pos, best_radii
        
    # Target steps: try to run as many as possible, but check time every 1000 steps
    # A single step involves N^2 distance calcs. N=26 -> 676 ops. Very fast.
    # We can afford ~50k-100k steps in 10-20 seconds.
    
    steps = 0
    max_steps = 200000 # Cap to prevent infinite loop if time check fails
    
    while time.time() < deadline - 2.0:
        # Determine current temperature
        progress = min(1.0, steps / max_steps)
        T = T_start * (T_end / T_start) ** progress
        
        # Calculate forces
        forces = np.zeros_like(pos)
        
        # 1. Repulsion between circles
        # For each pair, if overlapping, push apart.
        # We use a stiff spring: F = k * overlap
        # Also add a small repulsion for all pairs to prevent clustering? 
        # No, standard force directed graph for packing usually only pushes when overlapping or close.
        
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                # If they are overlapping
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Force magnitude proportional to overlap
                    # Normalize direction
                    if dist > 1e-8:
                        fx = (dx / dist) * (min_dist - dist)
                        fy = (dy / dist) * (min_dist - dist)
                    else:
                        # Random direction if exactly same point
                        fx = rng.uniform(-1, 1)
                        fy = rng.uniform(-1, 1)
                    
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                else:
                    # Optional: Weak attraction to pull circles together to pack denser?
                    # Usually, gravity to center is enough.
                    pass
        
        # 2. Boundary forces
        # Push back if outside
        # Left
        viol = np.maximum(0, radii - pos[:, 0])
        forces[:, 0] += viol
        # Right
        viol = np.maximum(0, pos[:, 0] - (1.0 - radii))
        forces[:, 0] -= viol
        # Bottom
        viol = np.maximum(0, radii - pos[:, 1])
        forces[:, 1] += viol
        # Top
        viol = np.maximum(0, pos[:, 1] - (1.0 - radii))
        forces[:, 1] -= viol
        
        # 3. Gravity towards center
        # This helps pack circles towards the middle, leaving corners for larger circles potentially?
        # Actually, for max sum of radii, pushing to center is good because boundaries constrain size.
        # F = k * (center - pos)
        k_gravity = 0.05
        forces += k_gravity * (0.5 - pos)
        
        # 4. Add noise (Simulated Annealing)
        noise = rng.normal(0, T, pos.shape)
        
        # Update positions
        # Step size
        # We want to move in the direction of force + noise
        # Normalize force to prevent huge jumps?
        # Let's use a fixed step size scaled by T or just a small constant.
        # A common trick: pos += force * step_size
        # But force magnitude varies. 
        # Let's normalize the force vector per circle to unit length if we want consistent step size,
        # or just use the raw force with a small coefficient.
        
        # Using raw force can lead to instability if overlaps are large.
        # Let's clamp the force magnitude.
        force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        max_force = 0.1
        forces = forces * np.minimum(1.0, max_force / (force_mags + 1e-8))
        
        pos += forces + noise
        
        # 5. Grow Radii
        # After moving, try to grow each circle as much as possible.
        for i in range(N):
            max_r = _max_radius(pos, radii, i)
            # Grow towards max_r, but maybe not fully to allow for future moves?
            # Growing fully is usually better for "sum of radii" objective.
            # If we grow fully, we might get stuck if we can't move anymore.
            # But the forces will push us if we overlap with the grown radius.
            # Wait, _max_radius uses CURRENT radii of neighbors.
            # If I grow i, it might overlap with j.
            # The force calculation in the next iteration will handle the overlap by pushing them apart.
            # So growing fully is a good strategy.
            radii[i] = max_r
            
        # Track best
        current_score = np.sum(radii)
        if current_score > best_score:
            best_score = current_score
            best_pos = pos.copy()
            best_radii = radii.copy()
            
        steps += 1
        
        # Early break if we are in a stable state and T is low?
        # Hard to define "stable". Just rely on time.
        
    return best_pos, best_radii

def _initial_lattice():
    """
    Generate a good initial guess using a hexagonal lattice.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Hexagonal packing is densest.
    # Try to fit N circles.
    # For N=26, a 5x5 grid with some removed/adjusted might work, or a hex grid.
    
    # Let's try a hex grid centered at (0.5, 0.5)
    # Radius of hex grid...
    
    # Simpler: Start with a grid and let annealing fix it.
    # 5x5 grid = 25 circles. We have 26.
    # Add one in the center? No, that's crowded.
    # Maybe 5x5 grid with one circle split or moved?
    
    # Let's just place them in a grid and let the optimizer do the work.
    # 5x5 grid:
    # Spacing 0.2
    # Centers: 0.1, 0.3, 0.5, 0.7, 0.9
    # This leaves 0.1 margin.
    
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N:
                pos[k] = [0.1 + i*0.2, 0.1 + j*0.2]
                radii[k] = 0.09 # Slightly less than half spacing 0.1 to allow initial overlap resolution
                k += 1
            if k >= N:
                break
        if k >= N:
            break
            
    # Add the 26th circle?
    # The 5x5 grid fills 25 spots.
    # We can place the 26th one by pushing the others?
    # Or use a 4x4 grid (16) + 10 in a ring?
    
    # Better initial guess:
    # Use a hexagonal lattice.
    # Row 0: 3 circles
    # Row 1: 4 circles
    # Row 2: 5 circles
    # Row 3: 4 circles
    # Row 4: 3 circles
    # Row 5: 3 circles?
    # Sum: 3+4+5+4+3+3 = 22.
    # Let's try to fit 26 in a hex pattern.
    
    # Actually, the grid initialization + annealing is robust enough.
    # Let's add a random jitter to break symmetry.
    rng = np.random.RandomState(123)
    pos += rng.uniform(-0.01, 0.01, pos.shape)
    pos = np.clip(pos, 0.05, 0.95)
    
    # Ensure initial radii are valid (shrink if overlapping)
    for _ in range(50):
        valid = True
        for i in range(N):
            max_r = _max_radius(pos, radii, i)
            if radii[i] > max_r:
                radii[i] = max_r
                valid = False
        if valid:
            break
        # If stuck, jitter
        pos += rng.uniform(-0.01, 0.01, pos.shape)
        pos = np.clip(pos, 0.05, 0.95)
        
    return pos, radii

def _initial_random():
    """
    Random initialization.
    """
    rng = np.random.RandomState(42)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = np.full(N, 0.05)
    
    # Shrink to validity
    for _ in range(100):
        valid = True
        for i in range(N):
            max_r = _max_radius(pos, radii, i)
            if radii[i] > max_r:
                radii[i] = max_r
                valid = False
        if valid:
            break
        pos += rng.uniform(-0.01, 0.01, pos.shape)
        pos = np.clip(pos, 0.05, 0.95)
        
    return pos, radii

def construct():
    start_time = time.time()
    # Safety margin: 140s out of 150s
    deadline = start_time + 140.0
    
    best_pos = None
    best_radii = None
    best_score = -1.0
    
    # Strategy: Run multiple restarts with different initializations and time allocations.
    # 1. Hex/Grid start (most likely to be good)
    # 2. Random start (to escape local minima of grid)
    
    # Restart 1: Grid
    pos, radii = _initial_lattice()
    pos_opt, radii_opt = _force_directed_anneal(pos, radii, start_time, deadline - 60.0, seed=1)
    
    if _check_validity(pos_opt, radii_opt):
        score = np.sum(radii_opt)
        if score > best_score:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
            
    # Restart 2: Random (if time permits)
    if time.time() < deadline - 60.0:
        pos, radii = _initial_random()
        pos_opt, radii_opt = _force_directed_anneal(pos, radii, time.time(), deadline - 30.0, seed=2)
        
        if _check_validity(pos_opt, radii_opt):
            score = np.sum(radii_opt)
            if score > best_score:
                best_score = score
                best_pos = pos_opt.copy()
                best_radii = radii_opt.copy()
                
    # Restart 3: Jittered Grid (if time permits)
    if time.time() < deadline - 30.0:
        pos, radii = _initial_lattice()
        rng = np.random.RandomState(99)
        pos += rng.uniform(-0.05, 0.05, pos.shape)
        pos = np.clip(pos, 0.05, 0.95)
        # Shrink to valid
        for _ in range(50):
            valid = True
            for i in range(N):
                max_r = _max_radius(pos, radii, i)
                if radii[i] > max_r:
                    radii[i] = max_r
                    valid = False
            if valid:
                break
        
        pos_opt, radii_opt = _force_directed_anneal(pos, radii, time.time(), deadline - 5.0, seed=3)
        
        if _check_validity(pos_opt, radii_opt):
            score = np.sum(radii_opt)
            if score > best_score:
                best_score = score
                best_pos = pos_opt.copy()
                best_radii = radii_opt.copy()
                
    # Final Polish: If we have time, do a very fine-grained anneal on the best solution
    if best_pos is not None and time.time() < deadline - 2.0:
        # Use very low temperature for fine tuning
        # We can just call the anneal again with a smaller T range by modifying the function?
        # Or just run it again, it will start with T_start=0.05 which might be too high for polish.
        # Let's create a specific polish function or just rely on the previous run.
        # The previous runs already cooled down.
        # A simple local search: try small random moves and accept if score improves.
        
        pos_polish = best_pos.copy()
        radii_polish = best_radii.copy()
        score_polish = best_score
        
        # Simple local search
        steps = 0
        max_steps = 5000
        while time.time() < deadline - 0.5 and steps < max_steps:
            # Pick a random circle
            i = np.random.randint(N)
            # Try a small move
            delta = np.random.normal(0, 0.001, 2)
            new_pos = pos_polish.copy()
            new_pos[i] += delta
            
            # Check if new pos is valid (within bounds)
            if new_pos[i, 0] < 0 or new_pos[i, 0] > 1 or new_pos[i, 1] < 0 or new_pos[i, 1] > 1:
                steps += 1
                continue
                
            # Grow radii for this circle
            new_radii = radii_polish.copy()
            new_radii[i] = _max_radius(new_pos, new_radii, i)
            
            # Check if this move causes any other circle to be invalid (overlap)
            # Since we only moved i, only i can overlap with others.
            # But wait, if i grows, it might overlap with j.
            # _max_radius ensures i does not overlap with j (given j's radius).
            # However, if i's radius shrinks (because it moved closer to a boundary),
            # does that help others? No, shrinking i frees space, so others could grow.
            # But we are only updating i's radius here.
            # To be strictly valid, we must ensure the whole configuration is valid.
            # Since we only changed i, and we set r_i to max possible,
            # i does not overlap with any j.
            # But do j's overlap with each other? No, they didn't change.
            # So the configuration is valid IF i is within bounds.
            
            if _check_validity(new_pos, new_radii):
                new_score = np.sum(new_radii)
                if new_score > score_polish:
                    score_polish = new_score
                    pos_polish = new_pos
                    radii_polish = new_radii
                    if score_polish > best_score:
                        best_score = score_polish
                        best_pos = pos_polish.copy()
                        best_radii = radii_polish.copy()
            steps += 1

    # Fallback if no valid solution found (should not happen)
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        # Grid fallback
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[k] = 0.09
                    k += 1
        # Shrink to valid
        for _ in range(100):
            valid = True
            for i in range(N):
                max_r = _max_radius(pos, radii, i)
                if radii[i] > max_r:
                    radii[i] = max_r
                    valid = False
            if valid:
                break
        best_pos, best_radii = pos, radii

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
