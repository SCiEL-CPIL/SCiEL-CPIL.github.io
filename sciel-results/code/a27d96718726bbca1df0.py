import numpy as np
import time

N = 50
DEADLINE = 135.0  # Safety margin before 150s kill

def _radius(centers):
    """
    Compute the max common radius for given centers.
    Uses O(N^2) vectorized distance calculation which is fast enough for N=50.
    """
    # Wall radius
    wall_r = np.min(0.5 - np.abs(centers))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    return min(wall_r, min_pair / 2.0)

def _resolve_collisions(centers, r, iterations=50):
    """
    Iteratively push apart overlapping circles to satisfy non-overlap constraints.
    This is a simple projection method onto the feasible set for a given r.
    """
    centers = centers.copy()
    for _ in range(iterations):
        moved = False
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Find overlapping pairs
        # We want dist_ij < 2r
        # To be stable, we only resolve the worst overlaps first or all at once?
        # All at once is simpler.
        overlaps = dist < 2.0 * r
        
        if not np.any(overlaps):
            # Check walls
            for i in range(N):
                # Left wall
                if centers[i, 0] < -0.5 + r:
                    centers[i, 0] = -0.5 + r
                    moved = True
                # Right wall
                if centers[i, 0] > 0.5 - r:
                    centers[i, 0] = 0.5 - r
                    moved = True
                # Bottom wall
                if centers[i, 1] < -0.5 + r:
                    centers[i, 1] = -0.5 + r
                    moved = True
                # Top wall
                if centers[i, 1] > 0.5 - r:
                    centers[i, 1] = 0.5 - r
                    moved = True
            
            if not moved:
                break
            
            continue
        
        # Resolve overlaps
        # For each overlapping pair, push them apart along the vector connecting them
        # We accumulate displacements
        disp = np.zeros_like(centers)
        
        # Iterate over unique pairs to avoid double counting in accumulation logic 
        # or just do it symmetrically.
        # Vectorized approach for displacement is tricky with conditional logic.
        # For N=50, loop is fine.
        for i in range(N):
            for j in range(i + 1, N):
                if overlaps[i, j]:
                    d_vec = centers[i] - centers[j]
                    d = dist[i, j]
                    if d > 1e-9:
                        # Required separation is 2r
                        # Current separation is d
                        # Need to move apart by (2r - d)
                        # Split the correction: each moves (2r - d) / 2
                        # Direction is d_vec / d
                        correction = (2.0 * r - d) / 2.0
                        unit_vec = d_vec / d
                        disp[i] += correction * unit_vec
                        disp[j] -= correction * unit_vec
                    else:
                        # Coincident points, push randomly
                        disp[i] += np.random.normal(0, r, 2)
                        disp[j] -= np.random.normal(0, r, 2)
        
        centers += disp
        
        # Hard clip to walls (projections don't always stay inside if pushed hard)
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
    return centers

def _pack_with_growth(start_centers, max_time=10.0):
    """
    Start with a configuration, and incrementally increase radius r.
    At each step, resolve collisions.
    If we get stuck (can't resolve), add noise (Langevin) and retry or restart growth.
    """
    centers = start_centers.copy()
    # Initial radius estimate
    r = _radius(centers)
    
    # We want to grow r.
    # Step size for growth
    # Start small, maybe increase? Or constant?
    # Let's use a dynamic step: if we succeed, keep growing. If we fail, shrink r slightly and add noise.
    
    current_r = r
    step_r = 0.001
    start_time = time.time()
    
    # Track best
    best_centers = centers.copy()
    best_r = current_r
    
    # Pre-allocate for speed? No, just copy.
    
    while time.time() - start_time < max_time:
        # Try to grow
        new_r = current_r + step_r
        
        # Attempt to resolve collisions at new_r
        # We need to check if a valid configuration exists for new_r.
        # The resolution loop tries to find one.
        
        # To make it more robust, we can run the resolution loop and then check if the final
        # radius is actually >= new_r.
        
        resolved = _resolve_collisions(centers, new_r, iterations=20)
        
        # Check if resolved is valid for new_r
        check_r = _radius(resolved)
        
        if check_r >= new_r - 1e-9:
            # Success
            centers = resolved
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
            
            # Maybe increase step size slightly?
            # step_r = min(0.01, step_r * 1.1)
            
        else:
            # Failure: couldn't find a configuration for new_r
            # Add noise to escape local minimum
            # Noise magnitude proportional to r or step_r
            noise = np.random.normal(0, step_r * 2.0, centers.shape)
            centers = centers + noise
            # Clip to box
            centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
            centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
            
            # Reduce current_r slightly to allow recovery
            current_r = max(0.01, current_r - step_r * 0.5)
            # Reduce step size?
            step_r = max(0.0001, step_r * 0.9)
            
            # Update best if noise accidentally improved it (unlikely but possible)
            check_r = _radius(centers)
            if check_r > best_r:
                best_r = check_r
                best_centers = centers.copy()
                
    return best_centers

def _generate_initial_configs():
    """
    Generate diverse initial configurations.
    """
    inits = []
    
    # 1. Rectangular Grids
    # 7x8 = 56, take 50
    for nx, ny in [(7, 8), (8, 7), (10, 5)]:
        ticks_x = (np.arange(nx) + 0.5) / nx - 0.5
        ticks_y = (np.arange(ny) + 0.5) / ny - 0.5
        grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
        inits.append(grid[:N].copy())
        
    # 2. Hexagonal Grid
    # Hexagonal packing is denser.
    # Approximate radius for hex packing of 50 circles in unit square.
    # Area per circle ~ sqrt(3)/2 * (2r)^2 ? No, area per point in hex lattice is sqrt(3)/2 * s^2 where s is distance.
    # Total area 1. N=50. Area per point ~ 0.02.
    # sqrt(3)/2 * s^2 = 0.02 => s^2 = 0.023 => s ~ 0.15.
    # r ~ s/2 = 0.075.
    
    s = 0.15
    dy = s * np.sqrt(3) / 2
    points = []
    for j in range(10):
        for i in range(10):
            x = i * s
            y = j * dy
            if j % 2 != 0:
                x += s / 2
            points.append((x, y))
    hex_grid = np.array(points)
    # Center and scale to fit in [-0.5, 0.5]
    hex_grid -= np.mean(hex_grid, axis=0)
    max_ext = np.max(np.abs(hex_grid))
    if max_ext > 0.45:
        hex_grid = hex_grid * (0.45 / max_ext)
    inits.append(hex_grid[:N].copy())
    
    # 3. Random
    np.random.seed(42)
    inits.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    
    return inits

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    inits = _generate_initial_configs()
    
    # Phase 1: Run growth on each initial config
    # Allocate time per init
    # We have ~135s. 5 inits. ~25s each.
    # But we want to be safe.
    
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Adaptive time allocation
        # If we have plenty of time, give it 20s.
        # If time is tight, give it remaining/2.
        time_left = DEADLINE - 10 - (time.time() - start_time)
        alloc = min(20.0, time_left / 2.0)
        
        if alloc < 1.0:
            break
            
        # Run growth
        res_centers = _pack_with_growth(init, max_time=alloc)
        r = _radius(res_centers)
        
        if r > best_r:
            best_r = r
            best_centers = res_centers.copy()
            
    # Phase 2: Refine the best found so far with more time
    # If we have time left, re-run growth on the best center found, but with different noise seeds or longer run.
    # Or just run another few random restarts if time permits.
    
    if time.time() - start_time < DEADLINE - 15:
        # Use remaining time for a few more random restarts or refining the best
        # Let's refine the best one with a longer run
        # Or start a new random one?
        # The growth method is stochastic (noise on failure), so re-running on the same start might find different paths.
        # But re-running on the BEST result might be better to push it further.
        
        # Let's try 2 more random restarts if time allows, and 1 refinement of the best.
        
        # Refinement of best
        if best_centers is not None:
            time_left = DEADLINE - 15 - (time.time() - start_time)
            if time_left > 5:
                res_centers = _pack_with_growth(best_centers, max_time=time_left * 0.5)
                r = _radius(res_centers)
                if r > best_r:
                    best_r = r
                    best_centers = res_centers.copy()
            
        # New random restarts
        for _ in range(3):
            if time.time() - start_time > DEADLINE - 5:
                break
            time_left = DEADLINE - 5 - (time.time() - start_time)
            if time_left < 5:
                break
            
            np.random.seed(int(time.time() * 1000) % 100000)
            c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
            
            res_centers = _pack_with_growth(c_rand, max_time=time_left * 0.5)
            r = _radius(res_centers)
            if r > best_r:
                best_r = r
                best_centers = res_centers.copy()

    if best_centers is None:
        # Fallback: return a grid
        return _generate_initial_configs()[0]
        
    return best_centers

if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct()
    r = _radius(res)
    print(f"Time: {time.time() - start:.2f}s, Radius: {r:.10f}")
    print(res)
