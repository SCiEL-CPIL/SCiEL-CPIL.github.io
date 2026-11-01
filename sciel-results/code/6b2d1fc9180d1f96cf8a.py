"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _is_valid(circles, n=N, tol=1e-9):
    """
    Check if the circles are valid:
    1. All circles are within [0, 1] x [0, 1].
    2. No two circles overlap.
    """
    if len(circles) != n:
        return False
    
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check boundaries
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol) or np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
        
    # Check overlaps using vectorized distance calculation
    # dist[i, j] is distance between circle i and j
    # We only need i < j
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist*min_dist - tol:
                return False
    return True

def _run_md_simulation(n, seed, max_steps, dt, cooling_rate, max_time):
    """
    Run a Molecular Dynamics simulation to pack n circles.
    Returns the best valid configuration found or None.
    """
    rng = np.random.RandomState(seed)
    
    # Initialization: Jittered grid
    side = int(np.ceil(np.sqrt(n)))
    # We need to place n circles in a side x side grid
    # If n is not a perfect square, we fill row by row
    positions = np.zeros((n, 2))
    radii = np.zeros(n)
    
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n:
                break
            # Center of cell
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            # Add jitter
            positions[idx, 0] = cx + rng.uniform(-0.1/side, 0.1/side)
            positions[idx, 1] = cy + rng.uniform(-0.1/side, 0.1/side)
            # Initial radius slightly smaller than cell half-width
            radii[idx] = 0.5 / side * 0.9
            idx += 1
            
    # Ensure we have exactly n circles (if grid was larger)
    if idx < n:
        # This shouldn't happen if side = ceil(sqrt(n))
        pass
        
    velocities = np.zeros((n, 2))
    
    best_score = 0.0
    best_config = None
    start_time = time.time()
    
    # MD Loop
    for step in range(max_steps):
        if time.time() - start_time > max_time:
            break
            
        # Calculate forces
        forces = np.zeros((n, 2))
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                
                if dist_sq < min_dist*min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-10:
                        # Random push if overlapping exactly
                        angle = rng.uniform(0, 2*np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1e-10
                    overlap = min_dist - dist
                    # Force magnitude proportional to overlap
                    # Spring constant k
                    k = 100.0 
                    f_mag = k * overlap
                    fx = f_mag * (dx / dist)
                    fy = f_mag * (dy / dist)
                    
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Boundary forces
        # Push circles back if they go out of [0, 1]
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            
            # Left
            if x - r < 0:
                forces[i, 0] += 100.0 * (r - x)
            # Right
            if x + r > 1:
                forces[i, 0] -= 100.0 * (x + r - 1)
            # Bottom
            if y - r < 0:
                forces[i, 1] += 100.0 * (r - y)
            # Top
            if y + r > 1:
                forces[i, 1] -= 100.0 * (y + r - 1)
            
        # Update velocities (Velocity Verlet simplified / Euler with damping)
        # v_new = v_old + F*dt - damping*v_old
        # We use damping to cool the system
        damping = 0.5
        velocities = velocities * (1 - damping) + forces * dt
        
        # Update positions
        positions = positions + velocities * dt
        
        # Enforce hard boundaries (clamp) to prevent explosion
        # This is a soft constraint in forces, but hard clamp helps stability
        for i in range(n):
            r = radii[i]
            positions[i, 0] = np.clip(positions[i, 0], r, 1-r)
            positions[i, 1] = np.clip(positions[i, 1], r, 1-r)
            
        # Check validity and score
        # We only check validity every few steps to save time, or if forces are small
        if step % 10 == 0 or step == max_steps - 1:
             if _is_valid(positions, n, tol=1e-8):
                current_score = np.sum(radii)
                if current_score > best_score:
                    best_score = current_score
                    best_config = np.column_stack((positions, radii)).copy()
                    
    return best_config

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    n = N
    best_overall_config = None
    best_overall_score = 0.0
    
    # We will run multiple MD simulations with different seeds and parameters
    # Strategy: 
    # 1. Start with a few fast simulations with coarse parameters.
    # 2. If time permits, run more refined simulations.
    
    seeds = [42, 123, 999, 2023, 555]
    
    # Parameters for MD
    # dt is time step
    # max_steps is number of steps per simulation
    # cooling_rate is not explicitly used in the simple loop above, 
    # but we can vary dt or damping if we wanted a more complex schedule.
    # For now, we rely on random restarts.
    
    # Let's try a mix of initializations
    # 1. Standard grid
    # 2. Random uniform
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        # Run MD
        # Allocate ~20s per simulation max, but we have 5 seeds, so 10s each?
        # Actually, MD is fast in Python if vectorized, but this is O(N^2) python loops.
        # N=32, so N^2/2 = 512 pairs. 
        # 512 operations per step. 1000 steps = 512,000 ops. Very fast.
        # We can afford many steps.
        
        config = _run_md_simulation(n, seed, max_steps=5000, dt=0.01, cooling_rate=0.99, max_time=15.0)
        
        if config is not None:
            score = np.sum(config[:, 2])
            if score > best_overall_score:
                best_overall_score = score
                best_overall_config = config
                
    # If we have time left, try a local refinement on the best found
    # Or just run more random restarts
    
    remaining_time = deadline - time.time()
    if remaining_time > 5.0 and best_overall_config is not None:
         # Try to perturb the best solution and re-optimize
         # Simple local search: jitter radii and positions slightly, then run MD again
         base = best_overall_config.copy()
         for i in range(5): # 5 local refinements
             if time.time() > deadline:
                 break
             perturbed = base.copy()
             rng = np.random.RandomState(1000 + i)
             # Jitter positions
             perturbed[:, 0] += rng.normal(0, 0.01, n)
             perturbed[:, 1] += rng.normal(0, 0.01, n)
             # Jitter radii (increase slightly to test if it fits)
             perturbed[:, 2] *= 1.001
             
             # Run a short MD to settle
             # We need to adapt _run_md_simulation to accept initial positions/radii
             # For simplicity, let's just call it with a new seed but the internal init is grid.
             # This local refinement is skipped for now as _run_md_simulation doesn't take init.
             # Let's just run more global restarts.
             pass
             
    # If no valid config found (unlikely), fallback to grid
    if best_overall_config is None:
        side = int(np.ceil(np.sqrt(n)))
        circles = np.zeros((n, 3))
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= n:
                    break
                circles[idx, 0] = (i + 0.5) / side
                circles[idx, 1] = (j + 0.5) / side
                circles[idx, 2] = 0.5 / side * 0.9
                idx += 1
        best_overall_config = circles
        
    return best_overall_config

# EVOLVE-BLOCK-END
