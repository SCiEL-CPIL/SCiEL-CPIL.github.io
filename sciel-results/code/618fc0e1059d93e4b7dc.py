"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True

def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])

def _physics_relax(xy, r, deadline, steps=2000, cooling_rate=0.999):
    """
    Physics-based relaxation using repulsive forces.
    - Position update: Repulsion from overlapping circles and walls.
    - Radius update: Grow towards minimum clearance to neighbors/walls.
    - Annealing: Damping factor decreases to settle into a local minimum.
    """
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    
    # Initial velocity
    v = np.zeros_like(xy)
    
    best_total = _eval_packing(xy, r)[0]
    best_xy = xy.copy()
    best_r = r.copy()
    
    temperature = 1.0
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # 1. Calculate Forces
        forces = np.zeros_like(xy)
        
        # Wall repulsion
        # If x < r, push right. If x > 1-r, push left.
        # Force magnitude proportional to violation
        for i in range(n):
            # Left wall
            if xy[i, 0] < r[i]:
                forces[i, 0] += (r[i] - xy[i, 0]) * 10.0
            # Right wall
            if xy[i, 0] > 1.0 - r[i]:
                forces[i, 0] -= (xy[i, 0] - (1.0 - r[i])) * 10.0
            # Bottom wall
            if xy[i, 1] < r[i]:
                forces[i, 1] += (r[i] - xy[i, 1]) * 10.0
            # Top wall
            if xy[i, 1] > 1.0 - r[i]:
                forces[i, 1] -= (xy[i, 1] - (1.0 - r[i])) * 10.0

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq)
                
                # Desired distance is r[i] + r[j]
                # If dist < desired, they overlap
                overlap = (r[i] + r[j]) - dist
                
                if overlap > 0:
                    if dist < 1e-6:
                        # Random kick if exactly overlapping
                        angle = np.random.rand() * 2 * np.pi
                        fx = np.cos(angle)
                        fy = np.sin(angle)
                    else:
                        fx = dx / dist
                        fy = dy / dist
                    
                    # Force is proportional to overlap (spring-like)
                    f_mag = overlap * 50.0
                    
                    forces[i, 0] += fx * f_mag
                    forces[i, 1] += fy * f_mag
                    forces[j, 0] -= fx * f_mag
                    forces[j, 1] -= fy * f_mag

        # 2. Update Radii
        # Target radius for each circle is the minimum clearance to walls and other circles
        # We want to maximize sum(r), so we grow r towards this clearance.
        new_r = r.copy()
        for i in range(n):
            min_clear = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            for j in range(n):
                if i == j: continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                # Clearance to circle j is dist - r[j]
                # We can grow r[i] until r[i] + r[j] = dist -> r[i] = dist - r[j]
                clear_to_j = dist - r[j]
                if clear_to_j < min_clear:
                    min_clear = clear_to_j
            
            # Grow radius towards min_clear, but don't exceed it
            # Use a learning rate for radius growth
            target_r = max(0.001, min_clear)
            # Exponential approach
            new_r[i] = r[i] + 0.1 * (target_r - r[i])
            
        r = new_r
        
        # 3. Update Positions
        # v = damping * v + force
        # xy += v * dt
        # Add some noise to escape local minima (simulated annealing aspect)
        noise = np.random.normal(0, temperature * 0.01, size=(n, 2))
        
        v = 0.5 * v + forces + noise
        xy += v * 0.01
        
        # Clamp positions to stay inside [0,1] roughly to prevent runaway
        xy[:, 0] = np.clip(xy[:, 0], 0.0, 1.0)
        xy[:, 1] = np.clip(xy[:, 1], 0.0, 1.0)
        
        # Cool down
        temperature *= cooling_rate
        
        # Evaluate
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_xy = xy.copy()
            best_r = r.copy()

    return best_xy, best_r, best_total

def _topology_init():
    """
    Initialize based on a hexagonal lattice pattern which is known to be dense.
    """
    positions = []
    # Hexagonal packing in a square
    # Row 0: 0.5/side, 1.5/side...
    # Row 1: offset by 0.5/side
    side = 6 # 6x5 or 6x6 grid
    # Let's try a 5x5 grid with some offsets to get 26
    # Actually, let's just use a perturbed grid
    rng = np.random.RandomState(42)
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    
    # Generate 26 points in a 5x5 grid (25 points) + 1 in center? 
    # Or 6x5 = 30, pick 26?
    # Let's do a 5x5 grid and add 1 in the middle of the largest gap?
    # Simpler: 5x5 grid, r = 0.1
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < N:
                xy[idx, 0] = (i + 0.5) / 5.0
                xy[idx, 1] = (j + 0.5) / 5.0
                r[idx] = 0.09
    if N > 25:
        # Add one more in the center? No, center is occupied.
        # Add one at a random position
        xy[25, 0] = 0.5
        xy[25, 1] = 0.5
        r[25] = 0.01
        # Actually, better to just use 5x5 and shrink slightly to fit?
        # Or use a 6x5 grid and remove 4?
        # Let's stick to 5x5 for now, the 26th circle will be handled by physics
    
    return xy, r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Strategy: Multiple restarts with different initializations and physics relaxation
    # 1. Grid Init
    for side in [5, 6]:
        if time.time() > deadline:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        
        out_xy, out_r, total = _physics_relax(xy, r, deadline, steps=1500)
        if total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # 2. Topology Init (Hex-like)
    if time.time() < deadline - 30:
        xy, r = _topology_init()
        out_xy, out_r, total = _physics_relax(xy, r, deadline, steps=1500)
        if total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # 3. Random Inits
    for seed in range(5):
        if time.time() > deadline:
            break
        rng = np.random.RandomState(seed)
        xy = rng.uniform(0.1, 0.9, size=(N, 2))
        r = rng.uniform(0.02, 0.04, size=N)
        
        out_xy, out_r, total = _physics_relax(xy, r, deadline, steps=1000)
        if total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Fallback
    if best_xy is None or not best_valid:
        side = 5
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        best_xy = xy
        best_r = r
        best_total = _eval_packing(xy, r)[0]

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
