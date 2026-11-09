"""Circle packing in unit square - force-directed with LP radius optimization."""
import numpy as np
import time
from scipy.optimize import linprog

N = 26
DEADLINE = 145.0  # safety margin before 150s kill

def _max_radius_to_wall(x, y):
    """Max radius for a circle at (x,y) to stay in [0,1]^2."""
    return min(x, 1.0 - x, y, 1.0 - y)

def _solve_radii_lp(pos, r_init=None, time_limit=0.5):
    """
    Given fixed positions, solve LP to maximize sum of radii
    subject to:
      r_i + r_j <= dist(i,j)  for all i<j
      r_i <= dist(i, wall)    for each circle
      r_i >= 0
    """
    n = len(pos)
    # Variables: r_0, ..., r_{n-1}
    # Maximize sum(r_i) => minimize -sum(r_i)
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    # Pairwise non-overlap: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
    
    # Wall constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    for i in range(n):
        max_r = _max_radius_to_wall(pos[i][0], pos[i][1])
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(max(max_r, 0.0))
    
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    
    bounds = [(0.0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            r = np.clip(res.x, 0.0, None)
            return r
    except Exception:
        pass
    
    # Fallback: simple min-based radii
    r = np.zeros(n)
    for i in range(n):
        max_r = _max_radius_to_wall(pos[i][0], pos[i][1])
        for j in range(n):
            if i != j:
                d = np.linalg.norm(pos[i] - pos[j])
                max_r = min(max_r, d / 2.0)
        r[i] = max(max_r, 0.0)
    return r

def _force_relax(pos, radii, steps=200, dt=0.01, time_limit=0.5):
    """
    Force-directed relaxation: circles repel when overlapping,
    and are pushed back into the unit square.
    """
    n = len(pos)
    pos = pos.copy().astype(float)
    radii = radii.copy()
    start = time.time()
    
    for step in range(steps):
        if time.time() - start > time_limit:
            break
        
        forces = np.zeros((n, 2))
        
        # Pairwise repulsion for overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[j] - pos[i]
                dist = np.linalg.norm(d)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    # Force proportional to overlap
                    f = overlap * d / dist
                    forces[i] -= f
                    forces[j] += f
                elif dist <= 1e-12:
                    # Nearly coincident, push apart randomly
                    f = np.array([np.random.randn(), np.random.randn()]) * 0.01
                    forces[i] -= f
                    forces[j] += f
        
        # Wall repulsion
        for i in range(n):
            r = radii[i]
            # Left wall
            if pos[i][0] < r:
                forces[i][0] += (r - pos[i][0]) * 2.0
            # Right wall
            if pos[i][0] > 1.0 - r:
                forces[i][0] -= (pos[i][0] - (1.0 - r)) * 2.0
            # Bottom wall
            if pos[i][1] < r:
                forces[i][1] += (r - pos[i][1]) * 2.0
            # Top wall
            if pos[i][1] > 1.0 - r:
                forces[i][1] -= (pos[i][1] - (1.0 - r)) * 2.0
        
        # Apply forces with adaptive step size
        max_force = np.max(np.linalg.norm(forces, axis=1)) if n > 0 else 0
        if max_force > 0:
            step_size = min(dt, 0.5 / max_force)
            pos += step_size * forces
        
        # Clamp to unit square (with small epsilon)
        eps = 1e-6
        pos[:, 0] = np.clip(pos[:, 0], eps, 1.0 - eps)
        pos[:, 1] = np.clip(pos[:, 1], eps, 1.0 - eps)
    
    return pos

def _is_valid(pos, radii, tol=1e-6):
    """Check if packing is valid: all circles inside [0,1]^2 and non-overlapping."""
    n = len(pos)
    for i in range(n):
        x, y, r = pos[i][0], pos[i][1], radii[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1.0 + tol:
            return False
        if y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _total_radius(pos, radii):
    return np.sum(radii)

def _initial_grid(n):
    """Create initial grid positions."""
    side = int(np.ceil(np.sqrt(n)))
    pos = []
    for i in range(side):
        for j in range(side):
            if len(pos) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            pos.append([x, y])
    pos = np.array(pos[:n])
    return pos

def _initial_random(n, seed=42):
    """Create random initial positions."""
    rng = np.random.RandomState(seed)
    pos = rng.rand(n, 2) * 0.8 + 0.1
    return pos

def construct():
    global DEADLINE
    t0 = time.time()
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Multiple restarts with different initial configurations
    restarts = []
    
    # Grid initial
    restarts.append(('grid', _initial_grid(N)))
    
    # Random initial with different seeds
    for seed in [0, 1, 2, 3, 4, 5]:
        restarts.append((f'rand{seed}', _initial_random(N, seed=seed)))
    
    # Hexagonal-like initial
    side = int(np.ceil(np.sqrt(N)))
    hex_pos = []
    for i in range(side):
        for j in range(side):
            if len(hex_pos) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side + (0.25 / side if i % 2 == 1 else 0)
            hex_pos.append([x, y])
    hex_pos = np.array(hex_pos[:N])
    # Clamp to valid range
    hex_pos[:, 0] = np.clip(hex_pos[:, 0], 0.05, 0.95)
    hex_pos[:, 1] = np.clip(hex_pos[:, 1], 0.05, 0.95)
    restarts.append(('hex', hex_pos))
    
    for label, init_pos in restarts:
        if time.time() - t0 > DEADLINE - 10:
            break
        
        pos = init_pos.copy()
        
        # Alternating optimization: LP radii + force relaxation
        for iteration in range(50):
            if time.time() - t0 > DEADLINE - 5:
                break
            
            # Step 1: Solve LP for optimal radii given positions
            radii = _solve_radii_lp(pos, time_limit=0.3)
            
            # Step 2: Force relaxation to reduce overlaps
            pos = _force_relax(pos, radii, steps=100, dt=0.02, time_limit=0.3)
            
            # Re-solve radii after relaxation
            radii = _solve_radii_lp(pos, time_limit=0.3)
            
            # Check validity and update best
            if _is_valid(pos, radii, tol=1e-5):
                score = _total_radius(pos, radii)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
            
            # If we have a good solution and time is running out, break
            if time.time() - t0 > DEADLINE - 15:
                break
        
        # Final refinement for this restart
        if time.time() - t0 < DEADLINE - 5:
            # More aggressive force relaxation
            radii = _solve_radii_lp(pos, time_limit=0.3)
            pos = _force_relax(pos, radii, steps=300, dt=0.01, time_limit=1.0)
            radii = _solve_radii_lp(pos, time_limit=0.3)
            
            if _is_valid(pos, radii, tol=1e-5):
                score = _total_radius(pos, radii)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
    
    # If no valid solution found, fall back to grid
    if best_pos is None:
        pos = _initial_grid(N)
        radii = _solve_radii_lp(pos)
        best_pos = pos
        best_radii = radii
    
    # Final check and clamp
    # Ensure all circles are strictly inside
    for i in range(N):
        r = best_radii[i]
        x = best_pos[i][0]
        y = best_pos[i][1]
        # Clamp radius to ensure it fits
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if r > max_r:
            best_radii[i] = max_r
    
    # Re-check validity and fix if needed
    if not _is_valid(best_pos, best_radii, tol=1e-4):
        # Reduce radii slightly
        best_radii *= 0.99
        if not _is_valid(best_pos, best_radii, tol=1e-4):
            # Fall back to simple grid
            side = int(np.ceil(np.sqrt(N)))
            cells = []
            for i in range(side):
                for j in range(side):
                    if len(cells) >= N:
                        break
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 1e-3
                    cells.append([x, y, r])
            return np.array(cells)
    
    # Build result
    result = np.zeros((N, 3))
    result[:, 0] = best_pos[:, 0]
    result[:, 1] = best_pos[:, 1]
    result[:, 2] = best_radii
    
    return result
