"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: Contact-topology driven nonlinear solver.
1. Generate random contact graphs (which circles touch which, and which touch walls).
2. Solve for positions/radii using least_squares to satisfy tangency constraints.
3. Iteratively increase a global scale factor (grow) and re-solve to maximize sum of radii.
"""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import least_squares

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True

def _generate_random_topology(n, seed=0):
    """
    Generate a random contact topology.
    Returns:
        contacts: list of (i, j) pairs that are intended to be tangent.
        wall_contacts: dict mapping circle index i to a bitmask of walls (1=left, 2=right, 4=bottom, 8=top)
    """
    rng = random.Random(seed)
    contacts = []
    
    # Randomly connect circles to form a graph with roughly 1.5 edges per node
    # Start with a random spanning tree to ensure connectivity (optional but helps stability)
    # For simplicity, just random edges
    num_contacts = int(n * 1.5)
    
    # Ensure some structure: connect neighbors in a grid-like fashion with noise?
    # Let's try a simple random graph with degree control
    for i in range(n):
        # Each circle connects to 2-3 random others
        num_links = rng.randint(2, 3)
        for _ in range(num_links):
            j = rng.randint(0, n - 1)
            if i != j and (i, j) not in contacts and (j, i) not in contacts:
                contacts.append((i, j))
    
    # Wall contacts: each circle touches 1-2 walls
    wall_contacts = {}
    for i in range(n):
        # Randomly pick 1 or 2 walls
        num_walls = rng.randint(1, 2)
        walls = rng.sample([1, 2, 4, 8], num_walls)
        mask = 0
        for w in walls:
            mask |= w
        wall_contacts[i] = mask
        
    return contacts, wall_contacts

def _solve_topology(contacts, wall_contacts, init_xy=None, init_r=None, n=N):
    """
    Solve for circle positions and radii given a topology.
    Variables: x[0..n-1], y[0..n-1], r[0..n-1]
    Constraints (converted to residuals for least_squares):
    1. Tangency: dist(i,j) - (r_i + r_j) = 0
    2. Wall tangency: if circle i touches wall w, distance to wall - r_i = 0
    3. Inequality constraints (handled via soft penalties or bounds):
       - r_i > 0
       - Circles inside [0,1]^2
       - Non-contact circles: dist(i,j) >= r_i + r_j (penalized if violated)
    
    To maximize sum of radii, we can add a target radius scale or use a penalty for small radii.
    However, least_squares minimizes residuals. 
    Strategy: 
    - Fix relative sizes? No, we want to find the optimal packing.
    - We can add a "dummy" variable for scale, or just penalize the negative of sum(r).
    - Better: Add a term that encourages large radii. 
    - Actually, the topology defines the *shape* of the packing up to scale and translation.
    - We can solve for the geometry that fits in the unit square with maximum scale.
    
    Let's define the residual vector:
    R = [
        (dist(i,j) - (r_i + r_j)) for each contact (i,j)
        (wall_dist(i,w) - r_i) for each wall contact (i,w)
        max(0, (r_i + r_j) - dist(i,j)) * penalty for non-contact pairs (to prevent overlap)
        max(0, r_i - wall_dist(i,w)) * penalty for non-contact walls (to prevent going out of bounds)
        -lambda * r_i  (to encourage growth? No, this will conflict with tangency)
    ]
    
    Alternative approach:
    1. Solve for unit-scale packing (radii sum to 1 or similar) that satisfies topology.
    2. Scale it to fit in the unit square.
    
    Let's use a variable 's' for global scale.
    Variables: x, y, r (normalized such that sum(r) = 1? No, let's keep r free).
    
    Let's try this:
    Minimize:
    sum( (dist(i,j) - (r_i + r_j))^2 ) for contacts
    sum( (wall_dist(i,w) - r_i)^2 ) for wall contacts
    sum( max(0, (r_i + r_j) - dist(i,j))^2 ) for non-contacts (overlap penalty)
    sum( max(0, r_i - wall_dist(i,w))^2 ) for non-wall contacts (out of bounds penalty)
    - alpha * sum(r_i)  (maximize size)
    
    This is a non-convex problem. least_squares can handle it if initialized well.
    """
    if init_xy is None:
        # Random initialization
        rng = np.random.RandomState(0)
        init_xy = rng.uniform(0.2, 0.8, size=(n, 2))
        init_r = np.ones(n) * 0.1
    else:
        init_xy = init_xy.copy()
        init_r = init_r.copy()
        
    v0 = np.concatenate([init_xy[:, 0], init_xy[:, 1], init_r])
    
    # Precompute non-contact pairs and non-contact walls for penalty
    contact_set = set()
    for i, j in contacts:
        contact_set.add((min(i,j), max(i,j)))
        
    non_contact_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in contact_set:
                non_contact_pairs.append((i, j))
                
    non_contact_walls = []
    for i in range(n):
        mask = wall_contacts[i]
        # Walls not in mask
        for w in [1, 2, 4, 8]:
            if not (mask & w):
                non_contact_walls.append((i, w))
                
    def objective(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        
        residuals = []
        
        # 1. Contact tangencies
        for i, j in contacts:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            residuals.append(dist - (r[i] + r[j]))
            
        # 2. Wall tangencies
        for i, mask in wall_contacts.items():
            if mask & 1: # Left
                residuals.append(x[i] - r[i])
            if mask & 2: # Right
                residuals.append((1.0 - x[i]) - r[i])
            if mask & 4: # Bottom
                residuals.append(y[i] - r[i])
            if mask & 8: # Top
                residuals.append((1.0 - y[i]) - r[i])
                
        # 3. Non-contact overlap penalty
        for i, j in non_contact_pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            viol = (r[i] + r[j]) - dist
            if viol > 0:
                residuals.append(viol * 10.0) # Heavy penalty
            
        # 4. Non-contact wall violation penalty
        for i, w in non_contact_walls:
            if w == 1: # Left
                viol = r[i] - x[i]
            elif w == 2: # Right
                viol = r[i] - (1.0 - x[i])
            elif w == 4: # Bottom
                viol = r[i] - y[i]
            else: # Top
                viol = r[i] - (1.0 - y[i])
            if viol > 0:
                residuals.append(viol * 10.0)
                
        # 5. Size maximization: penalize small radii
        # We want to maximize sum(r). 
        # If we just minimize residuals, r might go to 0.
        # Add a term: (target_r - r_i) where target_r is a running average?
        # Or simply: minimize -sum(r) is not directly possible in least_squares (minimizes sum of squares).
        # We can add residuals: (r_min - r_i) where r_min is a threshold.
        # Let's add a soft constraint: r_i should be at least some value.
        # But we don't know the value.
        # Instead, let's normalize the problem.
        # Let's add residuals: (1.0/n - r_i) * weight? No.
        
        # Trick: Add a dummy variable 'scale' that we want to maximize?
        # Least squares minimizes. 
        # Let's add residuals: (r_i - r_target) where r_target is updated?
        
        # Better: 
        # The topology defines the shape. The scale is determined by the fit to the unit square.
        # If we fix the topology, the relative sizes are fixed by the tangencies?
        # No, the system is under-determined or has a degree of freedom for scale.
        # Actually, if we have enough constraints, the scale is fixed.
        # If not, the solver might pick a small scale.
        
        # To encourage larger circles, we can add a penalty for being far from a "typical" radius.
        # Or, we can run the solver, get a solution, scale it up, and re-solve.
        
        # For now, let's add a gentle penalty for small radii.
        # Residuals: (0.05 - r_i) if r_i < 0.05
        for i in range(n):
            if r[i] < 0.05:
                residuals.append((0.05 - r[i]) * 5.0)
                
        return np.array(residuals)
        
    try:
        result = least_squares(objective, v0, method='trf', bounds=([0.01]*n + [0.01]*n + [0.01]*n, [0.99]*n + [0.99]*n + [0.5]*n), max_nfev=1000)
    except Exception:
        return None, None
        
    v_opt = result.x
    xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    r = v_opt[2*n:3*n]
    
    # Check validity
    total, valid = _eval_packing(xy, r)
    if not valid:
        # Try to fix by shrinking
        scale = 0.9
        for _ in range(10):
            r_new = r * scale
            # Re-center?
            xy_new = xy
            total, valid = _eval_packing(xy_new, r_new)
            if valid:
                return xy_new, r_new
            scale *= 0.9
        return None, None
        
    return xy, r

def _topology_search(deadline):
    """
    Search over random topologies.
    """
    best_xy = None
    best_r = None
    best_total = 0.0
    
    num_attempts = 20
    start_time = time.time()
    
    for attempt in range(num_attempts):
        if time.time() > deadline - 10:
            break
            
        seed = attempt * 123 + 42
        contacts, wall_contacts = _generate_random_topology(N, seed=seed)
        
        # Try to solve
        xy, r = _solve_topology(contacts, wall_contacts)
        
        if xy is not None:
            total, valid = _eval_packing(xy, r)
            if valid and total > best_total:
                best_total = total
                best_xy = xy.copy()
                best_r = r.copy()
                
    return best_xy, best_r, best_total

def _grow_and_repack(xy, r, deadline):
    """
    Iteratively scale up the packing and re-solve to find a better fit.
    This is a heuristic to escape local minima in scale.
    """
    best_xy = xy.copy()
    best_r = r.copy()
    best_total, _ = _eval_packing(best_xy, best_r)
    
    # Extract topology from current best packing
    # Determine contacts: pairs with dist < r_i + r_j + epsilon
    # Determine walls: circles close to walls
    contacts = []
    for i in range(N):
        for j in range(i+1, N):
            dx = xy[i,0] - xy[j,0]
            dy = xy[i,1] - xy[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] + 0.01:
                contacts.append((i, j))
                
    wall_contacts = {}
    for i in range(N):
        mask = 0
        if xy[i,0] < r[i] + 0.01: mask |= 1
        if 1.0 - xy[i,0] < r[i] + 0.01: mask |= 2
        if xy[i,1] < r[i] + 0.01: mask |= 4
        if 1.0 - xy[i,1] < r[i] + 0.01: mask |= 8
        if mask == 0:
            # Assign random walls if none close
            mask = 1
        wall_contacts[i] = mask
        
    # Try to grow
    scale = 1.01
    for _ in range(50):
        if time.time() > deadline - 5:
            break
            
        # Scale radii
        r_new = r * scale
        
        # Solve with new initial guess
        xy_opt, r_opt = _solve_topology(contacts, wall_contacts, init_xy=xy, init_r=r_new)
        
        if xy_opt is not None:
            total, valid = _eval_packing(xy_opt, r_opt)
            if valid and total > best_total:
                best_total = total
                best_xy = xy_opt.copy()
                best_r = r_opt.copy()
                # Continue growing
                scale = 1.01
            else:
                # Reduce scale if it failed
                scale = 0.99
                if scale < 1.0:
                    break
        else:
            break
            
    return best_xy, best_r, best_total

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_xy = None
    best_r = None
    best_total = 0.0
    
    # Phase 1: Topology Search
    xy, r, total = _topology_search(deadline)
    if xy is not None:
        best_xy = xy
        best_r = r
        best_total = total
        
    # Phase 2: Grow and Repack
    if best_xy is not None and time.time() < deadline - 10:
        xy, r, total = _grow_and_repack(best_xy, best_r, deadline)
        if total > best_total:
            best_xy = xy
            best_r = r
            best_total = total
            
    # Fallback: Simple grid if topology search fails
    if best_xy is None:
        side = 5
        best_xy = np.zeros((N, 2))
        best_r = np.zeros(N)
        for i in range(N):
            gx = i % side
            gy = i // side
            best_xy[i, 0] = (gx + 0.5) / side
            best_xy[i, 1] = (gy + 0.5) / side
            best_r[i] = 0.4 / side
        best_total, _ = _eval_packing(best_xy, best_r)
        
    # Final validation and clipping
    result = np.zeros((N, 3))
    for i in range(N):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        # Emergency fallback
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
