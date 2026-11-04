"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
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
    return np.sum(r), True

def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Initial radius: half the cell size minus a small margin
                r = 0.5 / side - 0.005
                positions.append([x, y, r])
    # Pad if needed
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])

def _sa_packing(xy, r, deadline, t_start=1.0, t_end=0.001, cooling=0.9999):
    """
    Simulated Annealing for circle packing.
    """
    n = len(xy)
    current_xy = xy.copy()
    current_r = r.copy()
    current_total, current_valid = _eval_packing(current_xy, current_r)
    
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    best_total = current_total if current_valid else 0.0
    best_valid = current_valid
    
    t = t_start
    step_size = 0.05
    
    # Precompute indices for faster distance checks? 
    # For N=26, O(N^2) is fast enough.
    
    while time.time() < deadline and t > t_end:
        # Choose a random circle to perturb
        i = random.randint(0, n - 1)
        
        # Perturb position and radius
        old_x, old_y, old_r = current_xy[i, 0], current_xy[i, 1], current_r[i]
        
        # Gaussian perturbation
        new_x = old_x + random.gauss(0, step_size)
        new_y = old_y + random.gauss(0, step_size)
        new_r = old_r + random.gauss(0, step_size * 0.5)
        
        # Clip to bounds to ensure validity checks are meaningful
        # We don't strictly need to clip for SA if we penalize, but it helps convergence
        new_x = np.clip(new_x, 0.0, 1.0)
        new_y = np.clip(new_y, 0.0, 1.0)
        new_r = np.clip(new_r, 0.01, 0.4)
        
        # Apply move
        current_xy[i, 0] = new_x
        current_xy[i, 1] = new_y
        current_r[i] = new_r
        
        # Evaluate
        new_total, new_valid = _eval_packing(current_xy, current_r)
        
        if new_valid:
            if new_total > best_total:
                best_total = new_total
                best_valid = True
                best_xy = current_xy.copy()
                best_r = current_r.copy()
            
            # Acceptance criterion
            if new_total > current_total:
                current_total = new_total
            else:
                # Calculate delta (negative)
                delta = new_total - current_total
                # Metropolis
                if random.random() < math.exp(delta / t):
                    current_total = new_total
                else:
                    # Revert
                    current_xy[i, 0] = old_x
                    current_xy[i, 1] = old_y
                    current_r[i] = old_r
        else:
            # Invalid configuration: strong penalty
            # Estimate penalty magnitude based on violations
            penalty = 0.0
            x, y, ri = new_x, new_y, new_r
            penalty += max(0, ri - x) + max(0, ri - (1-x)) + max(0, ri - y) + max(0, ri - (1-y))
            for j in range(n):
                if i == j: continue
                dx = new_x - current_xy[j, 0]
                dy = new_y - current_xy[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = new_r + current_r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist)
            
            new_score = new_total - 10.0 * penalty
            if new_score > current_total:
                current_total = new_score
            else:
                delta = new_score - current_total
                if random.random() < math.exp(delta / t):
                    current_total = new_score
                else:
                    current_xy[i, 0] = old_x
                    current_xy[i, 1] = old_y
                    current_r[i] = old_r
            
        # Cool down
        t *= cooling
        # Slowly reduce step size
        step_size *= 0.99995
        
    return best_xy, best_r, best_total, best_valid

def _contact_refinement(xy, r, deadline):
    """
    Refine packing using contact graph information.
    For each circle, try to move it slightly to increase its radius or move it away from constraints.
    """
    n = len(xy)
    current_xy = xy.copy()
    current_r = r.copy()
    current_total, current_valid = _eval_packing(current_xy, current_r)
    
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    best_total = current_total
    best_valid = current_valid
    
    if not best_valid:
        return best_xy, best_r, best_total, best_valid
        
    # Identify contacts
    def get_contacts(i):
        contacts = []
        x, y, ri = current_xy[i, 0], current_xy[i, 1], current_r[i]
        # Wall contacts
        if abs(x - ri) < 1e-6: contacts.append(('wall', 'left'))
        if abs(1-x - ri) < 1e-6: contacts.append(('wall', 'right'))
        if abs(y - ri) < 1e-6: contacts.append(('wall', 'bottom'))
        if abs(1-y - ri) < 1e-6: contacts.append(('wall', 'top'))
        
        for j in range(n):
            if i == j: continue
            dx = x - current_xy[j, 0]
            dy = y - current_xy[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            if abs(dist - (ri + current_r[j])) < 1e-6:
                contacts.append(('circle', j))
        return contacts

    # Local search on contact graph
    # We try to "push" circles apart or expand them
    for iteration in range(100):
        if time.time() > deadline: break
        
        improved = False
        for i in range(n):
            contacts = get_contacts(i)
            if not contacts:
                # If no contacts, we can potentially expand
                # Try expanding radius slightly
                old_r = current_r[i]
                new_r = old_r * 1.01
                # Check if valid
                current_r[i] = new_r
                total, valid = _eval_packing(current_xy, current_r)
                if valid and total > best_total:
                    best_total = total
                    best_valid = True
                    best_xy = current_xy.copy()
                    best_r = current_r.copy()
                    improved = True
                else:
                    current_r[i] = old_r
                continue
            
            # If contacts, try moving in direction away from contacts
            # Calculate force vector
            fx, fy = 0.0, 0.0
            for c in contacts:
                if c[0] == 'wall':
                    if c[1] == 'left': fx += 1.0
                    if c[1] == 'right': fx -= 1.0
                    if c[1] == 'bottom': fy += 1.0
                    if c[1] == 'top': fy -= 1.0
                else:
                    j = c[1]
                    dx = current_xy[i, 0] - current_xy[j, 0]
                    dy = current_xy[i, 1] - current_xy[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist > 1e-9:
                        fx += dx / dist
                        fy += dy / dist
            
            norm = math.sqrt(fx*fx + fy*fy)
            if norm < 1e-9: continue
            
            # Move slightly in force direction
            move_mag = 0.005
            old_x, old_y = current_xy[i, 0], current_xy[i, 1]
            new_x = old_x + (fx / norm) * move_mag
            new_y = old_y + (fy / norm) * move_mag
            
            # Keep inside bounds
            new_x = np.clip(new_x, current_r[i], 1.0 - current_r[i])
            new_y = np.clip(new_y, current_r[i], 1.0 - current_r[i])
            
            current_xy[i, 0] = new_x
            current_xy[i, 1] = new_y
            
            # Try to expand radius after moving
            total, valid = _eval_packing(current_xy, current_r)
            if valid:
                # Try to expand
                max_expand = 0.01
                for _ in range(10):
                    old_r = current_r[i]
                    new_r = old_r + max_expand/10
                    current_r[i] = new_r
                    t2, v2 = _eval_packing(current_xy, current_r)
                    if v2 and t2 > total:
                        total = t2
                    else:
                        current_r[i] = old_r
                        break
                
                if total > best_total:
                    best_total = total
                    best_valid = True
                    best_xy = current_xy.copy()
                    best_r = current_r.copy()
                    improved = True
            else:
                # Revert if invalid
                current_xy[i, 0] = old_x
                current_xy[i, 1] = old_y
        
        if not improved:
            break
            
    return best_xy, best_r, best_total, best_valid

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Initialize with a few different grid configurations
    inits = []
    for side in [5, 6]:
        pos = _grid_init(side)
        inits.append(pos)
        
    # Also try a random initialization
    random_xy = np.random.rand(N, 2)
    random_r = np.full(N, 0.05)
    inits.append(np.column_stack([random_xy, random_r]))
    
    for init in inits:
        if time.time() > deadline - 20.0: break
        
        xy = init[:, :2].copy()
        r = init[:, 2].copy()
        
        # Ensure initial validity
        total, valid = _eval_packing(xy, r)
        if not valid:
            # Shrink radii until valid
            scale = 0.9
            for _ in range(10):
                r_new = r * scale
                total, valid = _eval_packing(xy, r_new)
                if valid:
                    r = r_new
                    break
                scale *= 0.9
            if not valid:
                continue
        
        # Run Simulated Annealing
        sa_xy, sa_r, sa_total, sa_valid = _sa_packing(xy, r, deadline - 10.0)
        
        if sa_valid and sa_total > best_total:
            best_total = sa_total
            best_valid = True
            best_xy = sa_xy
            best_r = sa_r
            
        # Run Contact Refinement on the best SA result
        if best_valid and time.time() < deadline - 5.0:
            ref_xy, ref_r, ref_total, ref_valid = _contact_refinement(best_xy, best_r, deadline - 2.0)
            if ref_valid and ref_total > best_total:
                best_total = ref_total
                best_valid = True
                best_xy = ref_xy
                best_r = ref_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

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
    
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
            
    return result
# EVOLVE-BLOCK-END
