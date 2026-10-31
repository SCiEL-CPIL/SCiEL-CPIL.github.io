"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) using softplus."""
    return _softplus(v, beta)

def _objective(x, penalty_weight=100.0, beta=50.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _objective_smooth(x, penalty_weight=100.0, beta=30.0):
    """
    Objective with softplus-based smooth penalties.
    Uses a soft-min (log-sum-exp) surrogate for the distance to the boundary
    and a smooth max(0, required - dist) for overlaps, which is more
    differentiable than the squared-penalty objective.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)

    # Smooth boundary: soft-min of the four wall distances
    d = np.stack([pos[:, 0], 1.0 - pos[:, 0], pos[:, 1], 1.0 - pos[:, 1]], axis=1)
    neg_d = -d
    m = np.max(neg_d, axis=1, keepdims=True)
    neg_d_clipped = np.clip(neg_d, -50.0, 50.0)
    soft_max_neg = m + np.log(np.sum(np.exp(neg_d_clipped - m), axis=1, keepdims=True))
    soft_min_d = -soft_max_neg[:, 0]
    boundary_pen = np.sum(_soft_max(radii - soft_min_d, beta))

    # Smooth overlap: for each pair, required = r_i + r_j, dist = ||p_i - p_j||
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol_raw = min_dist - dist
    overlap_pen = np.sum(_soft_max(viol_raw[mask], beta))

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_best_fit():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)
        
        for _ in range(5000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
        
    return pos, radii

def _initial_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        for _ in range(100):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _initial_corner_biased(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[i] = 0.18
    
    for i in range(4, N):
        if rng.random() < 0.4:
            edge = rng.randint(4)
            t = rng.uniform(0.1, 0.9)
            if edge == 0:
                x, y = t, 0.08
            elif edge == 1:
                x, y = t, 0.92
            elif edge == 2:
                x, y = 0.08, t
            else:
                x, y = 0.92, t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
        
        if max_r < 0.01:
            for attempt in range(200):
                x2 = rng.uniform(0.05, 0.95)
                y2 = rng.uniform(0.05, 0.95)
                max_r2 = min(x2, 1.0 - x2, y2, 1.0 - y2)
                for j in range(i):
                    dx = x2 - pos[j, 0]
                    dy = y2 - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r2 = min(max_r2, dist - radii[j])
                if max_r2 > max_r:
                    max_r = max_r2
                    x, y = x2, y2
        
        pos[i] = [x, y]
        radii[i] = max(max_r, 0.01) * 0.95
        
    return pos, radii

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _incremental_energy(pos, radii, changed_idx, penalty_weight=50.0):
    """
    Fast incremental energy computation for a single changed circle.
    Only recomputes terms involving the changed circle.
    """
    n = len(radii)
    x, y, r = pos[changed_idx, 0], pos[changed_idx, 1], radii[changed_idx]
    
    # Boundary penalty for changed circle
    bv = np.maximum(0.0, r - x) + np.maximum(0.0, x - (1.0 - r)) + \
         np.maximum(0.0, r - y) + np.maximum(0.0, y - (1.0 - r))
    boundary_pen = bv * bv
    
    # Overlap penalty with all other circles
    overlap_pen = 0.0
    for j in range(n):
        if j == changed_idx:
            continue
        dx = x - pos[j, 0]
        dy = y - pos[j, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = r + radii[j]
        viol = max(0.0, min_dist * min_dist - dist_sq)
        overlap_pen += viol * viol
    
    return penalty_weight * (boundary_pen + overlap_pen)

def _full_energy(pos, radii, penalty_weight=50.0):
    """Full energy computation (same as _objective but returns penalty only)."""
    n = len(radii)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return penalty_weight * (boundary_pen + overlap_pen)

def _gradient_descent(pos, radii, n_steps=5000, lr_init=0.005, lr_decay=0.999, seed=0, deadline=None):
    """
    Vectorized gradient descent with pairwise force computation.
    Uses repulsive forces for overlaps and attractive forces for boundary violations.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    lr = lr_init
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces vectorized
        forces = np.zeros((n, 2))
        r_forces = np.zeros(n)
        
        # Boundary forces: push circles away from walls
        # Left wall
        left_viol = np.maximum(0.0, radii - pos[:, 0])
        forces[:, 0] += left_viol * 10.0
        r_forces += left_viol * 5.0
        # Right wall
        right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
        forces[:, 0] -= right_viol * 10.0
        r_forces += right_viol * 5.0
        # Bottom wall
        bottom_viol = np.maximum(0.0, radii - pos[:, 1])
        forces[:, 1] += bottom_viol * 10.0
        r_forces += bottom_viol * 5.0
        # Top wall
        top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
        forces[:, 1] -= top_viol * 10.0
        r_forces += top_viol * 5.0
        
        # Overlap forces: repel overlapping circles (vectorized)
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
        dist = np.sqrt(dist_sq + 1e-12)
        overlap = np.maximum(0.0, sum_r - dist)
        
        # Force magnitude per pair
        force_mag = overlap * 5.0 / (dist + 1e-12)
        
        # Accumulate forces using matrix operations
        # For each i, force from j is force_mag[i,j] * (dx[i,j], dy[i,j])
        fx_matrix = force_mag * dx
        fy_matrix = force_mag * dy
        
        # Sum over j for each i (excluding self)
        forces[:, 0] += np.sum(fx_matrix, axis=1)
        forces[:, 1] += np.sum(fy_matrix, axis=1)
        
        # Radius forces from overlaps
        r_forces += 2.0 * np.sum(overlap, axis=1)
        
        # Update positions and radii
        pos[:, 0] += lr * forces[:, 0]
        pos[:, 1] += lr * forces[:, 1]
        radii -= lr * r_forces
        
        # Clamp
        pos[:, 0] = np.clip(pos[:, 0], 0.001, 0.999)
        pos[:, 1] = np.clip(pos[:, 1], 0.001, 0.999)
        radii = np.clip(radii, 0.001, 0.4)
        
        # Adaptive learning rate
        total_force = np.sum(forces ** 2) + np.sum(r_forces ** 2)
        if total_force < 1e-6:
            lr *= 0.9
        else:
            lr *= lr_decay
        
        # Periodically check validity and update best
        if step % 50 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    # Final check
    pos_pp, radii_pp = _post_process(pos, radii)
    if _is_valid(pos_pp, radii_pp):
        score = np.sum(radii_pp)
        if score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _slsqp_constraints(pos, radii):
    """Generate constraint functions for SLSQP with hard non-overlap constraints."""
    n = len(radii)
    constraints = []
    
    # Boundary constraints: x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*n+i]})
    
    # Non-overlap constraints: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 - (x[2*n+i] + x[2*n+j])**2
            })
    
    return constraints

def _lp_radius_optimize(pos):
    """
    Solve for maximum sum of radii given fixed positions using Linear Programming.
    Maximize sum(r_i) subject to:
      r_i + r_j <= dist(i,j) for all i<j
      r_i <= min(x_i, 1-x_i, y_i, 1-y_i) for all i
      r_i >= 0
    """
    n = len(pos)
    c = -np.ones(n)  # minimize -sum(r)
    
    A_rows = []
    b_vals = []
    
    # Pairwise distance constraints: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_vals.append(d)
    
    # Wall constraints: r_i <= wall_dist_i
    for i in range(n):
        w = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        row = np.zeros(n)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(max(w, 0.0))
    
    if not A_rows:
        return np.zeros(n)
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 0.5)] * n, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    
    return np.zeros(n)

def _jitter_and_expand(pos, radii, deadline, seed=0):
    """
    Jitter-and-expand heuristic: iteratively perturb circle positions and
    re-solve LP for maximal radii. Decouples topological search from geometric
    optimization. The LP gives exact maximal radii for each configuration.
    
    Strategy:
    1. Solve LP to get current maximal radii
    2. Identify "stuck" circles (at LP bound, i.e., constrained by neighbors/walls)
    3. Jitter stuck circles in random directions
    4. Re-solve LP
    5. Accept if sum of radii improved
    6. Repeat until deadline
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Initial LP solve
    lp_radii = _lp_radius_optimize(pos)
    if np.sum(lp_radii) > best_score:
        best_score = np.sum(lp_radii)
        best_radii = lp_radii.copy()
        radii = lp_radii.copy()
    
    # Precompute pairwise distances for speed
    def _get_pairwise_dists(p):
        dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
        dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
        return np.sqrt(dx*dx + dy*dy)
    
    # Jitter parameters
    jitter_scale = 0.015
    n_stuck_threshold = 0.001  # if LP radius < wall_dist - threshold, circle is "free"
    
    iteration = 0
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        iteration += 1
        
        # Solve LP for current positions
        lp_radii = _lp_radius_optimize(pos)
        current_score = np.sum(lp_radii)
        
        if current_score > best_score + 1e-10:
            best_score = current_score
            best_pos = pos.copy()
            best_radii = lp_radii.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Restart with larger jitter
            jitter_scale *= 1.5
            if jitter_scale > 0.05:
                jitter_scale = 0.015
            no_improve_count = 0
            # Randomize positions slightly
            pos = best_pos + rng.normal(0, 0.005, (n, 2))
            pos = np.clip(pos, 0.01, 0.99)
            continue
        
        # Identify stuck circles: those where LP radius is close to their max possible
        # A circle is "stuck" if its LP radius is less than its wall clearance
        # (meaning it's constrained by neighbors rather than walls)
        wall_clearance = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                                    np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
        stuck_mask = lp_radii < (wall_clearance - n_stuck_threshold)
        
        # Also mark circles that are very small as stuck (they need room)
        stuck_mask |= (lp_radii < 0.03)
        
        # If all circles are stuck, jitter all; otherwise jitter a random subset of stuck
        if not np.any(stuck_mask):
            # All circles are at wall limit - try jittering a random one
            stuck_mask[rng.randint(n)] = True
        
        # Select 1-3 stuck circles to jitter
        stuck_indices = np.where(stuck_mask)[0]
        n_jitter = min(rng.randint(1, 4), len(stuck_indices))
        jitter_indices = rng.choice(stuck_indices, size=n_jitter, replace=False)
        
        # Apply jitter: random angular perturbation
        for idx in jitter_indices:
            angle = rng.uniform(0, 2 * np.pi)
            magnitude = jitter_scale * rng.uniform(0.5, 1.5)
            pos[idx, 0] += magnitude * np.cos(angle)
            pos[idx, 1] += magnitude * np.sin(angle)
        
        # Clamp to valid range
        pos = np.clip(pos, 0.01, 0.99)
        
        # Also try a "swap" perturbation: exchange positions of two stuck circles
        if len(stuck_indices) >= 2 and rng.random() < 0.2:
            i, j = rng.choice(stuck_indices, size=2, replace=False)
            pos[i], pos[j] = pos[j].copy(), pos[i].copy()
    
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """High-precision polishing: jitter-and-expand + SLSQP + LP radius optimization."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))
        
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # First: LP-based radius optimization to get exact maximal radii
    if time.time() < deadline - 10:
        try:
            lp_radii = _lp_radius_optimize(best_pos)
            pos_pp, radii_pp = _post_process(best_pos, lp_radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Jitter-and-expand: the main optimization driver
    if time.time() < deadline - 15:
        try:
            pos_je, rad_je = _jitter_and_expand(best_pos, best_radii, deadline - 15, seed=777)
            pos_pp, radii_pp = _post_process(pos_je, rad_je)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # SLSQP with hard constraints for local refinement
    if time.time() < deadline - 10:
        try:
            x_cur = np.concatenate([best_pos.flatten(), best_radii])
            cons = _slsqp_constraints(best_pos, best_radii)
            result = minimize(lambda x: -np.sum(np.maximum(x[2*N:], 1e-6)), x_cur, method='SLSQP',
                              bounds=bounds, constraints=cons,
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success or result.fun < -np.sum(np.maximum(x_cur[2*N:], 1e-6)) + 1e-10:
                x_new = result.x
                pos_slq = x_new[:2*N].reshape(N, 2)
                rad_slq = np.maximum(x_new[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_slq, rad_slq)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Another jitter-and-expand from the SLSQP result
    if time.time() < deadline - 10:
        try:
            pos_je2, rad_je2 = _jitter_and_expand(best_pos, best_radii, deadline - 10, seed=888)
            pos_pp, radii_pp = _post_process(pos_je2, rad_je2)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Gradient descent for fine-tuning
    if time.time() < deadline - 8:
        try:
            pos_gd, rad_gd = _gradient_descent(best_pos, best_radii, n_steps=3000, lr_init=0.004, lr_decay=0.9995, seed=42, deadline=deadline - 6)
            pos_pp, radii_pp = _post_process(pos_gd, rad_gd)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final LP pass
    if time.time() < deadline - 4:
        try:
            lp_radii_final = _lp_radius_optimize(best_pos)
            pos_pp, radii_pp = _post_process(best_pos, lp_radii_final)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    return best_pos, best_radii

def _simulated_annealing(pos, radii, n_steps=20000, t_init=0.01, t_min=1e-6, cooling=0.9997, seed=0, deadline=None):
    """Simulated annealing with efficient incremental energy evaluation."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    t = t_init
    step = 0
    while step < n_steps:
        if deadline is not None and time.time() > deadline:
            break
        
        # Choose a random circle to perturb
        i = rng.randint(n)
        
        # Save old state
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        
        # Generate perturbation
        scale = 0.01 * t / t_init + 0.001
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        dr = rng.normal(0, scale * 0.5)
        
        new_x = np.clip(old_x + dx, 0.001, 0.999)
        new_y = np.clip(old_y + dy, 0.001, 0.999)
        new_r = np.clip(old_r + dr, 0.001, 0.4)
        
        # Compute old contribution (before modifying)
        old_bv = np.maximum(0.0, old_r - old_x) + np.maximum(0.0, old_x - (1.0 - old_r)) + \
                 np.maximum(0.0, old_r - old_y) + np.maximum(0.0, old_y - (1.0 - old_r))
        old_overlap = 0.0
        for j in range(n):
            if j == i:
                continue
            dx2 = old_x - pos[j, 0]
            dy2 = old_y - pos[j, 1]
            dist_sq2 = dx2 * dx2 + dy2 * dy2
            min_dist2 = old_r + radii[j]
            viol2 = max(0.0, min_dist2 * min_dist2 - dist_sq2)
            old_overlap += viol2 * viol2
        old_contrib = 50.0 * (old_bv * old_bv + old_overlap)
        
        # Compute new contribution (without modifying state)
        new_bv = np.maximum(0.0, new_r - new_x) + np.maximum(0.0, new_x - (1.0 - new_r)) + \
                 np.maximum(0.0, new_r - new_y) + np.maximum(0.0, new_y - (1.0 - new_r))
        new_overlap = 0.0
        for j in range(n):
            if j == i:
                continue
            dx3 = new_x - pos[j, 0]
            dy3 = new_y - pos[j, 1]
            dist_sq3 = dx3 * dx3 + dy3 * dy3
            min_dist3 = new_r + radii[j]
            viol3 = max(0.0, min_dist3 * min_dist3 - dist_sq3)
            new_overlap += viol3 * viol3
        new_contrib = 50.0 * (new_bv * new_bv + new_overlap)
        
        delta_energy = new_contrib - old_contrib
        delta_score = new_r - old_r
        delta_total = delta_energy - delta_score  # We minimize energy - score
        
        if delta_total < 0 or rng.random() < np.exp(-delta_total / max(t, 1e-10)):
            # Accept: modify state
            pos[i, 0] = new_x
            pos[i, 1] = new_y
            radii[i] = new_r
        # else: reject, state unchanged
        
        # Cool down
        t *= cooling
        t = max(t, t_min)
        step += 1
        
        # Periodically check validity
        if step % 100 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    # Final check
    pos_pp, radii_pp = _post_process(pos, radii)
    if _is_valid(pos_pp, radii_pp):
        score = np.sum(radii_pp)
        if score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    for i in range(3):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
    
    # For each initialization, run a quick LP + jitter-and-expand cycle
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        # Quick LP to get maximal radii for this initial configuration
        try:
            lp_rad = _lp_radius_optimize(pos_init)
            pos_pp, radii_pp = _post_process(pos_init, lp_rad)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
        
        # Short jitter-and-expand from this initialization
        if time.time() < deadline - 25:
            try:
                pos_je, rad_je = _jitter_and_expand(pos_init, rad_init, deadline - 25, seed=idx * 100 + 1)
                pos_pp, radii_pp = _post_process(pos_je, rad_je)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Main optimization: longer jitter-and-expand from the best configuration
    if time.time() < deadline - 20:
        try:
            pos_je, rad_je = _jitter_and_expand(best_pos, best_radii, deadline - 20, seed=42)
            pos_pp, radii_pp = _post_process(pos_je, rad_je)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Polish: SLSQP + more jitter-and-expand + LP
    if time.time() < deadline - 8:
        try:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # LP radius optimization on best positions
    if time.time() < deadline - 4:
        try:
            lp_rad = _lp_radius_optimize(best_pos)
            pos_pp, radii_pp = _post_process(best_pos, lp_rad)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Second jitter-and-expand with different seed
    if time.time() < deadline - 8:
        try:
            pos_je2, rad_je2 = _jitter_and_expand(best_pos, best_radii, deadline - 8, seed=999)
            pos_pp, radii_pp = _post_process(pos_je2, rad_je2)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final LP
    if time.time() < deadline - 3:
        try:
            lp_rad = _lp_radius_optimize(best_pos)
            pos_pp, radii_pp = _post_process(best_pos, lp_rad)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
