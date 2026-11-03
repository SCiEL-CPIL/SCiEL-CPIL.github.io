"""Packing 300 equal circles in a unit circle via Simulated Annealing with spatial hashing."""
import numpy as np
import time
import math

N = 300
TIME_LIMIT = 115.0  # safety margin before 150s kill


def _hex_init(n=N):
    """Dense hexagonal lattice clipped to unit disk, keep n closest to center."""
    spacing = 0.095
    k = int(1.15 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * math.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        spacing = 0.08
        k = int(1.2 / spacing) + 2
        xs, ys = [], []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * math.sqrt(3.0) / 2.0
                xs.append(x)
                ys.append(y)
        pts = np.column_stack([xs, ys])
        r = np.hypot(pts[:, 0], pts[:, 1])
        inside = r <= 1.0
        pts = pts[inside]
        r = r[inside]
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _score(pts):
    """Achieved common radius."""
    n = len(pts)
    # Use spatial hash for efficiency, but for final score use full O(n^2) for accuracy
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    min_pair = math.sqrt(np.min(d2)) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)


def _spatial_hash(pts, cell_size):
    """Build spatial hash: dict mapping cell -> list of point indices."""
    h = {}
    n = len(pts)
    for i in range(n):
        cx = int(math.floor(pts[i, 0] / cell_size))
        cy = int(math.floor(pts[i, 1] / cell_size))
        key = (cx, cy)
        if key not in h:
            h[key] = []
        h[key].append(i)
    return h


def _neighbors_in_hash(h, pt, cell_size, search_radius):
    """Get indices of points within search_radius of pt using spatial hash."""
    cx0 = int(math.floor((pt[0] - search_radius) / cell_size))
    cy0 = int(math.floor((pt[1] - search_radius) / cell_size))
    cx1 = int(math.floor((pt[0] + search_radius) / cell_size))
    cy1 = int(math.floor((pt[1] + search_radius) / cell_size))
    result = []
    for cx in range(cx0, cx1 + 1):
        for cy in range(cy0, cy1 + 1):
            key = (cx, cy)
            if key in h:
                result.extend(h[key])
    return result


def _resolve_violations(pts, r, max_iter=50):
    """
    Resolve all overlap and boundary violations for a given radius r.
    Returns (resolved_pts, success).
    Uses iterative pairwise push + boundary clamp.
    """
    n = len(pts)
    pts = pts.copy()
    min_dist = 2.0 * r
    cell_size = max(min_dist, 0.01)
    
    for _ in range(max_iter):
        # Build hash
        h = _spatial_hash(pts, cell_size)
        
        any_move = False
        
        # Boundary check first
        rad = np.hypot(pts[:, 0], pts[:, 1])
        max_allowed = 1.0 - r
        out = rad > max_allowed + 1e-12
        if np.any(out):
            for i in np.where(out)[0]:
                if rad[i] > 1e-10:
                    pts[i] = pts[i] * (max_allowed / rad[i])
                else:
                    pts[i] = np.array([max_allowed, 0.0])
                any_move = True
        
        # Pairwise overlap resolution
        for i in range(n):
            # Find neighbors
            nbrs = _neighbors_in_hash(h, pts[i], cell_size, min_dist + 0.01)
            for j in nbrs:
                if j <= i:
                    continue
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                d2 = dx * dx + dy * dy
                if d2 < min_dist * min_dist:
                    d = math.sqrt(max(d2, 1e-15))
                    # Push apart
                    overlap = min_dist - d
                    if overlap > 0:
                        push = overlap / 2.0 + 1e-8
                        ux = dx / d
                        uy = dy / d
                        pts[i, 0] += ux * push
                        pts[i, 1] += uy * push
                        pts[j, 0] -= ux * push
                        pts[j, 1] -= uy * push
                        any_move = True
        
        if not any_move:
            return pts, True
    
    # Final check
    return pts, False


def _sa_optimize(pts, r, start_time, time_limit):
    """
    Simulated Annealing to find feasible packing with radius r.
    Returns (best_pts, best_r) where best_r is the achieved radius.
    """
    n = len(pts)
    best_pts = pts.copy()
    best_achieved = 0.0
    
    # Try to find a feasible packing at radius r using SA
    # State: positions of all n points
    # Energy: sum of overlap violations + boundary violations
    
    current = pts.copy()
    
    # Initial: resolve violations greedily
    current, ok = _resolve_violations(current, r, max_iter=100)
    if ok:
        achieved = _score(current)
        if achieved >= r - 1e-9:
            best_achieved = achieved
            best_pts = current.copy()
            return best_pts, best_achieved
    
    # SA loop
    T_start = 0.02
    T_end = 0.0005
    n_steps = 200000  # will be limited by time
    step_base = 0.03
    
    # Adaptive: use time budget
    t0 = time.time()
    step = 0
    no_improve = 0
    max_no_improve = 5000
    
    while True:
        elapsed = time.time() - t0
        if elapsed > time_limit:
            break
        if no_improve > max_no_improve:
            break
        if step >= n_steps:
            break
        
        # Temperature schedule (exponential cooling with restarts)
        progress = min(1.0, elapsed / time_limit)
        T = T_start * math.exp(-5.0 * progress) + T_end
        
        # Pick random point to perturb
        i = np.random.randint(n)
        
        # Perturb: random displacement
        angle = np.random.uniform(0, 2 * math.pi)
        mag = np.random.exponential(step_base * T / T_start + 0.005)
        dx = math.cos(angle) * mag
        dy = math.sin(angle) * mag
        
        new_pt = current[i].copy()
        new_pt[0] += dx
        new_pt[1] += dy
        
        # Clamp to boundary
        rad = math.hypot(new_pt[0], new_pt[1])
        max_allowed = 1.0 - r
        if rad > max_allowed:
            if rad > 1e-10:
                new_pt = new_pt * (max_allowed / rad)
            else:
                new_pt = np.array([max_allowed, 0.0])
        
        # Compute local energy change (only neighbors of i)
        # Old energy contribution of i
        old_e = 0.0
        cell_size = max(2.0 * r, 0.01)
        h = _spatial_hash(current, cell_size)
        nbrs = _neighbors_in_hash(h, current[i], cell_size, 2.0 * r + 0.05)
        for j in nbrs:
            if j == i:
                continue
            d = math.hypot(current[i, 0] - current[j, 0], current[i, 1] - current[j, 1])
            if d < 2.0 * r:
                old_e += (2.0 * r - d) ** 2
        # Boundary violation
        rad_i = math.hypot(current[i, 0], current[i, 1])
        if rad_i > max_allowed:
            old_e += (rad_i - max_allowed) ** 2
        
        # New energy contribution
        new_e = 0.0
        for j in nbrs:
            if j == i:
                continue
            d = math.hypot(new_pt[0] - current[j, 0], new_pt[1] - current[j, 1])
            if d < 2.0 * r:
                new_e += (2.0 * r - d) ** 2
        rad_new = math.hypot(new_pt[0], new_pt[1])
        if rad_new > max_allowed:
            new_e += (rad_new - max_allowed) ** 2
        
        delta_e = new_e - old_e
        
        # Accept?
        if delta_e < 0 or np.random.random() < math.exp(-delta_e / max(T, 1e-10)):
            current[i] = new_pt
            no_improve += 1
        else:
            no_improve += 1
        
        # Periodically check if we have a feasible solution
        if step % 500 == 0:
            # Try to resolve and check
            test_pts, test_ok = _resolve_violations(current, r, max_iter=20)
            if test_ok:
                test_score = _score(test_pts)
                if test_score >= r - 1e-9:
                    if test_score > best_achieved:
                        best_achieved = test_score
                        best_pts = test_pts.copy()
                        no_improve = 0
                elif test_score > best_achieved:
                    best_achieved = test_score
                    best_pts = test_pts.copy()
        
        # Restart if stuck
        if no_improve > max_no_improve:
            # Random restart from best
            current = best_pts.copy()
            # Add small random noise
            noise = np.random.normal(0, 0.01, size=(n, 2))
            current += noise
            rad = np.hypot(current[:, 0], current[:, 1])
            max_allowed = 1.0 - r
            out = rad > max_allowed
            if np.any(out):
                current[out] = current[out] * (max_allowed / rad[out, None])[:, None]
            no_improve = 0
        
        step += 1
    
    return best_pts, best_achieved


def _binary_search_radius(pts, start_time, time_limit):
    """
    Binary search for the maximum feasible radius using SA feasibility checks.
    """
    # Lower bound: current score
    r_low = _score(pts)
    # Upper bound: theoretical max ~ 1/sqrt(300*2/sqrt(3)) approx, use 0.06
    r_high = 0.058
    
    best_pts = pts.copy()
    best_r = r_low
    
    # We'll do a few binary search steps, each with a time-limited SA
    n_bs_steps = 5
    time_per_sa = (time_limit - (time.time() - start_time)) / (n_bs_steps + 2)
    
    for bs in range(n_bs_steps):
        if time.time() - start_time > time_limit - 5:
            break
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find feasible packing at r_mid
        sa_pts, sa_r = _sa_optimize(pts, r_mid, start_time, time.time() - start_time + time_per_sa)
        
        if sa_r >= r_mid - 1e-9:
            # Feasible: raise lower bound
            r_low = r_mid
            if sa_r > best_r:
                best_r = sa_r
                best_pts = sa_pts.copy()
        else:
            # Not feasible: lower upper bound
            r_high = r_mid
        
        # Update pts for next iteration
        pts = best_pts.copy()
    
    return best_pts, best_r


def _polish(pts, r, start_time, time_limit):
    """
    Final polishing: use gradient-free local moves to improve the packing.
    Try to increase the minimum distance.
    """
    n = len(pts)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Local search: try small random moves that improve the min constraint
    t0 = time.time()
    step = 0
    max_steps = 100000
    
    while True:
        elapsed = time.time() - t0
        if elapsed > time_limit:
            break
        if step >= max_steps:
            break
        
        i = np.random.randint(n)
        angle = np.random.uniform(0, 2 * math.pi)
        mag = np.random.exponential(0.005)
        dx = math.cos(angle) * mag
        dy = math.sin(angle) * mag
        
        new_pt = best_pts[i].copy()
        new_pt[0] += dx
        new_pt[1] += dy
        
        # Check boundary
        rad = math.hypot(new_pt[0], new_pt[1])
        if rad >= 1.0:
            step += 1
            continue
        
        # Check if this move improves the minimum constraint
        # Compute new min distance involving i
        old_min = best_score
        new_pts = best_pts.copy()
        new_pts[i] = new_pt
        
        # Only check neighbors for efficiency
        cell_size = max(2.0 * old_min, 0.01)
        h = _spatial_hash(new_pts, cell_size)
        nbrs = _neighbors_in_hash(h, new_pts[i], cell_size, 2.0 * old_min + 0.05)
        
        min_d = 1.0  # large
        for j in nbrs:
            if j == i:
                continue
            d = math.hypot(new_pts[i, 0] - new_pts[j, 0], new_pts[i, 1] - new_pts[j, 1])
            min_d = min(min_d, d)
        
        new_score = min(min_d / 2.0, 1.0 - rad)
        
        if new_score > old_min + 1e-12:
            best_pts = new_pts
            best_score = new_score
        
        step += 1
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    
    # Generate diverse initializations
    np.random.seed(42)
    
    # Multiple hex inits with slight variations
    inits = []
    for seed in range(4):
        np.random.seed(seed * 123 + 7)
        pts = _hex_init(N)
        # Add small random perturbation for diversity
        noise = np.random.normal(0, 0.003, size=(N, 2))
        pts += noise
        # Clamp
        rad = np.hypot(pts[:, 0], pts[:, 1])
        out = rad > 1.0
        if np.any(out):
            pts[out] = pts[out] / rad[out, None]
        inits.append(pts)
    
    best_pts = inits[0].copy()
    best_score = _score(best_pts)
    
    # For each init, run SA-based optimization
    n_inits = len(inits)
    time_per_init = (TIME_LIMIT - (time.time() - start_time)) / (n_inits + 1)
    
    for idx in range(n_inits):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        pts = inits[idx].copy()
        init_score = _score(pts)
        
        # Phase 1: SA with binary search for radius
        # Start with a modest radius and try to improve
        r_start = init_score
        r_target = min(r_start + 0.002, 0.055)
        
        # Try SA at target radius
        sa_time = min(time_per_init * 0.6, 30.0)
        sa_pts, sa_r = _sa_optimize(pts, r_target, start_time, time.time() - start_time + sa_time)
        
        if sa_r > init_score:
            best_pts = sa_pts.copy()
            best_score = sa_r
            pts = sa_pts.copy()
        
        # Phase 2: Polish
        if time.time() - start_time < TIME_LIMIT - 5:
            polish_time = min(time_per_init * 0.3, 15.0)
            pol_pts, pol_score = _polish(pts, best_score, start_time, polish_time)
            if pol_score > best_score:
                best_pts = pol_pts.copy()
                best_score = pol_score
    
    # Final safety
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (N, 2):
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]
    
    return best_pts
