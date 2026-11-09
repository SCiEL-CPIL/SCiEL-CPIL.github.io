"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.spatial.distance import cdist

DEADLINE = 140.0  # safety margin before 150s kill


def _score(pts):
    """Compute the radius: min of half min-pair-dist and min distance to boundary."""
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = D.min() / 2.0
    min_bound = np.min(0.5 - np.abs(pts))
    return min(min_pair, min_bound)


def _score_fast(pts):
    """Faster score using squared distances for comparison."""
    dx = pts[:, 0][:, None] - pts[:, 0][None, :]
    dy = pts[:, 1][:, None] - pts[:, 1][None, :]
    dists_sq = dx * dx + dy * dy
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_sq = dists_sq.min()
    min_bound = np.min(0.5 - np.abs(pts))
    return min(np.sqrt(min_pair_sq) / 2.0, min_bound)


def _make_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _make_honeycomb():
    """Honeycomb-like initial config."""
    pts = np.zeros((N, 2))
    spacing = 0.12
    row = 0
    idx = 0
    y = -0.45
    while idx < N and y < 0.5:
        x_offset = (spacing * 0.5) if (row % 2 == 1) else 0.0
        x = -0.45 + x_offset
        col = 0
        while x < 0.5 and idx < N:
            pts[idx, 0] = x
            pts[idx, 1] = y
            idx += 1
            x += spacing
            col += 1
        y += spacing * 0.866
        row += 1
    return pts


def _make_structured():
    """Structured 8x8 grid with 14 points removed."""
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    remove = set()
    remove.update([(0,0),(0,7),(7,0),(7,7)])
    remove.update([(0,1),(0,6),(7,1),(7,6),(1,0),(6,0),(1,7),(6,7),(3,0),(4,7)])
    
    kept = []
    for i in range(m):
        for j in range(m):
            if (i,j) not in remove:
                kept.append(grid[i*m + j])
    
    centers = np.array(kept)
    if len(centers) != N:
        centers = grid[:N]
    
    centers += np.random.normal(0, 0.001, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _make_random():
    return np.random.uniform(-0.4, 0.4, (N, 2))


def _anneal(pts, time_budget, rng):
    """Simulated annealing to maximize the minimum radius."""
    pts = pts.copy()
    current_score = _score_fast(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    t_start = time.time()
    t = 0.05
    cooling = 0.9995
    step_size = 0.05
    
    while time.time() - t_start < time_budget:
        i = rng.randint(0, N - 1)
        dx = (rng.random() - 0.5) * 2 * step_size
        dy = (rng.random() - 0.5) * 2 * step_size
        old_x, old_y = pts[i, 0], pts[i, 1]
        pts[i, 0] = old_x + dx
        pts[i, 1] = old_y + dy
        pts[i, 0] = np.clip(pts[i, 0], -0.5, 0.5)
        pts[i, 1] = np.clip(pts[i, 1], -0.5, 0.5)
        
        new_score = _score_fast(pts)
        
        if new_score >= current_score or rng.random() < np.exp((new_score - current_score) / t):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        else:
            pts[i, 0] = old_x
            pts[i, 1] = old_y
        
        t *= cooling
        step_size *= 0.9999
        if step_size < 0.001:
            step_size = 0.001
    
    return best_pts, best_score


def _constraint_solve(pts, time_budget, rng):
    """
    Constraint-satisfaction approach: directly maximize minimum distance
    by treating pairwise and wall constraints as hard inequalities.
    Uses iterative pushing of violating constraints.
    """
    pts = pts.copy()
    current_score = _score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    t_start = time.time()
    push_strength = 0.002
    n_iter = 0
    
    while time.time() - t_start < time_budget:
        # Compute all pairwise distances
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        
        # Find minimum pair distance
        min_pair_dist = D.min()
        min_bound = np.min(0.5 - np.abs(pts))
        current_r = min(min_pair_dist / 2.0, min_bound)
        
        if current_r > best_score:
            best_score = current_r
            best_pts = pts.copy()
        
        # Find the most violated constraint
        # Closest pair
        idx_min = np.unravel_index(np.argmin(D), D.shape)
        i, j = idx_min[0], idx_min[1]
        
        # Check if pair constraint is more violated than boundary
        pair_violation = min_pair_dist / 2.0
        bound_violation = min_bound
        
        if pair_violation <= bound_violation:
            # Push closest pair apart
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                direction = diff / dist
                # Push each point by half the needed separation
                needed = 2 * current_r + 0.001
                push = direction * push_strength
                pts[i] += push
                pts[j] -= push
        else:
            # Push point closest to boundary inward
            # Find which point and which boundary
            abs_pts = np.abs(pts)
            min_abs = abs_pts.min()
            idx = np.unravel_index(np.argmin(abs_pts), abs_pts.shape)
            k, coord = idx[0], idx[1]
            
            if pts[k, coord] > 0:
                pts[k, coord] -= push_strength
            else:
                pts[k, coord] += push_strength
        
        # Also check all boundary constraints and push inward if violated
        for k in range(N):
            for coord in range(2):
                if pts[k, coord] > 0.5 - current_r:
                    pts[k, coord] = 0.5 - current_r - 0.0005
                elif pts[k, coord] < -0.5 + current_r:
                    pts[k, coord] = -0.5 + current_r + 0.0005
        
        # Clamp to square
        pts = np.clip(pts, -0.5, 0.5)
        
        # Occasionally add small random perturbation to escape local minima
        if n_iter % 50 == 0 and n_iter > 0:
            pts += np.random.normal(0, 0.001, size=pts.shape)
            pts = np.clip(pts, -0.49, 0.49)
        
        # Reduce push strength over time
        push_strength *= 0.999
        if push_strength < 0.0001:
            push_strength = 0.0001
        
        n_iter += 1
    
    return best_pts, best_score


def _polish(pts, time_budget, rng):
    """Local polishing: small random perturbations that must improve score."""
    pts = pts.copy()
    current_score = _score_fast(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    t_start = time.time()
    step_size = 0.01
    
    while time.time() - t_start < time_budget:
        i = rng.randint(0, N - 1)
        dx = (rng.random() - 0.5) * 2 * step_size
        dy = (rng.random() - 0.5) * 2 * step_size
        old_x, old_y = pts[i, 0], pts[i, 1]
        pts[i, 0] = np.clip(old_x + dx, -0.5, 0.5)
        pts[i, 1] = np.clip(old_y + dy, -0.5, 0.5)
        
        new_score = _score_fast(pts)
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        else:
            pts[i, 0] = old_x
            pts[i, 1] = old_y
        
        if rng.random() < 0.01:
            step_size *= 0.95
            if step_size < 0.0005:
                step_size = 0.0005
    
    return best_pts, best_score


def _bisection_refine(pts, time_budget, rng):
    """
    Bisection on radius: for a given target radius r, check if feasible
    (all pairwise distances >= 2r and all points within distance r of boundary).
    If feasible, try larger r; if not, try smaller r.
    """
    pts = pts.copy()
    current_score = _score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    t_start = time.time()
    r_low = current_score
    r_high = current_score + 0.01  # upper bound estimate
    
    # First, establish a valid upper bound
    # The theoretical max is 0.5, but practically much less
    # Use current score + small margin as initial high
    for _ in range(20):
        if time.time() - t_start > time_budget:
            break
        r_mid = (r_low + r_high) / 2
        # Try to find a feasible configuration at radius r_mid
        # by running constraint solver targeting r_mid
        feasible_pts, feasible_score = _constraint_solve(pts, time_budget * 0.1, rng)
        if feasible_score >= r_mid:
            r_low = r_mid
            best_pts = feasible_pts.copy()
            best_score = max(best_score, feasible_score)
        else:
            r_high = r_mid
        
        if r_high - r_low < 1e-6:
            break
    
    return best_pts, best_score


def construct():
    t_global_start = time.time()
    rng = random.Random(42)
    np.random.seed(42)
    
    # Generate diverse initial configurations
    inits = []
    inits.append(_make_grid())
    inits.append(_make_honeycomb())
    inits.append(_make_structured())
    inits.append(_make_random())
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Anneal each initial config
    n_inits = len(inits)
    for i, init in enumerate(inits):
        remaining = DEADLINE - (time.time() - t_global_start)
        if remaining < 5:
            break
        time_budget = min(remaining * 0.25, max(2.0, 10.0 - i * 2.0))
        pts, score = _anneal(init, time_budget, rng)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        return _make_grid()
    
    # Phase 2: Constraint-satisfaction refinement
    remaining = DEADLINE - (time.time() - t_global_start)
    if remaining > 10:
        cs_time = min(remaining * 0.4, 30.0)
        best_pts, best_score = _constraint_solve(best_pts, cs_time, rng)
    
    # Phase 3: Bisection refinement
    remaining = DEADLINE - (time.time() - t_global_start)
    if remaining > 15:
        bis_time = min(remaining * 0.35, 25.0)
        best_pts, best_score = _bisection_refine(best_pts, bis_time, rng)
    
    # Phase 4: Polish the best result
    remaining = DEADLINE - (time.time() - t_global_start)
    if remaining > 5:
        polish_time = min(remaining * 0.4, 15.0)
        best_pts, best_score = _polish(best_pts, polish_time, rng)
    
    # Phase 5: Final micro-polish
    remaining = DEADLINE - (time.time() - t_global_start)
    if remaining > 3:
        polish_time = min(remaining * 0.5, 10.0)
        best_pts, best_score = _polish(best_pts, polish_time, rng)
    
    # Ensure validity: clamp to square
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    # Final check
    final_score = _score(best_pts)
    if final_score <= 0 or not np.all(np.isfinite(best_pts)):
        return _make_grid()
    
    return best_pts
# EVOLVE-BLOCK-END
