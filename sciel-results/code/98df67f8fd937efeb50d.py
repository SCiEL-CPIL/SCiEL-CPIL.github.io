# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, positive sides, non-overlapping. Vectorized."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Inside unit square
    if np.any(centers - sides[:, None] / 2.0 < -tol):
        return False
    if np.any(centers + sides[:, None] / 2.0 > 1.0 + tol):
        return False
    # Non-overlap: vectorized
    dx = np.abs(centers[:, 0, None] - centers[None, :, 0])
    dy = np.abs(centers[:, 1, None] - centers[None, :, 1])
    t = (sides[:, None] + sides[None, :]) / 2.0 - tol
    # Only check i < j
    iu = np.triu_indices(n, k=1)
    if np.any(dx[iu] < t[iu]) and np.any(dy[iu] < t[iu]):
        # Need both conditions for the SAME pair
        overlap = (dx[iu] < t[iu]) & (dy[iu] < t[iu])
        if np.any(overlap):
            return False
    return True


def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))


def _max_side_at(cx, cy, centers, sides, i, tol=1e-7):
    """
    Compute the maximum side length for square i centered at (cx,cy)
    given the positions and sizes of all other squares.
    """
    # Bound from unit square
    max_s = min(2.0 * cx, 2.0 * (1.0 - cx), 2.0 * cy, 2.0 * (1.0 - cy))
    if max_s < 0:
        max_s = 0.0
    
    for j in range(len(sides)):
        if j == i:
            continue
        dx = abs(cx - centers[j, 0])
        dy = abs(cy - centers[j, 1])
        bound_x = 2.0 * (dx + tol) - sides[j]
        bound_y = 2.0 * (dy + tol) - sides[j]
        bound = max(bound_x, bound_y)
        if bound < max_s:
            max_s = bound
    
    if max_s < 0:
        max_s = 0.0
    return max_s


def _max_side_all(centers, sides, tol=1e-7):
    """Vectorized: compute max side for all squares simultaneously."""
    n = len(sides)
    max_s = np.empty(n)
    # Boundary bounds
    max_s[0] = 2.0 * centers[0, 0]
    max_s[1] = 2.0 * (1.0 - centers[1, 0])
    # Actually compute properly
    bounds = np.minimum(
        np.minimum(2.0 * centers[:, 0], 2.0 * (1.0 - centers[:, 0])),
        np.minimum(2.0 * centers[:, 1], 2.0 * (1.0 - centers[:, 1]))
    )
    max_s = np.maximum(bounds, 0.0)
    
    # Pairwise constraints
    dx = np.abs(centers[:, 0, None] - centers[None, :, 0])
    dy = np.abs(centers[:, 1, None] - centers[None, :, 1])
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            bx = 2.0 * (dx[i, j] + tol) - sides[j]
            by = 2.0 * (dy[i, j] + tol) - sides[j]
            bound = max(bx, by)
            if bound < max_s[i]:
                max_s[i] = bound
        max_s[i] = max(max_s[i], 0.0)
    
    return max_s


def _clamp_to_unit(centers, sides):
    """Clamp centers so squares stay inside [0,1]^2."""
    for i in range(len(sides)):
        half = sides[i] / 2.0
        centers[i, 0] = max(half, min(1.0 - half, centers[i, 0]))
        centers[i, 1] = max(half, min(1.0 - half, centers[i, 1]))
    return centers


def _resolve_overlap(centers, sides, tol=1e-7):
    """
    Attempt to resolve overlaps by pushing squares apart.
    """
    n = len(sides)
    for iteration in range(500):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                if t < 0:
                    continue
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                if abs(dx) < t and abs(dy) < t:
                    # Push apart
                    if abs(dx) < abs(dy):
                        push = (t - abs(dx)) / 2.0 + 2*tol
                        if dx >= 0:
                            centers[j, 0] += push
                            centers[i, 0] -= push
                        else:
                            centers[j, 0] -= push
                            centers[i, 0] += push
                    else:
                        push = (t - abs(dy)) / 2.0 + 2*tol
                        if dy >= 0:
                            centers[j, 1] += push
                            centers[i, 1] -= push
                        else:
                            centers[j, 1] -= push
                            centers[i, 1] += push
                    moved = True
        if not moved:
            break
    return centers


def _grow_all_squares(centers, sides, tol=1e-7, max_rounds=100):
    """
    Iteratively grow all squares to their maximum given current positions.
    Repeat until convergence.
    """
    n = len(sides)
    for _ in range(max_rounds):
        improved = False
        for i in range(n):
            new_s = _max_side_at(centers[i, 0], centers[i, 1], centers, sides, i, tol)
            if new_s > sides[i] + 1e-10:
                sides[i] = new_s
                improved = True
        if not improved:
            break
    return sides


def _repulsion_energy(centers, sides, tol=1e-7, beta=50.0):
    """
    Smooth repulsion energy: high when squares overlap, ~0 when separated.
    Uses soft-min of (t - dx, t - dy) per pair, squared and summed.
    """
    n = len(sides)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t < 0:
                continue
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            a = -beta * (t - dx)
            b = -beta * (t - dy)
            m = max(a, b)
            if m > 700:
                soft_min = m / beta
            else:
                soft_min = (m - math.log(math.exp(a - m) + math.exp(b - m))) / beta
            if soft_min > 0:
                energy += soft_min ** 2
    return energy


def _smooth_penalty(centers, sides, tol=1e-7, beta=50.0):
    """
    Smooth surrogate for the nonsmooth overlap constraint using log-sum-exp.
    Returns a penalty that is ~0 when no overlap, and large when overlaps exist.
    This can guide local search away from infeasible regions.
    """
    n = len(sides)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t < 0:
                continue
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            # Violation amount: how much dx and dy are below t
            # Use soft-min: -logsumexp(-beta*(t-dx), -beta*(t-dy)) approximates min(t-dx, t-dy)
            # If both t-dx and t-dy are positive, no overlap. Penalty should be 0.
            # If either is negative, there's a violation.
            a = beta * (t - dx)
            b = beta * (t - dy)
            # We want penalty when min(a, b) < 0, i.e., when overlap occurs
            # soft_min = -log(exp(a) + exp(b)) / beta  (approx min(a,b)/beta)
            m = max(a, b)
            if m > 700:
                soft_min = m / beta
            else:
                soft_min = (m - math.log(math.exp(a - m) + math.exp(b - m))) / beta
            # soft_min approximates min(t-dx, t-dy)/beta * beta = min(t-dx, t-dy)
            # Wait, let me redo. We want a smooth version of max(0, max(dx-t, dy-t))
            # Overlap occurs when dx < t AND dy < t, i.e., max(dx-t, dy-t) < 0
            # Actually overlap when min(t-dx, t-dy) > 0
            # Penalty = max(0, min(t-dx, t-dy)) smoothed
            # Use: soft_min(t-dx, t-dy) = -1/beta * log(exp(-beta*(t-dx)) + exp(-beta*(t-dy)))
            # This approximates min(t-dx, t-dy)
            # Then penalty = max(0, soft_min)
            # But for smoothness, just use soft_min if positive
            if soft_min > 0:
                penalty += soft_min ** 2
    return penalty


def _local_search(centers, sides, tol=1e-7, n_iter=200, rng=None, use_smooth=False):
    """
    Local search: perturb positions, then grow all squares.
    Keep the best result. Optionally use smooth penalty to guide search.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)
    
    for it in range(n_iter):
        # Perturb a random subset of squares
        k = rng.integers(1, n)
        idx = rng.choice(n, size=k, replace=False)
        trial_c = best_c.copy()
        for i in idx:
            step = rng.uniform(0.01, 0.08)
            trial_c[i, 0] += rng.uniform(-step, step)
            trial_c[i, 1] += rng.uniform(-step, step)
        
        trial_c = _clamp_to_unit(trial_c, best_s)
        
        if not _is_valid(trial_c, best_s, tol):
            # Try to resolve
            trial_c2 = trial_c.copy()
            trial_c2 = _resolve_overlap(trial_c2, best_s, tol)
            trial_c2 = _clamp_to_unit(trial_c2, best_s)
            if _is_valid(trial_c2, best_s, tol):
                trial_c = trial_c2
            else:
                continue
        
        # Grow all squares
        trial_s = best_s.copy()
        trial_s = _grow_all_squares(trial_c, trial_s, tol)
        
        if _is_valid(trial_c, trial_s, tol):
            obj = _objective(trial_c, trial_s)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_c = trial_c.copy()
                best_s = trial_s.copy()
    
    return best_c, best_s


def _simulated_annealing(centers, sides, tol=1e-7, n_iter=800, rng=None,
                          t_start=0.15, t_end=0.003, cooling=None):
    """
    Simulated annealing using smooth repulsion energy as the objective landscape.
    Moves are accepted probabilistically based on energy change, allowing escape
    from local optima. The side lengths are recomputed from positions each step.
    """
    if rng is None:
        rng = np.random.default_rng(777)
    
    n = len(sides)
    if cooling is None:
        cooling = (t_end / t_start) ** (1.0 / max(n_iter - 1, 1))
    
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)
    
    cur_c = centers.copy()
    cur_s = sides.copy()
    cur_energy = _repulsion_energy(cur_c, cur_s, tol=tol, beta=50.0)
    cur_obj = _objective(cur_c, cur_s)
    
    T = t_start
    for it in range(n_iter):
        T *= cooling
        
        # Generate a move: perturb 1-3 squares
        k = rng.integers(1, 4)
        idx = rng.choice(n, size=k, replace=False)
        trial_c = cur_c.copy()
        for i in idx:
            step = rng.uniform(0.005, 0.10) * (1.0 + 0.5 * T / t_start)
            trial_c[i, 0] += rng.uniform(-step, step)
            trial_c[i, 1] += rng.uniform(-step, step)
        
        trial_c = _clamp_to_unit(trial_c, cur_s)
        
        # Recompute sides from positions (max possible)
        trial_s = np.empty(n)
        for i in range(n):
            trial_s[i] = _max_side_at(trial_c[i, 0], trial_c[i, 1], trial_c, cur_s, i, tol)
        
        # Check validity
        if not _is_valid(trial_c, trial_s, tol):
            # Try resolving
            trial_c2 = trial_c.copy()
            trial_c2 = _resolve_overlap(trial_c2, trial_s, tol)
            trial_c2 = _clamp_to_unit(trial_c2, trial_s)
            if not _is_valid(trial_c2, trial_s, tol):
                # Reject this move
                continue
        
        trial_c = trial_c2 if not _is_valid(trial_c, trial_s, tol) else trial_c
        # Re-grow sides after resolution
        for i in range(n):
            trial_s[i] = _max_side_at(trial_c[i, 0], trial_c[i, 1], trial_c, trial_s, i, tol)
        
        if not _is_valid(trial_c, trial_s, tol):
            continue
        
        trial_obj = _objective(trial_c, trial_s)
        trial_energy = _repulsion_energy(trial_c, trial_s, tol=tol, beta=50.0)
        
        # Accept if better, or with probability based on energy difference
        delta_obj = trial_obj - cur_obj
        delta_energy = trial_energy - cur_energy
        
        if delta_obj > 0 or delta_energy < 0:
            accept = True
        else:
            # Boltzmann acceptance: probability of accepting worse state
            # Use combined penalty
            delta = -delta_obj + 0.1 * delta_energy
            if T > 1e-10:
                accept_prob = math.exp(-delta / T)
                accept = rng.random() < accept_prob
            else:
                accept = False
        
        if accept:
            cur_c = trial_c.copy()
            cur_s = trial_s.copy()
            cur_obj = trial_obj
            cur_energy = trial_energy
            
            if cur_obj > best_obj + 1e-10:
                best_obj = cur_obj
                best_c = cur_c.copy()
                best_s = cur_s.copy()
    
    return best_c, best_s


def _high_precision_polish(centers, sides, tol=1e-9, max_rounds=500):
    """
    Final high-precision polish: grow squares with tighter tolerance and more iterations.
    Also do a few more local search steps with smaller steps.
    """
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)
    
    # First, grow with tight tolerance
    trial_s = best_s.copy()
    trial_s = _grow_all_squares(best_c, trial_s, tol=tol, max_rounds=max_rounds)
    if _is_valid(best_c, trial_s, tol=1e-7):
        obj = _objective(best_c, trial_s)
        if obj > best_obj:
            best_obj = obj
            best_s = trial_s.copy()
    
    # Small-step local search with tight tolerance
    rng = np.random.default_rng(99999)
    for it in range(100):
        k = rng.integers(1, 4)
        idx = rng.choice(10, size=k, replace=False)
        trial_c = best_c.copy()
        for i in idx:
            step = rng.uniform(0.001, 0.01)
            trial_c[i, 0] += rng.uniform(-step, step)
            trial_c[i, 1] += rng.uniform(-step, step)
        
        trial_c = _clamp_to_unit(trial_c, best_s)
        
        if not _is_valid(trial_c, best_s, tol=1e-7):
            trial_c2 = trial_c.copy()
            trial_c2 = _resolve_overlap(trial_c2, best_s, tol=1e-7)
            trial_c2 = _clamp_to_unit(trial_c2, best_s)
            if _is_valid(trial_c2, best_s, tol=1e-7):
                trial_c = trial_c2
            else:
                continue
        
        trial_s = best_s.copy()
        trial_s = _grow_all_squares(trial_c, trial_s, tol=tol, max_rounds=max_rounds)
        
        if _is_valid(trial_c, trial_s, tol=1e-7):
            obj = _objective(trial_c, trial_s)
            if obj > best_obj + 1e-10:
                best_obj = obj
                best_c = trial_c.copy()
                best_s = trial_s.copy()
    
    return best_c, best_s


def _init_symmetric_pattern1():
    """
    Pattern: 4 large squares in corners, 6 smaller filling the cross.
    Large: side 0.4, centers at (0.2,0.2), (0.8,0.2), (0.2,0.8), (0.8,0.8)
    Remaining: horizontal strip [0.4,0.6]x[0,1] and vertical strip [0,1]x[0.4,0.6]
    But they overlap in [0.4,0.6]x[0.4,0.6].
    Let's place 6 small squares of side 0.2:
    Top: [0.4,0.6]x[0.6,0.8] -> center (0.5, 0.7), side 0.2
    Bottom: [0.4,0.6]x[0.2,0.4] -> center (0.5, 0.3), side 0.2
    Left: [0.2,0.4]x[0.4,0.6] -> center (0.3, 0.5), side 0.2
    Right: [0.6,0.8]x[0.4,0.6] -> center (0.7, 0.5), side 0.2
    Center: [0.4,0.6]x[0.4,0.6] -> center (0.5, 0.5), side 0.2
    That's 5 small. Need one more.
    Let's use 4 large (0.38) and 6 small.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    
    s_large = 0.38
    centers[0] = [s_large/2, s_large/2]
    centers[1] = [1 - s_large/2, s_large/2]
    centers[2] = [s_large/2, 1 - s_large/2]
    centers[3] = [1 - s_large/2, 1 - s_large/2]
    sides[0:4] = s_large
    
    s_small = 0.20
    centers[4] = [0.5, s_large + s_small/2]      # bottom center
    centers[5] = [0.5, 1 - s_large - s_small/2]  # top center
    centers[6] = [s_large + s_small/2, 0.5]      # left center
    centers[7] = [1 - s_large - s_small/2, 0.5]  # right center
    centers[8] = [0.5, 0.5]                       # center
    sides[4:9] = s_small
    
    # 10th square: need to fit somewhere
    # Try placing it in a gap
    centers[9] = [0.5, 0.5 + s_small]  # just above center
    sides[9] = s_small * 0.8
    
    return centers, sides


def _init_symmetric_pattern2():
    """
    2x5 grid: 2 rows of 5.
    Side = 1/5 = 0.2
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.full(10, 0.2, dtype=np.float64)
    idx = 0
    for row in range(2):
        for col in range(5):
            centers[idx, 0] = (col + 0.5) * 0.2
            centers[idx, 1] = (row + 0.5) * 0.5
            idx += 1
    return centers, sides


def _init_symmetric_pattern3():
    """
    One large (0.5) in bottom-left, 3x3 grid of small (1/6) in the L-shape.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    side_s = 1.0/6.0
    idx = 1
    # Top strip [0,1]x[0.5,1]: 6 in a row
    for col in range(6):
        centers[idx, 0] = (col + 0.5) * side_s
        centers[idx, 1] = 0.5 + 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    # Right strip [0.5,1]x[0,0.5]: 3 in a row
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_s
        centers[idx, 1] = 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_symmetric_pattern4():
    """
    3x3 grid (9 squares of 1/3) + 1 large in center gap? No, 3x3 fills it.
    Let's do: 1 large (0.44) center, 8 around.
    Large: [0.28, 0.72]x[0.28, 0.72], center (0.5, 0.5)
    8 squares around: 4 corners and 4 edge-mids.
    Corner squares: side 0.28, at (0.14,0.14), (0.86,0.14), (0.14,0.86), (0.86,0.86)
    Edge squares: side 0.28, at (0.5, 0.14), (0.5, 0.86), (0.14, 0.5), (0.86, 0.5)
    Check: large is [0.28,0.72]x[0.28,0.72]. Corner (0.14,0.14) with side 0.28: [0,0.28]x[0,0.28]. OK.
    Edge (0.5,0.14) with side 0.28: [0.36,0.64]x[0,0.28]. OK, no overlap with large.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.5, 0.5]
    sides[0] = 0.44
    s_small = 0.28
    positions = [
        [0.14, 0.14], [0.86, 0.14], [0.14, 0.86], [0.86, 0.86],
        [0.5, 0.14], [0.5, 0.86], [0.14, 0.5], [0.86, 0.5]
    ]
    for i, pos in enumerate(positions):
        centers[i+1] = pos
        sides[i+1] = s_small
    return centers, sides


def _init_symmetric_pattern5():
    """
    5 large in a cross, 5 small in corners.
    Center: (0.5, 0.5), side 0.3
    Arms: (0.5, 0.2), (0.5, 0.8), (0.2, 0.5), (0.8, 0.5), side 0.3
    Corners: (0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85), side 0.15
    Check overlaps: Center [0.35,0.65]x[0.35,0.65]. Arm (0.5,0.2) side 0.3: [0.35,0.65]x[0.05,0.35]. OK.
    Corner (0.15,0.15) side 0.15: [0.075,0.225]x[0.075,0.225]. OK.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.5, 0.5]
    sides[0] = 0.3
    centers[1] = [0.5, 0.2]
    centers[2] = [0.5, 0.8]
    centers[3] = [0.2, 0.5]
    centers[4] = [0.8, 0.5]
    sides[1:5] = 0.3
    centers[5] = [0.15, 0.15]
    centers[6] = [0.85, 0.15]
    centers[7] = [0.15, 0.85]
    centers[8] = [0.85, 0.85]
    sides[5:9] = 0.15
    # 10th: try (0.5, 0.5) is taken. Let's put a small one somewhere.
    # Actually we have 4 corners, but only 4 positions. Let's adjust.
    # Use 4 large in cross arms, 1 center, 5 small.
    # Redo: Center (0.5,0.5) side 0.25. Arms at dist 0.35, side 0.25.
    centers[0] = [0.5, 0.5]
    sides[0] = 0.25
    centers[1] = [0.5, 0.25]
    centers[2] = [0.5, 0.75]
    centers[3] = [0.25, 0.5]
    centers[4] = [0.75, 0.5]
    sides[1:5] = 0.25
    # 5 small in corners and mid-edges
    s_small = 0.15
    centers[5] = [0.125, 0.125]
    centers[6] = [0.875, 0.125]
    centers[7] = [0.125, 0.875]
    centers[8] = [0.875, 0.875]
    centers[9] = [0.5, 0.05]
    sides[5:10] = s_small
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 non-overlapping axis-aligned squares in the unit square.
    """
    start_time = time.time()
    deadline = start_time + 570.0
    
    inits = [
        _init_symmetric_pattern1(),
        _init_symmetric_pattern2(),
        _init_symmetric_pattern3(),
        _init_symmetric_pattern4(),
        _init_symmetric_pattern5(),
    ]
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    rng_global = np.random.default_rng(12345)
    
    for init_idx, (centers, sides) in enumerate(inits):
        if time.time() > deadline:
            break
        
        centers = centers.copy()
        sides = sides.copy()
        
        # Ensure validity
        centers = _clamp_to_unit(centers, sides)
        if not _is_valid(centers, sides, tol=1e-7):
            centers = _resolve_overlap(centers, sides, tol=1e-7)
            centers = _clamp_to_unit(centers, sides)
            if not _is_valid(centers, sides, tol=1e-7):
                continue
        
        # Grow all squares
        sides = _grow_all_squares(centers, sides, tol=1e-7)
        if not _is_valid(centers, sides, tol=1e-7):
            continue
        
        obj = _objective(centers, sides)
        if obj > best_obj:
            best_obj = obj
            best_centers = centers.copy()
            best_sides = sides.copy()
        
        # Local search from this point
        if time.time() < deadline - 10:
            rng = np.random.default_rng(init_idx * 7919 + 1)
            opt_c, opt_s = _local_search(centers, sides, tol=1e-7, n_iter=200, rng=rng)
            if _is_valid(opt_c, opt_s, tol=1e-7):
                obj2 = _objective(opt_c, opt_s)
                if obj2 > best_obj:
                    best_obj = obj2
                    best_centers = opt_c.copy()
                    best_sides = opt_s.copy()
        
        # Simulated annealing from this point (uses smooth energy landscape)
        if time.time() < deadline - 30:
            rng_sa = np.random.default_rng(init_idx * 104729 + 3)
            sa_c, sa_s = _simulated_annealing(centers, sides, tol=1e-7, n_iter=800, rng=rng_sa,
                                              t_start=0.15, t_end=0.003)
            if _is_valid(sa_c, sa_s, tol=1e-7):
                obj3 = _objective(sa_c, sa_s)
                if obj3 > best_obj:
                    best_obj = obj3
                    best_centers = sa_c.copy()
                    best_sides = sa_s.copy()
    
    # If nothing valid, fallback
    if best_centers is None:
        centers, sides = _init_symmetric_pattern2()
        centers = _clamp_to_unit(centers, sides)
        sides = _grow_all_squares(centers, sides, tol=1e-7)
        best_centers = centers
        best_sides = sides
    else:
        # Run SA on the best result found
        if time.time() < deadline - 20:
            rng_sa2 = np.random.default_rng(987654)
            sa_c, sa_s = _simulated_annealing(best_centers, best_sides, tol=1e-7, n_iter=600, rng=rng_sa2,
                                              t_start=0.08, t_end=0.002)
            if _is_valid(sa_c, sa_s, tol=1e-7):
                obj_sa = _objective(sa_c, sa_s)
                if obj_sa > best_obj:
                    best_obj = obj_sa
                    best_centers = sa_c.copy()
                    best_sides = sa_s.copy()
        
        # High-precision final polish with tighter tolerance
        if time.time() < deadline - 5:
            polished_c, polished_s = _high_precision_polish(best_centers, best_sides, tol=1e-9, max_rounds=500)
            if _is_valid(polished_c, polished_s, tol=1e-7):
                obj_polished = _objective(polished_c, polished_s)
                if obj_polished > best_obj:
                    best_centers = polished_c
                    best_sides = polished_s
                    best_obj = obj_polished
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
