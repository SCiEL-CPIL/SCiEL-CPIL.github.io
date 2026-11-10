# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, no overlaps, positive sides."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))


def _repulsion_penalty(centers, sides, tol=1e-7):
    """
    Soft repulsion penalty for overlapping squares.
    For each pair (i,j), if they overlap, add a quadratic penalty proportional to
    the penetration depth in both x and y.
    """
    n = len(sides)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            # Penetration in x and y
            pen_x = max(0.0, t - dx)
            pen_y = max(0.0, t - dy)
            if pen_x > 0 and pen_y > 0:
                # Both dimensions overlap -> actual overlap
                penalty += pen_x ** 2 + pen_y ** 2
    return penalty


def _boundary_penalty(centers, sides, tol=1e-7):
    """Penalty for squares extending outside the unit square."""
    n = len(sides)
    penalty = 0.0
    for i in range(n):
        # Left
        v = centers[i, 0] - sides[i] / 2
        if v < 0:
            penalty += v ** 2
        # Right
        v = centers[i, 0] + sides[i] / 2 - 1
        if v > 0:
            penalty += v ** 2
        # Bottom
        v = centers[i, 1] - sides[i] / 2
        if v < 0:
            penalty += v ** 2
        # Top
        v = centers[i, 1] + sides[i] / 2 - 1
        if v > 0:
            penalty += v ** 2
    return penalty


def _sa_objective(centers, sides, tol=1e-7):
    """
    Combined objective: maximize sum(sides) - penalty.
    Higher is better.
    """
    total_side = np.sum(sides)
    pen = _repulsion_penalty(centers, sides, tol) + _boundary_penalty(centers, sides, tol)
    return total_side - 10.0 * pen


def _project_feasible(centers, sides, tol=1e-7):
    """
    Project the configuration to be feasible:
    1. Clamp sides to (0, 1].
    2. Clamp centers so squares stay in [0,1]^2.
    3. Iteratively resolve overlaps by shrinking the smaller square and pushing centers apart.
    """
    n = len(sides)
    centers = centers.copy().astype(np.float64)
    sides = sides.copy().astype(np.float64)
    
    # Clamp sides
    sides = np.clip(sides, 0.001, 1.0)
    
    # Clamp centers for boundary
    for i in range(n):
        centers[i, 0] = max(sides[i] / 2, min(1 - sides[i] / 2, centers[i, 0]))
        centers[i, 1] = max(sides[i] / 2, min(1 - sides[i] / 2, centers[i, 1]))
    
    # Iteratively resolve overlaps
    for iteration in range(200):
        resolved = True
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                adx = abs(dx)
                ady = abs(dy)
                
                if adx < t and ady < t:
                    resolved = False
                    # Overlap: push apart along the axis of least penetration
                    pen_x = t - adx
                    pen_y = t - ady
                    
                    if pen_x < pen_y:
                        # Push apart in x
                        if dx >= 0:
                            push = pen_x / 2 + 1e-9
                            centers[i, 0] += push
                            centers[j, 0] -= push
                        else:
                            push = pen_x / 2 + 1e-9
                            centers[i, 0] -= push
                            centers[j, 0] += push
                    else:
                        # Push apart in y
                        if dy >= 0:
                            push = pen_y / 2 + 1e-9
                            centers[i, 1] += push
                            centers[j, 1] -= push
                        else:
                            push = pen_y / 2 + 1e-9
                            centers[i, 1] -= push
                            centers[j, 1] += push
                    
                    # Re-clamp boundaries
                    for k in [i, j]:
                        centers[k, 0] = max(sides[k] / 2, min(1 - sides[k] / 2, centers[k, 0]))
                        centers[k, 1] = max(sides[k] / 2, min(1 - sides[k] / 2, centers[k, 1]))
        
        if resolved:
            break
    
    # Final check: if still invalid, shrink squares
    if not _is_valid(centers, sides, tol):
        for i in range(n):
            # Shrink square i until it doesn't overlap
            for attempt in range(50):
                if _is_valid(centers, sides, tol):
                    break
                sides[i] *= 0.95
                centers[i, 0] = max(sides[i] / 2, min(1 - sides[i] / 2, centers[i, 0]))
                centers[i, 1] = max(sides[i] / 2, min(1 - sides[i] / 2, centers[i, 1]))
    
    return centers, sides


def _simulated_annealing(centers0, sides0, tol=1e-7, time_limit=None, rng_seed=42):
    """
    Simulated annealing on the continuous (centers, sides) space.
    Objective: maximize sum(sides) - penalty.
    Moves: random perturbation of centers and sides.
    Accept: Metropolis criterion with cooling schedule.
    Periodically project to feasible set to maintain validity.
    """
    n = len(sides0)
    rng = np.random.default_rng(rng_seed)
    
    centers = centers0.copy().astype(np.float64)
    sides = sides0.copy().astype(np.float64)
    
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = _sa_objective(centers, sides, tol)
    
    T_init = 0.5
    T_min = 1e-8
    n_steps = 50000
    # Exponential cooling
    cool_rate = (T_min / T_init) ** (1.0 / n_steps)
    
    start_time = time.time()
    step = 0
    
    for step in range(n_steps):
        if time_limit is not None and time.time() > time_limit:
            break
        
        T = T_init * (cool_rate ** step)
        
        # Generate a random move
        # Perturb a random subset of squares
        n_perturb = rng.integers(1, 4)  # 1-3 squares
        indices = rng.choice(n, size=n_perturb, replace=False)
        
        new_centers = centers.copy()
        new_sides = sides.copy()
        
        for idx in indices:
            # Perturb center
            new_centers[idx, 0] += rng.normal(0, 0.05)
            new_centers[idx, 1] += rng.normal(0, 0.05)
            # Perturb side (log-normal to keep positive)
            new_sides[idx] *= np.exp(rng.normal(0, 0.03))
            new_sides[idx] = max(0.001, min(1.0, new_sides[idx]))
        
        # Clamp centers
        for idx in indices:
            new_centers[idx, 0] = max(new_sides[idx] / 2, min(1 - new_sides[idx] / 2, new_centers[idx, 0]))
            new_centers[idx, 1] = max(new_sides[idx] / 2, min(1 - new_sides[idx] / 2, new_centers[idx, 1]))
        
        old_score = _sa_objective(centers, sides, tol)
        new_score = _sa_objective(new_centers, new_sides, tol)
        
        delta = new_score - old_score
        
        if delta > 0 or (T > T_min and rng.random() < np.exp(delta / T)):
            centers = new_centers
            sides = new_sides
        
        # Track best
        cur_score = _sa_objective(centers, sides, tol)
        if cur_score > best_score:
            best_score = cur_score
            best_centers = centers.copy()
            best_sides = sides.copy()
        
        # Periodically project to feasible set (every 500 steps or when penalty is high)
        if step % 500 == 0 or _repulsion_penalty(centers, sides, tol) > 0.01:
            proj_centers, proj_sides = _project_feasible(centers, sides, tol)
            proj_score = _sa_objective(proj_centers, proj_sides, tol)
            if proj_score > _sa_objective(centers, sides, tol):
                centers = proj_centers
                sides = proj_sides
                if proj_score > best_score:
                    best_score = proj_score
                    best_centers = proj_centers.copy()
                    best_sides = proj_sides.copy()
    
    # Final projection
    proj_centers, proj_sides = _project_feasible(best_centers, best_sides, tol)
    if _is_valid(proj_centers, proj_sides, tol):
        return proj_centers, proj_sides
    
    return best_centers, best_sides


def _seed_grid():
    """4x4 grid, use 10 cells."""
    n = 10
    side = 0.25
    centers = []
    sides = []
    for idx in range(10):
        col = idx % 4
        row = idx // 4
        cx = (col + 0.5) * side
        cy = (row + 0.5) * side
        centers.append([cx, cy])
        sides.append(side)
    return np.array(centers, dtype=np.float64), np.array(sides, dtype=np.float64)


def _seed_heterogeneous():
    """One large square in corner, fill rest."""
    n = 10
    centers = []
    sides = []
    
    # Large square in bottom-left
    s0 = 0.4
    centers.append([s0 / 2, s0 / 2])
    sides.append(s0)
    
    # Right strip: [0.4, 1] x [0, 1], width 0.6, height 1
    # 2 cols x 3 rows of 0.3
    s1 = 0.3
    for r in range(3):
        for c in range(2):
            cx = 0.4 + (c + 0.5) * s1
            cy = (r + 0.5) * s1
            centers.append([cx, cy])
            sides.append(s1)
    
    # Top-left: [0, 0.4] x [0.4, 1], side 0.4
    s2 = 0.4
    centers.append([s2 / 2, 0.4 + (1 - s2) / 2])
    sides.append(s2)
    
    # That's 1 + 6 + 1 = 8. Need 2 more.
    # Add 2 more in remaining gaps
    # Gap: [0.4, 1] x [0.9, 1] is 0.6 x 0.1 - too small for 0.3
    # Let's add smaller squares in corners
    s3 = 0.15
    centers.append([0.4 + s3/2, 1 - s3/2])  # top-right of right strip
    sides.append(s3)
    centers.append([s3/2, 0.4 + s3/2])  # bottom of top-left area
    sides.append(s3)
    
    if len(centers) == 10:
        return np.array(centers, dtype=np.float64), np.array(sides, dtype=np.float64)
    
    return _seed_grid()


def _seed_two_large():
    """Two large squares in opposite corners."""
    n = 10
    centers = []
    sides = []
    
    s0 = 0.35
    centers.append([s0 / 2, s0 / 2])
    sides.append(s0)
    
    s1 = 0.35
    centers.append([1 - s1 / 2, 1 - s1 / 2])
    sides.append(s1)
    
    # Fill with 8 small squares of 0.2
    s2 = 0.2
    positions = []
    for r in range(5):
        for c in range(5):
            cx = (c + 0.5) * s2
            cy = (r + 0.5) * s2
            positions.append([cx, cy])
    
    valid_positions = []
    for pos in positions:
        cx, cy = pos
        ok = True
        t0 = (s2 + s0) / 2
        if abs(cx - s0/2) < t0 and abs(cy - s0/2) < t0:
            ok = False
        t1 = (s2 + s1) / 2
        if abs(cx - (1-s1/2)) < t1 and abs(cy - (1-s1/2)) < t1:
            ok = False
        if ok:
            valid_positions.append(pos)
    
    for pos in valid_positions[:8]:
        centers.append(pos)
        sides.append(s2)
    
    if len(centers) == 10:
        return np.array(centers, dtype=np.float64), np.array(sides, dtype=np.float64)
    
    return _seed_grid()


def _seed_strip():
    """2 rows of 5, side 0.2."""
    n = 10
    centers = []
    sides = []
    s = 0.2
    for r in range(2):
        for c in range(5):
            cx = (c + 0.5) * s
            cy = (r + 0.5) * s
            centers.append([cx, cy])
            sides.append(s)
    return np.array(centers, dtype=np.float64), np.array(sides, dtype=np.float64)


def _seed_random(rng_seed):
    """Random initial packing with small squares."""
    n = 10
    rng = np.random.default_rng(rng_seed)
    sides = rng.uniform(0.05, 0.2, size=n)
    centers = np.zeros((n, 2))
    for i in range(n):
        centers[i, 0] = rng.uniform(sides[i] / 2, 1 - sides[i] / 2)
        centers[i, 1] = rng.uniform(sides[i] / 2, 1 - sides[i] / 2)
    # Project to feasible
    centers, sides = _project_feasible(centers, sides)
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares in the unit square to maximize sum of side lengths.
    
    Strategy: Simulated annealing with soft-constraint repulsion potential.
    Multiple restarts from diverse seeds. Periodic projection to feasible set.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 570  # Safety margin
    
    seeds = [
        _seed_grid(),
        _seed_heterogeneous(),
        _seed_two_large(),
        _seed_strip(),
        _seed_random(123),
        _seed_random(456),
    ]
    
    best_centers = None
    best_sides = None
    best_val = -float('inf')
    
    for seed_idx, (centers, sides) in enumerate(seeds):
        if time.time() > deadline - 60:
            break
        
        # Ensure seed is valid
        if not _is_valid(centers, sides, tol):
            centers, sides = _project_feasible(centers, sides, tol)
            if not _is_valid(centers, sides, tol):
                continue
        
        # Run SA
        sa_centers, sa_sides = _simulated_annealing(
            centers, sides, tol,
            time_limit=deadline - 30,
            rng_seed=1000 + seed_idx
        )
        
        # Verify and update best
        if _is_valid(sa_centers, sa_sides, tol):
            val = _objective(sa_centers, sa_sides)
            if val > best_val:
                best_val = val
                best_centers = sa_centers.copy()
                best_sides = sa_sides.copy()
        
        # Also check the seed itself
        if _is_valid(centers, sides, tol):
            val = _objective(centers, sides)
            if val > best_val:
                best_val = val
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # If no valid result, fall back
    if best_centers is None:
        centers, sides = _seed_grid()
        if not _is_valid(centers, sides, tol):
            centers = np.array([[0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
                                 [0.1, 0.3], [0.3, 0.3], [0.5, 0.3], [0.7, 0.3], [0.9, 0.3]], dtype=np.float64)
            sides = np.full(10, 0.15, dtype=np.float64)
        best_centers = centers
        best_sides = sides
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
