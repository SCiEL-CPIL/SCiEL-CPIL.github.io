"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _score(pts):
    """Largest common radius for a set of centres."""
    if pts is None or len(pts) == 0:
        return 0.0
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # distance to each of the 4 walls
    wall = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5),
    )
    wall_min = wall.min()
    # nearest neighbour distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min()
    return min(wall_min, 0.5 * nn)


def _make_hex(rows, spacing, dy, xoff):
    """Build a hex lattice with the given row lengths."""
    pts = []
    h = spacing * np.sqrt(3.0) / 2.0
    for i, n in enumerate(rows):
        y = -dy + i * h
        # rows of odd length offset by half spacing
        base = (0.5 if (n % 2 == 1) else 0.0) * spacing
        xs = np.arange(n) * spacing + base
        # center this row horizontally
        xs = xs - (xs.max() + xs.min()) / 2.0 + xoff
        for x in xs:
            pts.append([x, y])
    return np.array(pts)


def _neg_score(params, rows):
    spacing, dy, xoff = params
    if spacing <= 0:
        return 1.0
    pts = _make_hex(rows, spacing, dy, xoff)
    return -_score(pts)


def _soft_neg_score(params, rows, beta=50.0):
    """Smooth surrogate for the nonsmooth min/max objective using log-sum-exp."""
    spacing, dy, xoff = params
    if spacing <= 0:
        return 1.0
    pts = _make_hex(rows, spacing, dy, xoff)
    
    # Wall distances
    wall = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5),
    )
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    d_half = 0.5 * d
    
    # Soft min of wall distances
    wall_min_soft = -np.log(np.sum(np.exp(-beta * wall))) / beta
    
    # Soft min of pairwise distances
    d_flat = d_half.flatten()
    d_flat = d_flat[d_flat > 0]  # remove zeros from diagonal
    if len(d_flat) == 0:
        return -wall_min_soft
    d_min_soft = -np.log(np.sum(np.exp(-beta * d_flat))) / beta
    
    # Soft min of both
    r_soft = -np.log(np.exp(-beta * wall_min_soft) + np.exp(-beta * d_min_soft)) / beta
    return -r_soft


def _soft_repulsion_force(pts, r_target, k_rep=100.0, k_wall=50.0):
    """Compute repulsion forces between circles and wall forces."""
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                # Random small push
                forces[i] += np.random.normal(0, 1e-4, 2)
                forces[j] -= forces[i]
                continue
            if dist < 2 * r_target:
                # Overlapping: strong repulsion
                overlap = 2 * r_target - dist
                push = k_rep * overlap / dist
                forces[i] += push * d
                forces[j] -= push * d
            elif dist < 2.5 * r_target:
                # Near-overlap: mild repulsion to encourage spreading
                overlap = 2.5 * r_target - dist
                push = k_rep * 0.1 * overlap / dist
                forces[i] += push * d
                forces[j] -= push * d
    
    # Wall forces
    for i in range(n):
        for k in range(2):
            if pts[i][k] < -0.5 + r_target:
                forces[i][k] += k_wall * (-0.5 + r_target - pts[i][k])
            elif pts[i][k] > 0.5 - r_target:
                forces[i][k] -= k_wall * (pts[i][k] - (0.5 - r_target))
    
    return forces


def _relax(pts, r_target, max_iter=2000, dt=0.001, deadline=None):
    """Relax positions to eliminate overlaps at given radius using gradient descent."""
    pts = pts.copy()
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        forces = _soft_repulsion_force(pts, r_target)
        pts += dt * forces
        # Clamp to box
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        # Check convergence
        max_overlap = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < 2 * r_target:
                    max_overlap = max(max_overlap, 2 * r_target - dist)
            for k in range(2):
                if pts[i][k] < -0.5 + r_target:
                    max_overlap = max(max_overlap, -0.5 + r_target - pts[i][k])
                elif pts[i][k] > 0.5 - r_target:
                    max_overlap = max(max_overlap, pts[i][k] - (0.5 - r_target))
        
        if max_overlap < 1e-7:
            break
    
    return pts


def _hill_climb(pts, max_iter=500, step_size=0.002, deadline=None):
    """Local hill-climbing: perturb one circle at a time to improve min distance."""
    pts = pts.copy()
    current_score = _score(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        improved = False
        idx = np.random.randint(len(pts))
        orig = pts[idx].copy()
        
        # Try several random perturbations
        for _ in range(20):
            new_pos = orig + np.random.normal(0, step_size, 2)
            # Check bounds
            if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
                continue
            pts[idx] = new_pos
            new_score = _score(pts)
            if new_score > current_score + 1e-9:
                current_score = new_score
                improved = True
                break
            else:
                pts[idx] = orig
        
        if not improved and it > 100:
            # Try smaller steps
            step_size *= 0.9
    
    return pts, current_score


def _high_precision_polish(pts, deadline=None, max_iter=200):
    """High-precision polish: fine-tune each point with very small steps."""
    pts = pts.copy()
    current_score = _score(pts)
    
    # Phase A: Coordinate descent with adaptive step sizes
    step_sizes = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
    for ss in step_sizes:
        if deadline is not None and time.time() > deadline - 3.0:
            break
        improved_any = False
        for idx in range(N):
            if deadline is not None and time.time() > deadline - 3.0:
                break
            orig = pts[idx].copy()
            best_local = current_score
            best_pos = orig.copy()
            # Try 8 directions
            for dx in [-ss, 0, ss]:
                for dy in [-ss, 0, ss]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = orig + np.array([dx, dy])
                    if new_pos[0] < -0.5 or new_pos[0] > 0.5 or new_pos[1] < -0.5 or new_pos[1] > 0.5:
                        continue
                    pts[idx] = new_pos
                    new_score = _score(pts)
                    if new_score > best_local + 1e-12:
                        best_local = new_score
                        best_pos = new_pos.copy()
            if best_pos is not orig:
                pts[idx] = best_pos
                if best_local > current_score:
                    current_score = best_local
                    improved_any = True
        if not improved_any:
            break
    
    # Phase B: Random perturbations with annealing
    if deadline is not None and time.time() < deadline - 3.0:
        for it in range(500):
            if deadline is not None and time.time() > deadline - 3.0:
                break
            idx = np.random.randint(N)
            orig = pts[idx].copy()
            # Decreasing step size
            step = 0.0005 * (1 - it / 600.0)
            if step < 0.00001:
                step = 0.00001
            new_pos = orig + np.random.normal(0, step, 2)
            if new_pos[0] < -0.5 or new_pos[0] > 0.5 or new_pos[1] < -0.5 or new_pos[1] > 0.5:
                continue
            pts[idx] = new_pos
            new_score = _score(pts)
            if new_score > current_score + 1e-12:
                current_score = new_score
            else:
                pts[idx] = orig
    
    # Phase C: Very fine coordinate descent
    if deadline is not None and time.time() < deadline - 3.0:
        for ss in [0.000005, 0.000002, 0.000001]:
            if deadline is not None and time.time() > deadline - 3.0:
                break
            improved_any = False
            for idx in range(N):
                if deadline is not None and time.time() > deadline - 3.0:
                    break
                orig = pts[idx].copy()
                best_local = current_score
                best_pos = orig.copy()
                for dx in [-ss, ss]:
                    for dy in [-ss, ss]:
                        new_pos = orig + np.array([dx, dy])
                        if new_pos[0] < -0.5 or new_pos[0] > 0.5 or new_pos[1] < -0.5 or new_pos[1] > 0.5:
                            continue
                        pts[idx] = new_pos
                        new_score = _score(pts)
                        if new_score > best_local + 1e-13:
                            best_local = new_score
                            best_pos = new_pos.copy()
                if np.any(best_pos != orig):
                    pts[idx] = best_pos
                    if best_local > current_score:
                        current_score = best_local
                        improved_any = True
            if not improved_any:
                break
    
    return pts, current_score


def construct():
    import time
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin under 150s

    best_pts = None
    best_score = -1.0

    # Candidate row-length patterns that sum to 50.
    patterns = [
        [7, 6, 7, 6, 7, 6, 7, 4],   # 50
        [6, 7, 6, 7, 6, 7, 6, 5],   # 50
        [8, 6, 7, 6, 7, 6, 7, 3],   # 50
        [7, 7, 6, 7, 6, 7, 6, 4],   # 50
        [5, 7, 6, 7, 6, 7, 6, 6],   # 50
        [6, 6, 7, 6, 7, 6, 7, 5],   # 50
        [7, 6, 6, 7, 6, 7, 6, 5],   # 50
    ]

    # Phase 1: Differential evolution for global search over lattice parameters
    for rows in patterns:
        if sum(rows) != N:
            continue
        if time.time() > deadline - 30.0:
            break

        max_n = max(rows)
        # Bounds for spacing, dy, xoff
        bounds = [(0.05, 0.2), (-0.1, 0.1), (-0.1, 0.1)]
        
        try:
            de_result = differential_evolution(
                lambda p, r=rows: _soft_neg_score(p, r, beta=30.0),
                bounds=bounds,
                seed=42,
                maxiter=100,
                tol=1e-6,
                popsize=15,
                polish=False
            )
        except Exception:
            continue
        
        if time.time() > deadline - 25.0:
            break
        
        # Refine with Nelder-Mead on the exact (nonsmooth) objective
        def obj_exact(p, r=rows):
            return _neg_score(p, r)
        
        res = minimize(obj_exact, de_result.x, method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-9, "fatol": 1e-11,
                                "adaptive": True})
        
        if time.time() > deadline - 20.0:
            break
        
        s = -res.fun
        if s > best_score:
            best_score = s
            best_pts = _make_hex(rows, *res.x)
        
        # Local polish with L-BFGS-B on a finer grid of starting offsets
        for xoff0 in (-0.03, -0.01, 0.0, 0.01, 0.03):
            for dy0 in (-0.02, 0.0, 0.02):
                if time.time() > deadline - 15.0:
                    break
                init2 = [res.x[0], dy0, xoff0]
                try:
                    res2 = minimize(obj_exact, init2, method="L-BFGS-B",
                                    options={"maxiter": 1000, "gtol": 1e-10})
                    s2 = -res2.fun
                    if s2 > best_score:
                        best_score = s2
                        best_pts = _make_hex(rows, *res2.x)
                except Exception:
                    pass

    # Phase 2: Physics-based refinement: start from best hex config and grow radius
    if best_pts is not None and time.time() < deadline - 30.0:
        r = best_score
        pts = best_pts.copy()
        
        # Try to grow radius using relaxation
        for grow_iter in range(60):
            if time.time() > deadline - 25.0:
                break
            r_new = r * 1.0015 + 1e-8
            pts_relaxed = _relax(pts, r_new, max_iter=1000, dt=0.0003, deadline=deadline)
            s_new = _score(pts_relaxed)
            if s_new >= r_new - 1e-9:
                r = r_new
                pts = pts_relaxed
                if r > best_score:
                    best_score = r
                    best_pts = pts.copy()
            else:
                # Revert and try smaller increment
                r_new = r * 1.0003 + 1e-9
                pts_relaxed = _relax(pts, r_new, max_iter=500, dt=0.0003, deadline=deadline)
                s_new = _score(pts_relaxed)
                if s_new >= r_new - 1e-9:
                    r = r_new
                    pts = pts_relaxed
                    if r > best_score:
                        best_score = r
                        best_pts = pts.copy()
                else:
                    break
        
        # Hill-climb polish
        if time.time() < deadline - 20.0:
            pts_polished, s_polished = _hill_climb(best_pts, max_iter=300, step_size=0.001, deadline=deadline)
            if s_polished > best_score:
                best_score = s_polished
                best_pts = pts_polished

        # High-precision final polish (extended)
        if time.time() < deadline - 8.0:
            pts_final, s_final = _high_precision_polish(best_pts, deadline=deadline, max_iter=500)
            if s_final > best_score:
                best_score = s_final
                best_pts = pts_final
            
            # One more polish pass if time allows
            if time.time() < deadline - 5.0:
                pts_final2, s_final2 = _high_precision_polish(best_pts, deadline=deadline, max_iter=200)
                if s_final2 > best_score:
                    best_score = s_final2
                    best_pts = pts_final2

    if best_pts is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_pts
# EVOLVE-BLOCK-END
