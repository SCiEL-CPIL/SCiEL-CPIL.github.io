"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.

Structural change: Use a two-stage approach.
Stage 1: For a fixed aspect ratio, generate a dense set of candidate centers
         (hexagonal grid + random points). Use a greedy algorithm with a
         "potential" heuristic to select N non-overlapping circles that
         maximize the sum of radii. The radius of a candidate is its distance
         to the nearest wall or other selected circle.
Stage 2: Polish the result using a gradient-based method (L-BFGS-B) with
         penalty terms for overlaps and boundary violations.

The aspect ratio is optimized using a coarse grid search followed by
Nelder-Mead refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0


def _check_validity(circles, W, H, tol=1e-9):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > W + tol or y + r > H + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True


def _separate_circles(circles, W, H, max_iter=2000, tol=1e-12):
    """Directly resolve overlaps and boundary violations."""
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False
        # Boundary constraint
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])
        
        # Pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol) ** 2:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True
        
        # Update radii to fit within bounds
        for i in range(n):
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])
        
        if not moved:
            break

    result = np.column_stack([x, y, r])
    valid = _check_validity(result, W, H)
    return result, valid


def _generate_candidates(W, H, n_grid=15, n_random=200, seed=42):
    """Generate candidate centers."""
    rng = np.random.RandomState(seed)
    candidates = []
    
    # Hexagonal grid
    # Estimate a typical radius. For 21 circles in area A, avg r ~ sqrt(A/(21*pi)) * factor
    A = W * H
    r_est = np.sqrt(A / (N * np.pi)) * 0.9
    
    dx = 2 * r_est * 1.05
    dy = 2 * r_est * np.sqrt(3) / 2 * 1.05
    
    row = 0
    y_pos = r_est * 0.8
    while y_pos <= H - r_est * 0.5 + 1e-9:
        offset = 0.0 if row % 2 == 0 else dx / 2
        x_pos = r_est * 0.8 + offset
        while x_pos <= W - r_est * 0.5 + 1e-9:
            candidates.append((x_pos, y_pos))
            x_pos += dx
        y_pos += dy
        row += 1
        
    # Add uniform grid points for robustness
    for i in range(n_grid):
        for j in range(n_grid):
            cx = (i + 0.5) * W / n_grid
            cy = (j + 0.5) * H / n_grid
            candidates.append((cx, cy))
            
    # Add random points
    for _ in range(n_random):
        cx = rng.uniform(0.05, W - 0.05)
        cy = rng.uniform(0.05, H - 0.05)
        candidates.append((cx, cy))
        
    # Boundary points
    for i in range(1, n_grid):
        cx = i * W / n_grid
        candidates.append((cx, 0.05))
        candidates.append((cx, H - 0.05))
        cy = i * H / n_grid
        candidates.append((0.05, cy))
        candidates.append((W - 0.05, cy))

    return np.array(candidates)


def _greedy_pack(W, H, seed=42):
    """
    Greedy packing.
    We select circles one by one. At each step, we evaluate the "potential"
    radius of every remaining candidate, given the already selected circles.
    We pick the candidate with the largest potential radius.
    To improve diversity, we can also use a temperature-like parameter,
    but for now we stick to greedy max.
    """
    candidates = _generate_candidates(W, H, seed=seed)
    if len(candidates) == 0:
        return None
        
    selected = []
    remaining_idx = list(range(len(candidates)))
    
    for _ in range(N):
        if not remaining_idx:
            break
            
        best_idx = -1
        best_r = -1
        best_pos = None
        
        # Vectorized calculation of potential radii is hard due to dynamic exclusion,
        # but N is small (21) and candidates are ~500.
        # Loop is fine.
        
        for idx in remaining_idx:
            cx, cy = candidates[idx]
            # Distance to walls
            r_wall = min(cx, W - cx, cy, H - cy)
            if r_wall <= 0:
                continue
                
            # Distance to selected circles
            r_min = r_wall
            for (sx, sy, sr) in selected:
                dist = np.sqrt((cx - sx)**2 + (cy - sy)**2)
                r_dist = dist - sr
                if r_dist < r_min:
                    r_min = r_dist
                    if r_min <= 0:
                        break
                        
            if r_min > best_r:
                best_r = r_min
                best_idx = idx
                best_pos = (cx, cy)
                
        if best_r <= 1e-6:
            # Try to find any spot, even tiny
            for idx in remaining_idx:
                cx, cy = candidates[idx]
                r_wall = min(cx, W - cx, cy, H - cy)
                if r_wall <= 0: continue
                r_min = r_wall
                for (sx, sy, sr) in selected:
                    dist = np.sqrt((cx - sx)**2 + (cy - sy)**2)
                    r_dist = dist - sr
                    if r_dist < r_min:
                        r_min = r_dist
                        break
                if r_min > 1e-6:
                    best_r = r_min
                    best_idx = idx
                    best_pos = (cx, cy)
                    break
            
            if best_r <= 1e-6:
                # Fallback: random search for a tiny circle
                found = False
                for _ in range(100):
                    cx = np.random.uniform(0.01, W-0.01)
                    cy = np.random.uniform(0.01, H-0.01)
                    r_wall = min(cx, W - cx, cy, H - cy)
                    r_min = r_wall
                    for (sx, sy, sr) in selected:
                        dist = np.sqrt((cx - sx)**2 + (cy - sy)**2)
                        r_dist = dist - sr
                        if r_dist < r_min:
                            r_min = r_dist
                            break
                    if r_min > 1e-4:
                        best_r = r_min
                        best_pos = (cx, cy)
                        found = True
                        break
                if not found:
                    break
                    
        selected.append((best_pos[0], best_pos[1], best_r))
        remaining_idx.remove(best_idx)
        
    if len(selected) < N:
        return None
        
    return np.array(selected)


def _polish(circles, W, H, max_iter=5000):
    """Polish using L-BFGS-B with penalty method."""
    n = N
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    params = np.concatenate([x, y, r])
    
    r_max = min(W, H) / 2
    bounds = []
    for i in range(n):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    def _obj_and_grad(params, W, H, eps, weight):
        n = N
        x = params[0:n]
        y = params[n:2*n]
        r = params[2*n:3*n]
        
        obj = -np.sum(r)
        grad = np.zeros_like(params)
        grad[2*n:3*n] = -1.0

        def _soft_max(v, scale):
            v_safe = np.clip(v / scale, -50, 50)
            smax = scale * np.log1p(np.exp(v_safe))
            sig = 1.0 / (1.0 + np.exp(-v_safe))
            return smax, sig

        # Boundary penalties
        # r - x > 0  => x < r
        # x + r - W > 0 => x > W - r
        # r - y > 0
        # y + r - H > 0
        
        for (vi, gi) in [(r - x, 0), (r - y, n), (x + r - W, 0), (y + r - H, n), (-r, 2*n)]:
            smax, gsmax = _soft_max(vi, eps)
            obj += weight * np.sum(smax ** 2)
            if gi == 0:
                grad[0:n] += -2.0 * weight * smax * gsmax
                grad[2*n:3*n] += 2.0 * weight * smax * gsmax
            elif gi == n:
                grad[n:2*n] += -2.0 * weight * smax * gsmax
                grad[2*n:3*n] += 2.0 * weight * smax * gsmax
            else:
                grad[2*n:3*n] += -2.0 * weight * smax * gsmax

        # Overlap penalties
        dx_mat = x[:, None] - x[None, :]
        dy_mat = y[:, None] - y[None, :]
        dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
        np.fill_diagonal(dist_mat, 1e12)
        dist_mat_safe = np.where(dist_mat < 1e-12, 1e-12, dist_mat)
        iu, ju = np.triu_indices(n, k=1)
        dist_ij = dist_mat_safe[iu, ju]
        dx_ij = dx_mat[iu, ju]
        dy_ij = dy_mat[iu, ju]
        r_sum = r[iu] + r[ju]
        v_vec = r_sum - dist_ij
        smax_vec, gsmax_vec = _soft_max(v_vec, eps)
        obj += weight * np.sum(smax_vec ** 2)
        factor_vec = 2.0 * weight * smax_vec * gsmax_vec
        nx_ij = -dx_ij / dist_ij
        ny_ij = -dy_ij / dist_ij
        np.add.at(grad, iu, factor_vec * nx_ij)
        np.add.at(grad, n + iu, factor_vec * ny_ij)
        np.add.at(grad, 2*n + iu, factor_vec)
        np.add.at(grad, ju, -factor_vec * nx_ij)
        np.add.at(grad, n + ju, -factor_vec * ny_ij)
        np.add.at(grad, 2*n + ju, factor_vec)
        
        return obj, grad

    # Multi-stage polish
    stages = [
        (1e-3, 100.0),
        (5e-4, 200.0),
        (1e-4, 400.0),
        (5e-5, 800.0),
        (1e-5, 1500.0),
    ]
    
    for eps, weight in stages:
        result = minimize(_obj_and_grad, params, args=(W, H, eps, weight),
                          method='L-BFGS-B', jac=True, bounds=bounds,
                          options={'maxiter': max_iter // len(stages), 'ftol': 1e-12, 'gtol': 1e-10})
        params = result.x
        
    return params.reshape(N, 3)


def _evaluate_aspect(ar, seed=42):
    """Evaluate packing for a given aspect ratio."""
    W = 2.0 * ar / (ar + 1.0)
    H = 2.0 / (ar + 1.0)
    
    circles = _greedy_pack(W, H, seed=seed)
    if circles is None:
        return -np.inf, None, W, H
        
    # Ensure validity
    circles, valid = _separate_circles(circles, W, H, max_iter=500)
    if not valid:
        return -np.inf, None, W, H
        
    score = np.sum(circles[:, 2])
    return score, circles, W, H


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W = 1.0
    best_H = 1.0
    
    # Phase 1: Coarse grid search over aspect ratio
    # Perimeter 4 => 2(W+H)=4 => W+H=2.
    # ar = W/H. W = 2*ar/(ar+1), H = 2/(ar+1).
    # Square is ar=1.
    # Try a range of aspect ratios.
    ar_range = np.linspace(0.6, 1.6, 21)
    
    for ar in ar_range:
        if time.time() - start_time > DEADLINE - 40:
            break
        score, circles, W, H = _evaluate_aspect(ar, seed=42)
        if score > best_score:
            best_score = score
            best_result = circles
            best_W, best_H = W, H
            
    # Phase 2: Refine aspect ratio around the best one
    if best_result is not None:
        # Find the best ar from the grid
        best_ar = best_W / best_H
        # Local search around best_ar
        local_ar_range = np.linspace(best_ar - 0.1, best_ar + 0.1, 11)
        for ar in local_ar_range:
            if time.time() - start_time > DEADLINE - 20:
                break
            if ar <= 0.1: continue
            score, circles, W, H = _evaluate_aspect(ar, seed=42)
            if score > best_score:
                best_score = score
                best_result = circles
                best_W, best_H = W, H
                
    # Phase 3: Polish the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 10:
        polished = _polish(best_result, best_W, best_H, max_iter=5000)
        polished = np.clip(polished, 1e-6, None)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 1e-6, min(best_W, best_H)/2)
        
        polished, valid = _separate_circles(polished, best_W, best_H, max_iter=1000)
        if valid:
            score = np.sum(polished[:, 2])
            if score > best_score:
                best_score = score
                best_result = polished

    # Fallback
    if best_result is None:
        # Create a simple valid packing
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0
        # Ensure validity for fallback
        best_result, valid = _separate_circles(best_result, best_W, best_H, max_iter=1000)
        if not valid:
            # Even simpler fallback
            r = 0.1
            best_result = np.array([[0.5, 0.5, r]])
            # This is invalid if N!=1, but we must return something.
            # Let's try to make a valid 21-circle packing.
            # 5x5 grid in 1x1 box, r=0.09
            cells = []
            for i in range(5):
                for j in range(5):
                    if len(cells) >= N: break
                    cells.append([0.1 + i*0.2, 0.1 + j*0.2, 0.09])
            best_result = np.array(cells[:N])

    # Final validity check
    if not _check_validity(best_result, best_W, best_H):
        # Force separation
        best_result, valid = _separate_circles(best_result, best_W, best_H, max_iter=5000)
        if not valid:
             # Last resort: tiny circles in a grid
             r = 0.01
             cells = []
             for i in range(5):
                 for j in range(5):
                     if len(cells) >= N: break
                     cells.append([0.1 + i*0.2, 0.1 + j*0.2, r])
             best_result = np.array(cells[:N])
             best_W, best_H = 1.0, 1.0

    return best_result
