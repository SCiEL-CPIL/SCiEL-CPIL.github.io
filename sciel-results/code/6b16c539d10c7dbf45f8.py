"""Circle packing in unit square n=32: maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import cdist
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    best_score = -np.inf
    best_result = None

    def _elapsed():
        return time.time() - start_time

    def _is_valid(arr, tol=1e-9):
        """Check if circles are valid: inside [0,1]^2 and non-overlapping."""
        x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
        if np.any(r <= 0):
            return False
        if np.any(x < -tol) or np.any(x > 1 + tol):
            return False
        if np.any(y < -tol) or np.any(y > 1 + tol):
            return False
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        # Check pairwise distances
        D = cdist(arr[:, :2], arr[:, :2])
        R = r[:, None] + r[None, :]
        if np.any(D < R - 1e-7):
            return False
        return True

    def _sum_r(arr):
        return np.sum(arr[:, 2])

    def _hex_init():
        """Create hexagonal lattice initialization."""
        points = []
        for row in range(10):
            for col in range(10):
                x = col + (0.5 if row % 2 else 0)
                y = row * np.sqrt(3) / 2
                points.append((x, y))
        pts = np.array(points)
        center = pts.mean(axis=0)
        pts = pts - center
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        init_pts = pts[idx].copy()
        min_x, min_y = init_pts.min(axis=0)
        max_x, max_y = init_pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        scale = 0.9 / max(width, height)
        init_pts[:, 0] = (init_pts[:, 0] - min_x) * scale
        init_pts[:, 1] = (init_pts[:, 1] - min_y) * scale
        current_w = init_pts.max(axis=0) - init_pts.min(axis=0)
        offset_x = (1.0 - current_w[0]) / 2
        offset_y = (1.0 - current_w[1]) / 2
        init_pts[:, 0] += offset_x
        init_pts[:, 1] += offset_y
        return init_pts

    def _rect_init():
        """Create rectangular grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])

    def _random_init(seed):
        """Create random initialization."""
        rng = np.random.RandomState(seed)
        return rng.uniform(0.1, 0.9, size=(N, 2))

    def _fast_radii(pos, iters=50, tol=1e-10):
        """Iteratively compute max radii for fixed positions."""
        rad = np.full(N, 0.04)
        for _ in range(iters):
            changed = False
            for i in range(N):
                # Max radius limited by walls
                r_wall = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
                # Max radius limited by other circles
                r_other = np.inf
                for j in range(N):
                    if i == j:
                        continue
                    d = np.linalg.norm(pos[i] - pos[j])
                    r_other = min(r_other, d / 2)
                new_r = min(r_wall, r_other)
                if new_r < 1e-8:
                    new_r = 1e-8
                if abs(new_r - rad[i]) > tol:
                    changed = True
                rad[i] = new_r
            if not changed:
                break
        return rad

    def _lp_radii(pos):
        """
        Use LP to maximize sum of radii for fixed positions.
        Constraints:
        r_i >= 0
        r_i <= wall_dist_i
        r_i + r_j <= d_ij
        """
        n_vars = N
        # Objective: maximize sum(r) -> minimize -sum(r)
        c = -np.ones(n_vars)
        
        # Inequality constraints A_ub @ x <= b_ub
        A_rows = []
        b_rows = []
        
        # Wall constraints: r_i <= wall_dist_i
        wall_dist = np.minimum(np.minimum(pos[:, 0], 1 - pos[:, 0]),
                               np.minimum(pos[:, 1], 1 - pos[:, 1]))
        # r_i <= wall_dist_i  =>  r_i - 0 <= wall_dist_i
        for i in range(N):
            row = np.zeros(N)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(wall_dist[i])
        
        # Pairwise: r_i + r_j <= d_ij
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                row = np.zeros(N)
                row[i] = 1.0
                row[j] = 1.0
                A_rows.append(row)
                b_rows.append(d)
        
        if not A_rows:
            return np.zeros(N)
            
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        bounds = [(0, None)] * N
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x
            else:
                return _fast_radii(pos)
        except Exception:
            return _fast_radii(pos)

    def _push_apart(pos, rad, time_budget=2.0):
        """
        Local search: repeatedly pick the most overlapping pair,
        push them apart slightly, and re-optimize radii.
        """
        current_pos = pos.copy()
        current_rad = rad.copy()
        current_score = _sum_r(current_rad)
        best_pos = current_pos.copy()
        best_rad = current_rad.copy()
        best_score = current_score
        
        start = time.time()
        rng = np.random.RandomState(42)
        step_size = 0.01
        no_improve_count = 0
        
        while (time.time() - start) < time_budget:
            # Find worst overlap
            D = cdist(current_pos, current_pos)
            R = current_rad[:, None] + current_rad[None, :]
            overlap = R - D
            # Find max positive overlap
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            overlap_masked = np.where(mask, overlap, -np.inf)
            i, j = np.unravel_index(np.argmax(overlap_masked), (N, N))
            
            if overlap[i, j] < 1e-7:
                break # No significant overlap
            
            # Push apart
            dir_vec = current_pos[i] - current_pos[j]
            d = np.linalg.norm(dir_vec)
            if d < 1e-10:
                dir_vec = np.array([1.0, 0.0])
                d = 1.0
            else:
                dir_vec = dir_vec / d
            
            move = step_size * 0.5
            new_pos = current_pos.copy()
            new_pos[i] += dir_vec * move
            new_pos[j] -= dir_vec * move
            
            # Clip to bounds
            new_pos = np.clip(new_pos, 0.01, 0.99)
            
            # Optimize radii
            new_rad = _lp_radii(new_pos)
            new_score = _sum_r(new_rad)
            
            # Check validity (LP should ensure it, but just in case)
            if not _is_valid(np.column_stack([new_pos, new_rad])):
                continue
                
            if new_score > current_score:
                current_pos = new_pos
                current_rad = new_rad
                current_score = new_score
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_pos = current_pos.copy()
                    best_rad = current_rad.copy()
            else:
                no_improve_count += 1
                # Random perturbation to escape local min
                if no_improve_count > 20:
                    idx = rng.randint(0, N)
                    noise = rng.normal(0, step_size, size=2)
                    current_pos[idx] += noise
                    current_pos[idx] = np.clip(current_pos[idx], 0.01, 0.99)
                    current_rad = _lp_radii(current_pos)
                    current_score = _sum_r(current_rad)
                    if not _is_valid(np.column_stack([current_pos, current_rad])):
                        current_pos = best_pos.copy()
                        current_rad = best_rad.copy()
                        current_score = best_score
                    no_improve_count = 0
                    step_size *= 0.95 # Anneal slightly
                    if step_size < 0.001:
                        break
        
        return best_pos, best_rad, best_score

    def _objective(vars_, penalty_weight=1000.0):
        """Objective: minimize -sum(radii) + penalty for violations."""
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        obj = -np.sum(R)
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = X[i] - X[j]
                dy = Y[i] - Y[j]
                d = np.sqrt(dx * dx + dy * dy)
                overlap = R[i] + R[j] - d
                if overlap > 0:
                    pen += overlap ** 2
            if X[i] - R[i] < 0:
                pen += (X[i] - R[i]) ** 2
            if 1.0 - X[i] - R[i] < 0:
                pen += (1.0 - X[i] - R[i]) ** 2
            if Y[i] - R[i] < 0:
                pen += (Y[i] - R[i]) ** 2
            if 1.0 - Y[i] - R[i] < 0:
                pen += (1.0 - Y[i] - R[i]) ** 2
        return obj + penalty_weight * pen

    def _objective_grad(vars_, penalty_weight=1000.0):
        """Analytical gradient of the objective."""
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        grad = np.zeros_like(vars_)
        grad[2::3] = -1.0
        weight = penalty_weight
        for i in range(N):
            for j in range(i + 1, N):
                dx = X[i] - X[j]
                dy = Y[i] - Y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-10:
                    continue
                overlap = R[i] + R[j] - d
                if overlap > 0:
                    factor = 2.0 * overlap
                    grad[3 * i] += weight * factor * (-dx / d)
                    grad[3 * i + 1] += weight * factor * (-dy / d)
                    grad[3 * j] += weight * factor * (dx / d)
                    grad[3 * j + 1] += weight * factor * (dy / d)
                    grad[3 * i + 2] += weight * factor
                    grad[3 * j + 2] += weight * factor
            val = X[i] - R[i]
            if val < 0:
                grad[3 * i] += weight * 2.0 * val
                grad[3 * i + 2] += weight * -2.0 * val
            val = 1.0 - X[i] - R[i]
            if val < 0:
                grad[3 * i] += weight * -2.0 * val
                grad[3 * i + 2] += weight * -2.0 * val
            val = Y[i] - R[i]
            if val < 0:
                grad[3 * i + 1] += weight * 2.0 * val
                grad[3 * i + 2] += weight * -2.0 * val
            val = 1.0 - Y[i] - R[i]
            if val < 0:
                grad[3 * i + 1] += weight * -2.0 * val
                grad[3 * i + 2] += weight * -2.0 * val
        return grad

    def _make_valid(arr):
        """Ensure validity by scaling down radii if necessary."""
        arr = arr.copy()
        for _ in range(100):
            if _is_valid(arr):
                return arr
            arr[:, 2] *= 0.95
        # Last resort: tiny radii
        arr[:, 2] = 0.001
        return arr

    # --- Main Execution Loop ---

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-5, 0.5))

    inits = []
    inits.append(('hex', _hex_init()))
    inits.append(('rect', _rect_init()))
    for seed in range(20):
        inits.append(('rand', _random_init(seed + 100)))

    for idx, (init_type, init_pts) in enumerate(inits):
        if _elapsed() > deadline - 5:
            break

        # Initial radii estimate
        init_rad = _fast_radii(init_pts, iters=10)
        
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = init_pts[i, 0]
            x0[3 * i + 1] = init_pts[i, 1]
            x0[3 * i + 2] = init_rad[i]

        # Phase 1: Global Optimization (L-BFGS-B)
        # Use fewer iterations for random inits to save time
        maxiter = 150 if idx < 2 else 80
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', jac=_objective_grad,
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-6})
            x_opt = res.x
        except Exception:
            x_opt = x0

        X = x_opt[0::3].copy()
        Y = x_opt[1::3].copy()
        R = x_opt[2::3].copy()
        pos = np.column_stack((X, Y))

        # Phase 2: LP Optimization of Radii for fixed centers
        R_lp = _lp_radii(pos)
        if _is_valid(np.column_stack([pos, R_lp])):
            R = R_lp
        else:
            R = _fast_radii(pos)
            if not _is_valid(np.column_stack([pos, R])):
                continue # Skip this init if we can't even make it valid

        score = _sum_r(R)
        
        # Phase 3: Push-apart Local Search
        if _elapsed() < deadline - 5 and score > -np.inf:
            ls_time = 2.0 if idx < 2 else 1.0
            pos_ls, R_ls, score_ls = _push_apart(pos, R, time_budget=ls_time)
            if score_ls > score:
                pos = pos_ls
                R = R_ls
                score = score_ls
        
        # Final LP pass
        if _elapsed() < deadline - 2:
            R_final = _lp_radii(pos)
            if _is_valid(np.column_stack([pos, R_final])):
                R = R_final
                score = _sum_r(R)

        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, R])

    # If no valid result found, return grid fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])

    # Final Polish: One more LP pass on the best result
    if _elapsed() < deadline - 1:
        pos = best_result[:, :2].copy()
        R_lp = _lp_radii(pos)
        if _is_valid(np.column_stack([pos, R_lp])):
            new_score = _sum_r(R_lp)
            if new_score > best_score:
                best_score = new_score
                best_result = np.column_stack([pos, R_lp])

    # Final validity check
    if not _is_valid(best_result):
        best_result = _make_valid(best_result)

    return best_result

BET: {mechanism: "Replaces heuristic radius polishing with exact LP optimization for fixed centers, enabling tighter packing of radii within the current position landscape, combined with a targeted 'push-apart' local search to resolve overlaps and improve positions.", q: 0.75, delta: [0.01, 0.05]}
