# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 615.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _build_sofa_grid():
    """Return (X, Y) for the sofa grid (450 x 300)."""
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return X, Y


def _densify_path(path):
    """Densify path per evaluator rules: add linear interpolations between consecutive rows."""
    p = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
    dense = [p[0]]
    for i in range(len(p) - 1):
        a1 = p[i, 2]
        a2 = p[i + 1, 2]
        m = max(1, int(np.ceil(abs(a2 - a1) - 1e-9)))
        if m > 1:
            ts = np.linspace(0.0, 1.0, m + 1)
            for t in ts[1:-1]:
                row = p[i] * (1.0 - t) + p[i + 1] * t
                dense.append(row)
        dense.append(p[i + 1])
    return np.array(dense)


def _eval_path_vectorized(path, X, Y, alive_mask=None):
    """Vectorized evaluation with optional active-set tracking."""
    dense = _densify_path(path)
    n_placements = len(dense)
    
    if alive_mask is None:
        alive_mask = np.ones(X.shape, dtype=bool)
    
    n_alive = alive_mask.sum()
    if n_alive == 0:
        return 0
    
    # Extract alive points
    Xa = X[alive_mask]
    Ya = Y[alive_mask]
    
    # Process placements in chunks
    chunk_size = max(1, min(n_placements, 300))
    alive = np.ones(n_alive, dtype=bool)
    
    for start in range(0, n_placements, chunk_size):
        end = min(start + chunk_size, n_placements)
        chunk = dense[start:end]
        n_chunk = end - start
        
        cx = chunk[:, 0]
        cy = chunk[:, 1]
        a_rad = np.deg2rad(chunk[:, 2])
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        
        dx = Xa[None, :] - cx[:, None]
        dy = Ya[None, :] - cy[:, None]
        
        u = c[:, None] * dx + s[:, None] * dy
        v = -s[:, None] * dx + c[:, None] * dy
        
        horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        mask = horiz | vert
        
        chunk_alive = mask.all(axis=0)
        alive &= chunk_alive
        
        if not alive.any():
            break
    
    return int(alive.sum())


def _score(path, X, Y, alive_mask=None):
    return _eval_path_vectorized(path, X, Y, alive_mask) * 0.0001


def _circular_arc_path(cx, cy, r, a_start=0.0, a_end=90.0, n_pts=8):
    """Generate path as points on a circular arc.
    The corridor inner corner follows an arc from start to end orientation.
    """
    if r <= 0.01:
        r = 0.01
    angles = np.linspace(a_start, a_end, n_pts)
    theta = (angles - a_start) * np.pi / 180.0
    
    # For a circular arc, the position is determined by the arc geometry
    # The inner corner traces an arc of radius r
    # Start position should be near (0,0) for a=0, end near (-2,0) for a=90
    # Parametrize: x = cx + r*sin(theta), y = cy - r*cos(theta) adjusted
    
    # Actually, let's use a simpler parametrization:
    # The path goes from angle 0 to 90, and the position follows a smooth curve
    # Let's use: x(t) = x0 + dx*t, y(t) = y0 + dy*t + curvature effect
    # Better: circular arc with center at (cx, cy)
    
    # For a=0, we want position near (0,0). For a=90, near (-2, 0).
    # Arc center at (cx, cy), radius r
    # Start angle phi0, end angle phi1
    # x = cx + r*cos(phi), y = cy + r*sin(phi)
    
    # Let's just use a smooth interpolation with a curvature parameter
    t = np.linspace(0, 1, n_pts)
    x = -0.0 - 2.0 * t  # linear from 0 to -2
    y = 0.0 + 0.3 * np.sin(np.pi * t) * (r / 1.0)  # bump with curvature
    angles = angles
    
    return np.column_stack([x, y, angles])


def _initial_paths():
    """Return multiple initial path candidates."""
    paths = []
    
    # Path 1: Original seed-like path
    paths.append(np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64))
    
    # Path 2: More uniform angle spacing
    angles = np.linspace(10, 80, 7)
    x = np.linspace(-0.1, -0.7, 7)
    y = np.full(7, 0.25)
    paths.append(np.column_stack([x, y, angles]))
    
    # Path 3: Curved path
    t = np.linspace(0, 1, 8)
    angles = 10 + 70 * t
    x = -0.1 - 0.6 * t
    y = 0.1 + 0.3 * np.sin(np.pi * t)
    paths.append(np.column_stack([x, y, angles]))
    
    # Path 4: Steep early, gentle late
    angles = np.array([12, 25, 40, 50, 60, 70, 80, 86])
    x = np.array([-0.1, -0.2, -0.35, -0.45, -0.55, -0.62, -0.68, -0.72])
    y = np.array([0.15, 0.2, 0.25, 0.28, 0.3, 0.28, 0.25, 0.2])
    paths.append(np.column_stack([x, y, angles]))
    
    # Path 5: S-curve
    t = np.linspace(0, 1, 9)
    angles = 10 + 75 * t
    x = -0.1 - 0.6 * t
    y = 0.2 + 0.15 * np.sin(2 * np.pi * t)
    paths.append(np.column_stack([x, y, angles]))
    
    # Path 6: Circular arc model
    paths.append(_circular_arc_path(0.0, 0.0, 1.0, 10, 80, 8))
    
    # Path 7: Another arc
    t = np.linspace(0, 1, 8)
    angles = 10 + 70 * t
    x = -0.05 - 0.65 * t
    y = 0.15 + 0.25 * np.sin(np.pi * t * 0.8)
    paths.append(np.column_stack([x, y, angles]))
    
    return paths


def _nelder_mead(f, x0, max_iter=2000, step=0.05, tol=1e-6):
    """Nelder-Mead minimization."""
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        pt = x0.copy()
        pt[i] += step
        simplex[i + 1] = pt
    vals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 15:
            break
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]
        if abs(vals[0] - vals[-1]) < tol:
            break
        
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                vals[-1] = fe
            else:
                simplex[-1] = xr
                vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr
            vals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc
                vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
    
    best_i = int(np.argmin(vals))
    return simplex[best_i], vals[best_i]


def _optimize_all(path, X, Y, alive_mask, max_iter=1200):
    """Optimize all parameters (x, y, angles) jointly."""
    n = len(path)
    x0 = np.concatenate([path[:, 0], path[:, 1], path[:, 2]])
    
    def f(flat):
        x = flat[0:n]
        y = flat[n:2*n]
        angles = np.clip(flat[2*n:3*n], 1.0, 89.0)
        p = np.column_stack([x, y, angles])
        return -_score(p, X, Y, alive_mask)
    
    xopt, fopt = _nelder_mead(f, x0, max_iter=max_iter, step=0.04)
    x = xopt[0:n]
    y = xopt[n:2*n]
    angles = np.clip(xopt[2*n:3*n], 1.0, 89.0)
    return np.column_stack([x, y, angles])


def _polish(path, X, Y, alive_mask, n_rounds=3):
    """Final high-precision polish with tighter tolerance."""
    current = path.copy()
    for round_idx in range(n_rounds):
        if _elapsed() > _DEADLINE - 5:
            break
        step = 0.01 * (0.5 ** round_idx)
        n = len(current)
        improved = False
        for i in range(n):
            for coord in range(3):
                best_val = _score(current, X, Y, alive_mask)
                best_coord = current[i, coord]
                for delta in [-step, -step/2, step/2, step]:
                    trial = current.copy()
                    trial[i, coord] = current[i, coord] + delta
                    if coord == 2:
                        trial[i, coord] = np.clip(trial[i, coord], 1.0, 89.0)
                    val = _score(trial, X, Y, alive_mask)
                    if val > best_val:
                        best_val = val
                        best_coord = trial[i, coord]
                current[i, coord] = best_coord
                if best_coord != path[i, coord]:
                    improved = True
        if not improved:
            break
    return current


def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n,3) array of corridor inner-corner placements (x,y,a_deg).
    """
    X, Y = _build_sofa_grid()
    alive_mask = np.ones(X.shape, dtype=bool)
    
    # Start with best initial path
    initial_paths = _initial_paths()
    best_path = initial_paths[0]
    best_score = _score(best_path, X, Y, alive_mask)
    
    for p in initial_paths[1:]:
        s = _score(p, X, Y, alive_mask)
        if s > best_score:
            best_score = s
            best_path = p
    
    # Update alive mask
    dense = _densify_path(best_path)
    n_placements = len(dense)
    chunk_size = max(1, min(n_placements, 300))
    alive = np.ones(X.shape, dtype=bool)
    for start in range(0, n_placements, chunk_size):
        end = min(start + chunk_size, n_placements)
        chunk = dense[start:end]
        n_chunk = end - start
        cx = chunk[:, 0]
        cy = chunk[:, 1]
        a_rad = np.deg2rad(chunk[:, 2])
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        Xa = X[alive]
        Ya = Y[alive]
        if Xa.size == 0:
            break
        dx = Xa[None, :] - cx[:, None]
        dy = Ya[None, :] - cy[:, None]
        u = c[:, None] * dx + s[:, None] * dy
        v = -s[:, None] * dx + c[:, None] * dy
        horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        mask = horiz | vert
        chunk_alive = mask.all(axis=0)
        alive_indices = np.where(alive)[0]
        alive[alive_indices[~chunk_alive]] = False
        if not alive.any():
            break
    alive_mask = alive
    
    # Optimize with Nelder-Mead on all parameters
    if _elapsed() < _DEADLINE - 60:
        p_opt = _optimize_all(best_path, X, Y, alive_mask, max_iter=1000)
        s_opt = _score(p_opt, X, Y, alive_mask)
        if s_opt > best_score:
            best_score = s_opt
            best_path = p_opt
            # Update alive mask
            dense = _densify_path(best_path)
            n_placements = len(dense)
            chunk_size = max(1, min(n_placements, 300))
            alive = np.ones(X.shape, dtype=bool)
            for start in range(0, n_placements, chunk_size):
                end = min(start + chunk_size, n_placements)
                chunk = dense[start:end]
                n_chunk = end - start
                cx = chunk[:, 0]
                cy = chunk[:, 1]
                a_rad = np.deg2rad(chunk[:, 2])
                c = np.cos(a_rad)
                s = np.sin(a_rad)
                Xa = X[alive]
                Ya = Y[alive]
                if Xa.size == 0:
                    break
                dx = Xa[None, :] - cx[:, None]
                dy = Ya[None, :] - cy[:, None]
                u = c[:, None] * dx + s[:, None] * dy
                v = -s[:, None] * dx + c[:, None] * dy
                horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
                vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
                mask = horiz | vert
                chunk_alive = mask.all(axis=0)
                alive_indices = np.where(alive)[0]
                alive[alive_indices[~chunk_alive]] = False
                if not alive.any():
                    break
            alive_mask = alive
    
    # Try different angle grids
    if _elapsed() < _DEADLINE - 40:
        angle_grids = [
            np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
            np.array([12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0]),
            np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]),
            np.array([8.0, 18.0, 28.0, 38.0, 48.0, 58.0, 68.0, 78.0, 86.0]),
        ]
        
        for angles_grid in angle_grids:
            if _elapsed() > _DEADLINE - 20:
                break
            n = len(angles_grid)
            x_interp = np.interp(angles_grid, best_path[:, 2], best_path[:, 0])
            y_interp = np.interp(angles_grid, best_path[:, 2], best_path[:, 1])
            p_init = np.column_stack([x_interp, y_interp, angles_grid])
            
            s_init = _score(p_init, X, Y, alive_mask)
            if s_init > best_score:
                best_score = s_init
                best_path = p_init
            
            if _elapsed() > _DEADLINE - 15:
                continue
            p_opt = _optimize_all(p_init, X, Y, alive_mask, max_iter=600)
            s_opt = _score(p_opt, X, Y, alive_mask)
            if s_opt > best_score:
                best_score = s_opt
                best_path = p_opt
    
    # Final polish
    if _elapsed() < _DEADLINE - 3:
        p_polish = _polish(best_path, X, Y, alive_mask, n_rounds=2)
        s_polish = _score(p_polish, X, Y, alive_mask)
        if s_polish > best_score:
            best_score = s_polish
            best_path = p_polish
    
    # Final sanity check
    best_path = np.asarray(best_path, dtype=np.float64)
    if best_path.ndim != 2 or best_path.shape[1] != 3 or best_path.shape[0] < 1:
        best_path = _initial_paths()[0]
    if not np.all(np.isfinite(best_path)):
        best_path = _initial_paths()[0]
    
    return best_path


# EVOLVE-BLOCK-END
