# EVOLVE-BLOCK-START
import numpy as np
import time


def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n, 3) array for the moving sofa problem.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 600s kill

    # Sofa grid points
    sofa_x = np.arange(-3, 1.5, 0.01)
    sofa_y = np.arange(-1.5, 1.5, 0.01)
    sofa_X, sofa_Y = np.meshgrid(sofa_x, sofa_y, indexing='ij')
    sofa_x_flat = sofa_X.ravel()
    sofa_y_flat = sofa_Y.ravel()
    n_sofa = len(sofa_x_flat)

    def _densify(path):
        """Add the evaluator's start/end and interpolate between consecutive rows."""
        full = np.vstack([[[0.0, 0.0, 0.0]], path, [[-2.0, 0.0, 90.0]]])
        result = [full[0].copy()]
        for i in range(len(full) - 1):
            p1, p2 = full[i], full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m):
                t = k / m
                result.append(p1 * (1 - t) + p2 * t)
            result.append(p2.copy())
        return np.array(result)

    def _count_inside(path):
        """Count sofa points inside all placements in the densified path."""
        full = _densify(path)
        inside = np.ones(n_sofa, dtype=bool)
        for px, py, pa in full:
            a_rad = np.deg2rad(pa)
            c, s = np.cos(a_rad), np.sin(a_rad)
            dx = sofa_x_flat - px
            dy = sofa_y_flat - py
            u = c * dx + s * dy
            v = -s * dx + c * dy
            in1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside &= (in1 | in2)
        return np.sum(inside)

    def _score(path):
        return _count_inside(path) * 0.0001

    def _arc_path(h, n_pts):
        """Generate a circular arc path from (0,0,0) to (-2,0,90) with center (-1, h)."""
        R = np.sqrt(1.0 + h * h)
        cx, cy = -1.0, h
        a0 = np.arctan2(-h, 1.0)
        a1 = np.arctan2(-h, -1.0)
        sweep = (a1 - a0) % (2 * np.pi)
        if sweep > np.pi:
            sweep -= 2 * np.pi

        t_vals = np.linspace(0, 1, n_pts + 2)
        path = []
        for t in t_vals:
            angle = a0 + t * sweep
            x = cx + R * np.cos(angle)
            y = cy + R * np.sin(angle)
            a_deg = 90.0 * t
            path.append([x, y, a_deg])
        return np.array(path, dtype=np.float64)

    def _bump_path(amp, n_pts):
        """Generate a path with sinusoidal bump in y."""
        t_vals = np.linspace(0, 1, n_pts + 2)
        path = []
        for t in t_vals:
            x = -2.0 * t
            y = amp * np.sin(np.pi * t)
            a_deg = 90.0 * t
            path.append([x, y, a_deg])
        return np.array(path, dtype=np.float64)

    # --- Phase 1: Coarse search over arc center h ---
    best_score = -1
    best_h = 1.0
    best_path = None

    h_vals = np.linspace(-3, 3, 61)
    for h in h_vals:
        if time.time() > deadline - 120:
            break
        path = _arc_path(h, 9)
        s = _score(path)
        if s > best_score:
            best_score = s
            best_h = h
            best_path = path

    # Also try bump paths
    amp_vals = np.linspace(-2, 2, 41)
    for amp in amp_vals:
        if time.time() > deadline - 120:
            break
        path = _bump_path(amp, 9)
        s = _score(path)
        if s > best_score:
            best_score = s
            best_path = path

    # --- Phase 2: Fine search around best_h ---
    if time.time() < deadline - 60:
        h_fine = np.linspace(best_h - 0.5, best_h + 0.5, 51)
        for h in h_fine:
            if time.time() > deadline - 60:
                break
            path = _arc_path(h, 9)
            s = _score(path)
            if s > best_score:
                best_score = s
                best_h = h
                best_path = path

    # --- Phase 3: Very fine search ---
    if time.time() < deadline - 30:
        h_fine2 = np.linspace(best_h - 0.1, best_h + 0.1, 41)
        for h in h_fine2:
            if time.time() > deadline - 30:
                break
            path = _arc_path(h, 9)
            s = _score(path)
            if s > best_score:
                best_score = s
                best_h = h
                best_path = path

    # --- Phase 4: Optimize number of points ---
    if time.time() < deadline - 20:
        for n_pts in [7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]:
            if time.time() > deadline - 20:
                break
            path = _arc_path(best_h, n_pts)
            s = _score(path)
            if s > best_score:
                best_score = s
                best_path = path

    # --- Phase 5: Local refinement of path coordinates ---
    if best_path is not None and time.time() < deadline - 10:
        # Try small perturbations of intermediate points
        current = best_path.copy()
        n = len(current)
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(1, n - 1):
                for dx in [-0.02, 0.02]:
                    for dy in [-0.02, 0.02]:
                        trial = current.copy()
                        trial[i, 0] += dx
                        trial[i, 1] += dy
                        s = _score(trial)
                        if s > best_score:
                            best_score = s
                            best_path = trial
                            current = trial
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

    if best_path is None:
        best_path = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    if not np.all(np.isfinite(best_path)):
        best_path = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    return best_path


# EVOLVE-BLOCK-END
