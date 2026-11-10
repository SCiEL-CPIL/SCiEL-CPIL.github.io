# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    """
    Find a trajectory for moving an L-shaped sofa from (0,0,0) to (-2,0,90).
    Uses Nelder-Mead on a coarse subgrid, then verifies on the full grid.
    """
    start_time = time.time()
    deadline = start_time + 620  # Safety margin before 630s kill

    # Sofa grid points
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    gx, gy = X.ravel(), Y.ravel()
    n_points = len(gx)

    # Coarse subgrid for fast evaluation during optimization
    step = 3
    xs_sub = np.arange(-3.0, 1.5, 0.01 * step)
    ys_sub = np.arange(-1.5, 1.5, 0.01 * step)
    Xs, Ys = np.meshgrid(xs_sub, ys_sub, indexing='ij')
    gx_sub, gy_sub = Xs.ravel(), Ys.ravel()
    n_points_sub = len(gx_sub)

    def densify_trajectory(trajectory):
        """Densify trajectory as the evaluator does."""
        full = np.vstack([[0.0, 0.0, 0.0], trajectory, [-2.0, 0.0, 90.0]])
        dense_parts = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m + 1):
                t = k / m
                interp = p1 + t * (p2 - p1)
                dense_parts.append(interp)
        return np.array(dense_parts)

    def evaluate_trajectory(trajectory, use_coarse=False):
        """
        Evaluate trajectory: return number of sofa points inside ALL placements.
        If use_coarse, use the coarse subgrid for speed.
        """
        if use_coarse:
            gx_c, gy_c = gx_sub, gy_sub
        else:
            gx_c, gy_c = gx, gy

        dense = densify_trajectory(trajectory)
        n_dense = len(dense)
        n_pts = len(gx_c)

        # We'll check in batches to manage memory
        batch_size = min(n_dense, 20)
        mask = np.ones(n_pts, dtype=bool)

        for batch_start in range(0, n_dense, batch_size):
            batch_end = min(batch_start + batch_size, n_dense)
            batch = dense[batch_start:batch_end]

            xs_b = batch[:, 0]
            ys_b = batch[:, 1]
            as_rad = np.deg2rad(batch[:, 2])
            cos_a = np.cos(as_rad)
            sin_a = np.sin(as_rad)

            # For each point and each placement in batch
            gx_r = gx_c.reshape(-1, 1)  # (n_pts, 1)
            gy_r = gy_c.reshape(-1, 1)
            xs_r = xs_b.reshape(1, -1)  # (1, n_batch)
            ys_r = ys_b.reshape(1, -1)
            cos_r = cos_a.reshape(1, -1)
            sin_r = sin_a.reshape(1, -1)

            dx = gx_r - xs_r  # (n_pts, n_batch)
            dy = gy_r - ys_r

            u = cos_r * dx + sin_r * dy
            v = -sin_r * dx + cos_r * dy

            eps = 0.001
            arm1 = (u >= -eps) & (u <= 1.0 + eps) & (v >= -3.0 - eps) & (v <= 1.0 + eps)
            arm2 = (u >= -3.0 - eps) & (u <= 1.0 + eps) & (v >= -eps) & (v <= 1.0 + eps)
            in_place = arm1 | arm2

            batch_mask = in_place.all(axis=1)
            mask &= batch_mask

            if not mask.any():
                return 0

        return int(mask.sum())

    def params_to_traj(params, n_ctrl):
        """Convert parameter vector to trajectory with fixed endpoints."""
        x = np.zeros(n_ctrl)
        y = np.zeros(n_ctrl)
        angle = np.zeros(n_ctrl)
        x[0], y[0], angle[0] = 0.0, 0.0, 0.0
        x[-1], y[-1], angle[-1] = -2.0, 0.0, 90.0
        for i in range(1, n_ctrl - 1):
            x[i] = params[3*(i-1)]
            y[i] = params[3*(i-1)+1]
            angle[i] = params[3*(i-1)+2]
        return np.column_stack([x, y, angle])

    def traj_to_params(traj, n_ctrl):
        """Convert trajectory to parameter vector (interior points only)."""
        params = []
        for i in range(1, n_ctrl - 1):
            params.extend([traj[i, 0], traj[i, 1], traj[i, 2]])
        return np.array(params)

    def objective_coarse(params):
        """Negative score on coarse grid (we minimize)."""
        if time.time() > deadline - 30:
            return 1e10
        traj = params_to_traj(params, n_ctrl)
        score = evaluate_trajectory(traj, use_coarse=True)
        return -score

    def objective_full(params):
        """Negative score on full grid (we minimize)."""
        if time.time() > deadline - 15:
            return 1e10
        traj = params_to_traj(params, n_ctrl)
        score = evaluate_trajectory(traj, use_coarse=False)
        return -score

    # Initial best
    best_score = 0
    best_traj = None

    # Search with parametric paths
    n_ctrl = 8  # Number of control points including endpoints
    n_params = 3 * (n_ctrl - 2)  # 18 parameters

    # Try several initial path shapes
    path_shapes = [
        # Arc paths with varying heights
        lambda t: (-0.25*t, 0.3*np.sin(np.pi*t), 11.25*t),
        lambda t: (-0.25*t, 0.4*np.sin(np.pi*t), 11.25*t),
        lambda t: (-0.25*t, 0.5*np.sin(np.pi*t), 11.25*t),
        lambda t: (-0.25*t, 0.35*np.sin(np.pi*t), 11.25*t),
        # S-curves
        lambda t: (-0.25*t + 0.05*np.sin(2*np.pi*t), 0.4*np.sin(np.pi*t), 11.25*t),
        lambda t: (-0.25*t - 0.05*np.sin(2*np.pi*t), 0.4*np.sin(np.pi*t), 11.25*t),
        # More complex
        lambda t: (-0.25*t, 0.45*np.sin(np.pi*t) + 0.05*np.sin(2*np.pi*t), 11.25*t),
        lambda t: (-0.25*t, 0.35*np.sin(np.pi*t) - 0.05*np.sin(2*np.pi*t), 11.25*t),
    ]

    for shape_idx, shape_fn in enumerate(path_shapes):
        if time.time() > deadline - 60:
            break

        t = np.linspace(0, 1, n_ctrl)
        x, y, angle = shape_fn(t)
        x[0], y[0], angle[0] = 0.0, 0.0, 0.0
        x[-1], y[-1], angle[-1] = -2.0, 0.0, 90.0

        traj = np.column_stack([x, y, angle])
        params0 = traj_to_params(traj, n_ctrl)

        # Evaluate initial on coarse grid
        score_coarse = evaluate_trajectory(traj, use_coarse=True)
        if score_coarse == 0:
            continue

        # Nelder-Mead on coarse grid
        result = minimize(
            objective_coarse,
            params0,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 0.01, 'fatol': 1.0, 'disp': False}
        )
        params_coarse = result.x
        traj_coarse = params_to_traj(params_coarse, n_ctrl)
        score_coarse_best = evaluate_trajectory(traj_coarse, use_coarse=True)

        # Evaluate on full grid
        full_score = evaluate_trajectory(traj_coarse, use_coarse=False)

        if full_score > best_score:
            best_score = full_score
            best_traj = traj_coarse.copy()

            # Refine on full grid with Nelder-Mead
            if time.time() < deadline - 30:
                result2 = minimize(
                    objective_full,
                    params_coarse,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 0.005, 'fatol': 1.0, 'disp': False}
                )
                params_full = result2.x
                traj_full = params_to_traj(params_full, n_ctrl)
                score_full = evaluate_trajectory(traj_full, use_coarse=False)

                if score_full > best_score:
                    best_score = score_full
                    best_traj = traj_full.copy()

    # If no valid trajectory found, return minimal valid one
    if best_traj is None:
        # Try a simple linear path
        n_simple = 5
        t = np.linspace(0, 1, n_simple)
        x = -0.5*t
        y = 0.3*np.sin(np.pi*t)
        angle = 22.5*t
        x[0], y[0], angle[0] = 0.0, 0.0, 0.0
        x[-1], y[-1], angle[-1] = -2.0, 0.0, 90.0
        best_traj = np.column_stack([x, y, angle])

    return best_traj

# EVOLVE-BLOCK-END
