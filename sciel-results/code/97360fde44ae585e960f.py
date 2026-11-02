"""Circle packing in unit square n=32: maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    best_score = -np.inf
    best_result = None

    def _make_hex_init():
        """Create hexagonal lattice initialization."""
        points = []
        for row in range(12):
            for col in range(12):
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
        scale = 0.95 / max(width, height)
        init_pts[:, 0] = (init_pts[:, 0] - min_x) * scale
        init_pts[:, 1] = (init_pts[:, 1] - min_y) * scale
        current_w = init_pts.max(axis=0) - init_pts.min(axis=0)
        offset_x = (1.0 - current_w[0]) / 2
        offset_y = (1.0 - current_w[1]) / 2
        init_pts[:, 0] += offset_x
        init_pts[:, 1] += offset_y
        return init_pts

    def _make_grid_init():
        """Create grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([0.5/side + i/side, 0.5/side + j/side])
        return np.array(centers[:N])

    def _make_perturbed_grid_init(seed):
        """Create perturbed grid initialization to break symmetry."""
        rng = np.random.RandomState(seed)
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    cx = 0.5/side + i/side
                    cy = 0.5/side + j/side
                    # Add small random perturbation
                    cx += rng.uniform(-0.02, 0.02)
                    cy += rng.uniform(-0.02, 0.02)
                    centers.append([cx, cy])
        return np.array(centers[:N])

    def _make_random_init(seed):
        """Create random initialization."""
        rng = np.random.RandomState(seed)
        return rng.uniform(0.1, 0.9, size=(N, 2))

    def _compute_initial_radius(pts):
        """Compute initial radius based on minimum pairwise distance."""
        # Vectorized min distance
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        min_dist = np.min(min_dists)
        return min_dist / 2 * 0.8

    def _objective_and_grad(vars_):
        """
        Objective: Minimize -sum(radii) + penalty for violations.
        Uses vectorized operations for speed.
        vars_ : (3*N,) array [x0, y0, r0, x1, y1, r1, ...]
        """
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        
        # Base objective: maximize sum of radii -> minimize -sum(R)
        obj = -np.sum(R)
        grad = np.zeros_like(vars_)
        grad[2::3] = -1.0
        
        # Vectorized pairwise differences
        # D: (N, N, 2)
        DX = X[:, np.newaxis] - X[np.newaxis, :]
        DY = Y[:, np.newaxis] - Y[np.newaxis, :]
        D2 = DX**2 + DY**2
        # Avoid division by zero
        D2_safe = np.where(D2 < 1e-12, 1e-12, D2)
        D = np.sqrt(D2_safe)
        
        # Overlap: R_i + R_j - D_ij
        # We want to penalize positive overlaps
        # Using a smooth approximation: penalty = sum(softplus(overlap)^2) or similar
        # For stability, we use a quadratic penalty on the violation amount
        # violation_ij = max(0, R_i + R_j - D_ij)
        
        R_sum = R[:, np.newaxis] + R[np.newaxis, :]
        overlap = R_sum - D
        
        # Only penalize where overlap > 0
        # To make it differentiable, we can use a soft-max or just clip and rely on L-BFGS-B
        # A better approach for gradient:
        # grad_penalty = 2 * weight * overlap * (direction)
        
        weight = 1000.0
        # Mask for active constraints (overlap > 0)
        active = overlap > 0
        
        # Penalty term
        pen = np.sum(np.where(active, overlap**2, 0))
        obj += weight * pen
        
        # Gradients for penalty
        # d(pen)/d(R_i) = 2 * sum_j (overlap_ij * active_ij)
        # d(pen)/d(X_i) = 2 * sum_j (overlap_ij * active_ij * (-DX_ij / D_ij))
        
        # Sum over j for each i
        # Note: D is symmetric, DX is antisymmetric.
        # We need to handle the diagonal (i=j) which is 0 overlap.
        
        # grad_R
        grad_R_pen = 2.0 * weight * np.sum(overlap * active, axis=1)
        
        # grad_X, grad_Y
        # Factor = 2 * weight * overlap * active / D
        factor = 2.0 * weight * overlap * active / D_safe
        # Set diagonal to 0 to avoid self-interaction issues in gradient calc
        np.fill_diagonal(factor, 0.0)
        
        grad_X_pen = np.sum(factor * (-DX), axis=1)
        grad_Y_pen = np.sum(factor * (-DY), axis=1)
        
        # Boundary constraints
        # x - r < 0, 1 - x - r < 0, etc.
        # We add penalties for these as well.
        
        # Left
        viol_L = X - R
        mask_L = viol_L < 0
        pen_L = np.sum(np.where(mask_L, viol_L**2, 0))
        obj += weight * pen_L
        grad_X_pen += 2.0 * weight * viol_L * mask_L
        grad_R_pen += -2.0 * weight * viol_L * mask_L
        
        # Right
        viol_R = 1.0 - X - R
        mask_R = viol_R < 0
        pen_R = np.sum(np.where(mask_R, viol_R**2, 0))
        obj += weight * pen_R
        grad_X_pen += -2.0 * weight * viol_R * mask_R
        grad_R_pen += -2.0 * weight * viol_R * mask_R
        
        # Bottom
        viol_B = Y - R
        mask_B = viol_B < 0
        pen_B = np.sum(np.where(mask_B, viol_B**2, 0))
        obj += weight * pen_B
        grad_Y_pen += 2.0 * weight * viol_B * mask_B
        grad_R_pen += -2.0 * weight * viol_B * mask_B
        
        # Top
        viol_T = 1.0 - Y - R
        mask_T = viol_T < 0
        pen_T = np.sum(np.where(mask_T, viol_T**2, 0))
        obj += weight * pen_T
        grad_Y_pen += -2.0 * weight * viol_T * mask_T
        grad_R_pen += -2.0 * weight * viol_T * mask_T
        
        grad[0::3] += grad_X_pen
        grad[1::3] += grad_Y_pen
        grad[2::3] += grad_R_pen
        
        return obj, grad

    def _objective(vars_):
        return _objective_and_grad(vars_)[0]

    def _objective_grad(vars_):
        return _objective_and_grad(vars_)[1]

    def _evaluate(centers, radii):
        """Compute sum of radii if valid, else -inf."""
        # Check boundaries
        if np.any(centers[:, 0] - radii < -1e-7) or np.any(centers[:, 0] + radii > 1 + 1e-7):
            return -np.inf
        if np.any(centers[:, 1] - radii < -1e-7) or np.any(centers[:, 1] + radii > 1 + 1e-7):
            return -np.inf
            
        # Check overlaps
        D2 = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        D = np.sqrt(D2)
        R_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        # Diagonal is 0, so D=0. We ignore diagonal.
        np.fill_diagonal(D, np.inf)
        # If any distance < sum of radii, invalid
        if np.any(D < R_sum - 1e-7):
            return -np.inf
            
        return np.sum(radii)

    def _polish(pos, rad, n_iters=20):
        """
        Iteratively recompute max radii for given positions.
        This is a local refinement step.
        """
        rad = rad.copy()
        for _ in range(n_iters):
            # For each circle, the max radius is limited by:
            # 1. Distance to walls
            # 2. Distance to other circles (minus their radii)
            
            # Wall limits
            limit_walls = np.minimum(
                np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                np.minimum(pos[:, 1], 1.0 - pos[:, 1])
            )
            
            # Neighbor limits
            # D_ij - r_j
            D2 = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
            D = np.sqrt(D2)
            # Limit_i = min_j (D_ij - r_j) for j != i
            # We compute D - rad[j] for all pairs, then take min over j for each i
            # Note: D_ii = 0, so D_ii - rad_i = -rad_i. We must ignore diagonal.
            limits = D - rad[np.newaxis, :]
            np.fill_diagonal(limits, np.inf)
            limit_neighbors = np.min(limits, axis=1)
            
            new_rad = np.minimum(limit_walls, limit_neighbors)
            # Ensure positive radius
            new_rad = np.maximum(new_rad, 1e-6)
            
            # Check for convergence
            if np.allclose(rad, new_rad, atol=1e-9):
                rad = new_rad
                break
            rad = new_rad
            
        return rad

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-5, 0.5))

    # Define initializations
    inits = []
    inits.append(('hex', _make_hex_init()))
    inits.append(('grid', _make_grid_init()))
    # Add a few perturbed grids and randoms
    for seed in range(5):
        inits.append(('pgrid', _make_perturbed_grid_init(seed + 100)))
    for seed in range(10):
        inits.append(('rand', _make_random_init(seed + 42)))

    for idx, (init_type, init_pts) in enumerate(inits):
        if time.time() > deadline - 5:
            break

        init_r = _compute_initial_radius(init_pts)
        # Ensure initial radii are small enough to be valid
        # If init_r is too big, scale down
        # Actually, let the optimizer handle it, but start small to avoid huge penalties
        init_r = min(init_r, 0.05)

        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = init_pts[i, 0]
            x0[3 * i + 1] = init_pts[i, 1]
            x0[3 * i + 2] = init_r

        # For later restarts, use fewer iterations to save time
        maxiter = 400 if idx < 5 else 200
        ftol = 1e-12 if idx < 5 else 1e-10
        gtol = 1e-8

        try:
            res = minimize(_objective, x0, method='L-BFGS-B', jac=_objective_grad,
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            x_opt = res.x
        except Exception:
            x_opt = x0

        X = x_opt[0::3].copy()
        Y = x_opt[1::3].copy()
        R = x_opt[2::3].copy()
        pos = np.column_stack((X, Y))

        # Polish radii: This step is crucial for maximizing the score.
        # We fix positions and find the largest possible radii.
        R = _polish(pos, R, n_iters=30)

        score = _evaluate(pos, R)
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, R])

    # If no valid result found, return grid fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
