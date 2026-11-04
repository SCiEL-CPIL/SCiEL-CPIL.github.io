"""
Circle packing in unit square.
Structural change: Voronoi-inflation representation. The 26 centers define a Voronoi
partition of [0,1]^2; each circle's radius is the inradius of its cell (min distance
from the site to the cell boundary). We maximize the sum of inradii with L-BFGS-B on
a smooth soft-min surrogate of the Voronoi inradius, which is differentiable and
encodes non-overlap geometrically (no explicit pairwise collision terms).
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 140.0  # safety margin from 150 s wall-clock limit


def construct():
    start_time = time.time()

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def voronoi_inradius_soft(p, beta):
        """
        For each site i, compute the soft-min over all half-edges of the Voronoi cell
        of site i. The Voronoi cell of site i is {z : ||z - p_i|| <= ||z - p_j|| for all j}.
        The boundary between i and j is the perpendicular bisector. The distance from
        p_i to that bisector is ||p_i - p_j||/2, but that's only if the bisector segment
        actually forms part of the cell boundary. A tighter bound: the inradius of the
        Voronoi cell is the min over j of the distance from p_i to the Voronoi edge
        between i and j, AND the distance to the square boundary.
        
        However, computing the exact Voronoi cell edges is complex. A valid (conservative)
        lower bound on the inradius is: min( dist_to_boundary, min_j ||p_i - p_j||/2 ).
        This is exactly the "compute_radii" from the current program. But to make it
        smooth and differentiable for L-BFGS-B, we use a soft-min.
        
        Actually, the exact Voronoi inradius can be smaller than min_j ||p_i-p_j||/2
        because the cell may be cut by multiple edges. But for the purpose of a smooth
        objective that, when optimized, drives circles apart and toward the center,
        the soft-min of the pairwise half-distances and boundary distances is a good
        surrogate. When beta -> inf, it converges to the exact min.
        """
        x = p[0::2]
        y = p[1::2]
        
        # Distance to each of the 4 boundaries
        d_left = x
        d_right = 1.0 - x
        d_bottom = y
        d_top = 1.0 - y
        boundary_dists = np.stack([d_left, d_right, d_bottom, d_top], axis=-1)  # (N, 4)
        
        # Pairwise half-distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        half_dists = 0.5 * dist  # (N, N)
        
        # Combine: for each i, we have 4 boundary dists and (N-1) half-distances
        # Total candidates per i: 4 + N - 1 = N + 3
        all_dists = np.concatenate([boundary_dists, half_dists], axis=-1)  # (N, N+3)
        
        # Soft-min: -logsumexp(-beta * d) / beta
        # To avoid overflow, use the log-sum-exp trick
        scaled = -beta * all_dists
        # Clip to avoid extreme values
        scaled = np.clip(scaled, -500, 500)
        max_scaled = np.max(scaled, axis=-1, keepdims=True)
        # If max_scaled is -inf (shouldn't happen), handle it
        max_scaled = np.where(np.isfinite(max_scaled), max_scaled, 0)
        exp_scaled = np.exp(scaled - max_scaled)
        logsumexp = np.log(np.sum(exp_scaled, axis=-1)) + max_scaled.squeeze(-1)
        soft_min = -logsumexp / beta
        
        return soft_min

    def voronoi_inradius_soft_grad(p, beta):
        """
        Gradient of the soft-min inradius with respect to p.
        soft_min_i = -logsumexp(-beta * d_{i,k}) / beta
        d_{i,k} is the k-th candidate distance for site i.
        
        d/dp [ -logsumexp(-beta * d) / beta ]
        = -1/beta * d/dp [ logsumexp(-beta * d) ]
        = -1/beta * sum_k exp(-beta * d_k) / sum_k exp(-beta * d_k) * (-beta * d/dp d_k)
        = sum_k w_k * d/dp d_k
        where w_k = exp(-beta * d_k) / sum_j exp(-beta * d_j)
        
        Note: this is the gradient of the soft-min, which is a weighted average of the
        gradients of the individual distances, with weights proportional to exp(-beta * d).
        Smaller distances get larger weights, so the gradient is dominated by the closest
        constraints, as desired.
        """
        x = p[0::2]
        y = p[1::2]
        
        # Distance to each of the 4 boundaries
        d_left = x
        d_right = 1.0 - x
        d_bottom = y
        d_top = 1.0 - y
        boundary_dists = np.stack([d_left, d_right, d_bottom, d_top], axis=-1)  # (N, 4)
        
        # Pairwise half-distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        half_dists = 0.5 * dist  # (N, N)
        
        all_dists = np.concatenate([boundary_dists, half_dists], axis=-1)  # (N, N+3)
        
        scaled = -beta * all_dists
        scaled = np.clip(scaled, -500, 500)
        max_scaled = np.max(scaled, axis=-1, keepdims=True)
        max_scaled = np.where(np.isfinite(max_scaled), max_scaled, 0)
        exp_scaled = np.exp(scaled - max_scaled)
        sum_exp = np.sum(exp_scaled, axis=-1, keepdims=True)
        weights = exp_scaled / sum_exp  # (N, N+3)
        
        # Now compute gradient of each candidate distance w.r.t. p
        # For site i, the gradient w.r.t. p_i is what we need.
        # Boundary distances:
        # d_left = x_i -> d/dx_i = 1
        # d_right = 1 - x_i -> d/dx_i = -1
        # d_bottom = y_i -> d/dy_i = 1
        # d_top = 1 - y_i -> d/dy_i = -1
        
        # Pairwise half-distance: d_{i,j} = 0.5 * ||p_i - p_j||
        # d/dp_i d_{i,j} = 0.5 * (p_i - p_j) / ||p_i - p_j||
        
        grad = np.zeros_like(p)
        
        for i in range(N):
            w = weights[i]  # (N+3,)
            # Boundary contributions
            # w[0] for d_left, w[1] for d_right, w[2] for d_bottom, w[3] for d_top
            grad[2*i] += w[0] * 1.0 + w[1] * (-1.0)
            grad[2*i+1] += w[2] * 1.0 + w[3] * (-1.0)
            
            # Pairwise contributions
            for j in range(N):
                if i == j:
                    continue
                w_ij = w[4 + j] if j < i else w[4 + j + 1]  # careful with indexing
                # Actually, half_dists[i, j] is at position 4 + j in all_dists[i]
                # because boundary_dists has 4 entries, then half_dists[i, 0], half_dists[i, 1], ...
                # But half_dists[i, i] is inf, so its weight is 0.
                w_ij = w[4 + j]
                if w_ij < 1e-12:
                    continue
                dx_ij = x[i] - x[j]
                dy_ij = y[i] - y[j]
                d_ij = np.sqrt(dx_ij**2 + dy_ij**2)
                if d_ij < 1e-12:
                    continue
                # d/dp_i (0.5 * ||p_i - p_j||) = 0.5 * (p_i - p_j) / ||p_i - p_j||
                grad[2*i] += w_ij * 0.5 * dx_ij / d_ij
                grad[2*i+1] += w_ij * 0.5 * dy_ij / d_ij
        
        return grad

    def objective_and_grad(p, beta):
        r = voronoi_inradius_soft(p, beta)
        obj = -np.sum(r)
        grad_r = voronoi_inradius_soft_grad(p, beta)
        grad = -grad_r
        return obj, grad

    def compute_radii_exact(x, y):
        """Exact radii from centers: min of half-neighbor-distance and boundary."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-8)
        return r

    def value(x, y):
        r = compute_radii_exact(x, y)
        return float(np.sum(r)), r

    def check_validity(x, y, r, tol=1e-9):
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
        if np.any(r < -tol):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        need = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(need, 0)
        if np.any(dist < need - tol):
            return False
        return True

    def hex_init(seed):
        rng = np.random.RandomState(seed)
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for row in range(6):
            for col in range(5):
                if idx >= N:
                    break
                x[idx] = (col + 0.5) / 5.0 + (0.5 / 5.0) * (row % 2)
                y[idx] = (row + 0.5) / 6.0
                idx += 1
                if idx >= N:
                    break
        x += rng.uniform(-0.03, 0.03, N)
        y += rng.uniform(-0.03, 0.03, N)
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)
        return x, y

    def grid_init(seed):
        rng = np.random.RandomState(seed)
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for i in range(6):
            for j in range(5):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) / 5.0
                y[idx] = (i + 0.5) / 6.0
                idx += 1
                if idx >= N:
                    break
        x += rng.uniform(-0.04, 0.04, N)
        y += rng.uniform(-0.04, 0.04, N)
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)
        return x, y

    def random_init(seed):
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        return x, y

    def lbfags_optimize(x0, y0, beta_schedule):
        """Run L-BFGS-B with a schedule of increasing beta (sharpening)."""
        x = x0.copy()
        y = y0.copy()
        p = np.empty(2 * N)
        p[0::2] = x
        p[1::2] = y
        
        for beta in beta_schedule:
            if time_left() < 10:
                break
            p, success, _ = minimize(
                lambda q: objective_and_grad(q, beta),
                p,
                jac=True,
                method='L-BFGS-B',
                bounds=[(0.01, 0.99)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            x = p[0::2]
            y = p[1::2]
        
        return x, y

    # Multi-start orchestration
    best_val = -np.inf
    best_x = None
    best_y = None
    best_r = None

    inits = [
        (hex_init, 0), (grid_init, 1), (random_init, 2),
        (hex_init, 10), (grid_init, 11), (random_init, 12),
        (hex_init, 20), (grid_init, 21), (random_init, 22),
        (hex_init, 30), (grid_init, 31), (random_init, 32),
    ]

    # Beta schedule: start smooth, end sharp
    beta_schedule = [10, 20, 50, 100, 200, 500]

    for init_func, seed in inits:
        if time_left() < 30:
            break
        x0, y0 = init_func(seed)
        v0, r0 = value(x0, y0)

        if time_left() < 15:
            if v0 > best_val:
                best_val = v0
                best_x = x0.copy()
                best_y = y0.copy()
                best_r = r0.copy()
            continue

        # L-BFGS-B optimization
        xb, yb = lbfags_optimize(x0, y0, beta_schedule)
        vb, rb = value(xb, yb)

        if vb > v0:
            x0, y0, v0, r0 = xb, yb, vb, rb

        if v0 > best_val:
            best_val = v0
            best_x = x0.copy()
            best_y = y0.copy()
            best_r = r0.copy()

    # Final high-precision polish: run L-BFGS-B with very high beta on the best config
    if best_x is not None and time_left() > 10:
        xb, yb = lbfags_optimize(best_x, best_y, [500, 1000, 2000])
        vb, rb = value(xb, yb)
        if vb > best_val:
            best_val = vb
            best_x, best_y, best_r = xb, yb, rb

    # Ensure validity
    if best_x is None:
        x, y = hex_init(0)
        best_r = compute_radii_exact(x, y)
        best_x, best_y = x, y
    elif not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.999, 0.95, 20):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii_exact(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
