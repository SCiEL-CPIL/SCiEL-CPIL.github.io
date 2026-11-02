"""Program for covering the sphere with 130 equal caps using
Sinkhorn-regularized optimal transport with exact hull-based polish."""
import numpy as np
import time
from scipy.spatial import ConvexHull
import warnings

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def generate_grid(n_lat, n_lon):
        lat = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, n_lat)
        lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    grid_coarse = generate_grid(40, 80)
    grid_fine = generate_grid(80, 160)

    def _norm(pts):
        n = np.linalg.norm(pts, axis=1, keepdims=True)
        n = np.where(n < 1e-12, 1.0, n)
        return pts / n

    def covering_radius(pts, grid=None):
        if grid is None:
            grid = grid_fine
        dots = np.dot(grid, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots)
        return float(np.max(np.min(ang, axis=1)))

    def _soft_min_logsumexp(angles, sigma):
        """Soft min of angles (M, N) -> (M, N, scalar)."""
        neg = -angles / sigma
        mx = np.max(neg, axis=1, keepdims=True)
        exp_neg = np.exp(neg - mx)
        logsumexp = mx + np.log(np.sum(exp_neg, axis=1, keepdims=True))
        weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)
        soft_min = sigma * (mx + np.log(np.sum(exp_neg, axis=1, keepdims=True)))
        return soft_min, weights

    def _sinkhorn_distance_grad(points, grid, eps, n_iter=20):
        """
        Sinkhorn divergence between source (points, uniform weights) and
        target (grid, uniform weights).
        Returns value and gradient w.r.t. points.
        """
        M = grid.shape[0]
        # Cost matrix: angular distance
        dots = np.dot(grid, points.T)  # (M, N)
        dots = np.clip(dots, -1.0, 1.0)
        cost = np.arccos(dots)  # (M, N)

        # Sinkhorn iteration
        u = np.zeros(N)
        v = np.zeros(M)
        a = np.full(N, 1.0 / N)
        b = np.full(M, 1.0 / M)

        K = np.exp(-cost / eps)  # (M, N)

        for _ in range(n_iter):
            # v = b / (K^T u)
            Ku = K.T @ u  # (M,)
            v = b / np.maximum(Ku, 1e-30)
            # u = a / (K v)
            Kvt = K @ v  # (N,)
            u = a / np.maximum(Kvt, 1e-30)

        # Transport plan
        P = np.exp(-cost / eps + u[:, None] + v[None, :])  # (M, N)
        P = P / np.sum(P)

        # Sinkhorn divergence value
        val = np.sum(cost * P)

        # Gradient w.r.t. points
        # d/dp_i cost(m,i) = -sin(angle) / cos(angle) * (p_i - proj)
        # Actually: d/dp_i arccos(p_i · g_m) = -(g_m - (p_i·g_m) p_i) / sqrt(1 - (p_i·g_m)^2)
        denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))  # (M, N)
        # grad_cost[i, m] = -(g_m - dots[m,i] * p_i) / denom[m,i]
        # In matrix form: for each i, sum over m of P[m,i] * (-(g_m - dots[m,i]*p_i) / denom[m,i])
        # = -sum_m P[m,i] * g_m / denom[m,i] + p_i * sum_m P[m,i] * dots[m,i] / denom[m,i]
        P_over_denom = P / denom  # (M, N)
        term1 = -(P_over_denom.T @ grid)  # (N, 3)
        coeff = np.sum(P_over_denom * dots, axis=0)  # (N,)
        term2 = coeff[:, None] * points  # (N, 3)
        grad = term1 + term2

        return val, grad

    def _sinkhorn_optimize(pts, grid, eps_list, max_iter_per_eps=30, lr=0.01):
        """Optimize using Sinkhorn divergence gradient descent with adaptive lr."""
        current = pts.copy()
        for eps in eps_list:
            if time.time() > deadline - 10.0:
                break
            lr_cur = lr
            for it in range(max_iter_per_eps):
                if time.time() > deadline - 10.0:
                    break
                val, grad = _sinkhorn_distance_grad(current, grid, eps, n_iter=15)
                # Gradient descent with normalization
                current = current - lr_cur * grad
                current = _norm(current)
                # Decay lr
                lr_cur *= 0.98
        return current

    def compute_hull_facets(p):
        try:
            h = ConvexHull(p)
        except Exception:
            return None
        simplices = h.simplices
        verts = p[simplices]
        if verts.shape[0] == 0:
            return None
        a = verts[:, 0, :]
        b = verts[:, 1, :]
        c = verts[:, 2, :]
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            return None
        normals = normals[valid]
        normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        centroids = np.mean(verts[valid], axis=1)
        dots_cent = np.einsum('ij,ij->i', normals, centroids)
        normals[dots_cent < 0] *= -1
        v_valid = verts[valid]
        dots_a = np.einsum('ij,ij->i', normals, v_valid[:, 0, :])
        dots_b = np.einsum('ij,ij->i', normals, v_valid[:, 1, :])
        dots_c = np.einsum('ij,ij->i', normals, v_valid[:, 2, :])
        angles_a = np.arccos(np.clip(dots_a, -1, 1))
        angles_b = np.arccos(np.clip(dots_b, -1, 1))
        angles_c = np.arccos(np.clip(dots_c, -1, 1))
        facet_radii = np.max(np.stack([angles_a, angles_b, angles_c]), axis=0)
        return simplices[valid], normals, facet_radii

    def _lloyd_step(pts, grid, n_iters=20, alpha=0.5):
        current = pts.copy()
        for it in range(n_iters):
            if time.time() > deadline - 15.0:
                break
            dots = np.dot(grid, current.T)
            nearest = np.argmax(dots, axis=1)
            new_pts = np.zeros_like(current)
            for i in range(N):
                mask = (nearest == i)
                if not np.any(mask):
                    new_pts[i] = current[i]
                    continue
                cell_pts = grid[mask]
                d = current[i] @ cell_pts.T
                d = np.clip(d, -1.0, 1.0)
                ang = np.arccos(d)
                temp = 3.0
                w = np.exp(ang * temp)
                w /= np.sum(w)
                new_c = np.sum(cell_pts * w[:, None], axis=0)
                nrm = np.linalg.norm(new_c)
                if nrm < 1e-10:
                    new_pts[i] = current[i]
                else:
                    new_pts[i] = new_c / nrm
            current = (1 - alpha) * current + alpha * new_pts
            current = _norm(current)
        return current

    def _exact_local_polish(pts, max_iters=100, initial_step=0.03):
        pts = pts.copy()
        fd = compute_hull_facets(pts)
        if fd is None:
            return pts
        worst_radius = np.max(fd[2])
        step_size = initial_step
        for it in range(max_iters):
            if time.time() > deadline - 8.0:
                break
            worst_idx = np.argmax(fd[2])
            worst_simplex = fd[0][worst_idx]
            worst_normal = fd[1][worst_idx]
            worst_radius = fd[2][worst_idx]
            improved = False
            for j in worst_simplex:
                p_j = pts[j]
                move_dir = worst_normal - np.dot(p_j, worst_normal) * p_j
                move_norm = np.linalg.norm(move_dir)
                if move_norm < 1e-10:
                    continue
                move_dir /= move_norm
                best_ss = 0
                best_new_radius = worst_radius
                for ss in [step_size, step_size * 0.5, step_size * 0.25, step_size * 0.1]:
                    new_p = p_j + ss * move_dir
                    new_p = new_p / np.linalg.norm(new_p)
                    new_pts = pts.copy()
                    new_pts[j] = new_p
                    new_fd = compute_hull_facets(new_pts)
                    if new_fd is None:
                        continue
                    new_worst = np.max(new_fd[2])
                    if new_worst < best_new_radius - 1e-12:
                        best_new_radius = new_worst
                        best_ss = ss
                        best_new_pts = new_pts
                        best_fd = new_fd
                if best_ss > 0:
                    pts = best_new_pts
                    fd = best_fd
                    worst_radius = best_new_radius
                    improved = True
                    step_size = best_ss * 1.3
                    break
            if not improved:
                step_size *= 0.5
                if step_size < 1e-7:
                    break
        return pts

    best_pts = None
    best_rad = np.inf

    n_restarts = 6
    for r in range(n_restarts):
        if time.time() > deadline - 50.0:
            break

        if r == 0:
            pts = fibonacci_spiral(N)
        elif r == 1:
            rng = np.random.RandomState(42)
            pts = rng.randn(N, 3)
            pts = _norm(pts)
        elif r == 2:
            rng = np.random.RandomState(789)
            pts = rng.randn(N, 3)
            pts = _norm(pts)
        elif r == 3:
            rng = np.random.RandomState(1234)
            pts = rng.randn(N, 3)
            pts = _norm(pts)
        elif r >= 4 and best_pts is not None:
            rng = np.random.RandomState(5000 + r)
            scale = max(0.01, 0.04 - 0.005 * (r - 4))
            pts = best_pts + scale * rng.randn(N, 3)
            pts = _norm(pts)
        else:
            pts = fibonacci_spiral(N)

        # Phase 1: Lloyd
        cand = _lloyd_step(pts, grid_coarse, n_iters=25, alpha=0.5)

        # Phase 2: Sinkhorn OT optimization
        if time.time() < deadline - 30.0:
            eps_list = [0.15, 0.08, 0.04, 0.02]
            cand = _sinkhorn_optimize(cand, grid_coarse, eps_list, max_iter_per_eps=20, lr=0.02)

        # Phase 3: Exact hull polish
        if time.time() < deadline - 20.0:
            cand = _exact_local_polish(cand, max_iters=80, initial_step=0.03)

        # Phase 4: Fine-tune
        if time.time() < deadline - 15.0:
            cand = _sinkhorn_optimize(cand, grid_fine, [0.01, 0.005], max_iter_per_eps=15, lr=0.01)
            cand = _exact_local_polish(cand, max_iters=150, initial_step=0.01)

        rad = covering_radius(cand)
        if rad < best_rad:
            best_rad = rad
            best_pts = cand.copy()
            print(f"Restart {r}: {np.degrees(best_rad):.6f} deg")

    # Final refinement
    if best_pts is not None and time.time() < deadline - 10.0:
        fp = _lloyd_step(best_pts, grid_coarse, n_iters=10, alpha=0.3)
        fp = _sinkhorn_optimize(fp, grid_fine, [0.005, 0.003], max_iter_per_eps=15, lr=0.008)
        fp = _exact_local_polish(fp, max_iters=200, initial_step=0.005)
        fr = covering_radius(fp)
        if fr < best_rad:
            best_rad = fr
            best_pts = fp

    if best_pts is None:
        best_pts = fibonacci_spiral(N)

    return best_pts
