"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]

            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)

            radii = np.zeros(len(faces))
            valid = norms > 1e-10

            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r

            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)

            return float(np.max(radii))
        except Exception:
            lat = np.linspace(-np.pi/2, np.pi/2, 50)
            lon = np.linspace(0, 2*np.pi, 100, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            dot = np.dot(grid_pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def spectral_norm_objective(pts):
        """
        Compute spectral norm of the integral operator associated with a 
        rotationally symmetric periodic potential on the sphere.
        
        The potential is built from spherical harmonics up to degree L.
        The integral operator K has kernel K(p, q) = sum_{l=0}^{L} c_l P_l(p·q)
        where P_l are Legendre polynomials. The spectral norm is the largest 
        eigenvalue of the discrete approximation.
        
        Minimizing this spectral norm encourages uniform coverage because 
        non-uniform distributions create large eigenvalues in the potential field.
        """
        L = 8  # Spherical harmonic degree
        # Build the Gram matrix of the potential
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        
        # Compute Legendre polynomials P_l(x) for l = 0..L
        # Using recurrence: P_0=1, P_1=x, (l+1)P_{l+1} = (2l+1)xP_l - lP_{l-1}
        P = np.zeros((L + 1, N, N))
        P[0, :, :] = 1.0
        if L >= 1:
            P[1, :, :] = dot
        for l in range(1, L):
            P[l + 1, :, :] = ((2 * l + 1) * dot * P[l, :, :] - l * P[l - 1, :, :]) / (l + 1)
        
        # Weights c_l that emphasize higher-order uniformity
        # c_l = 1/(l+1) gives more weight to low-order modes (overall shape)
        # but we want to penalize non-uniformity, so use c_l = 1 for all l
        # Actually, to minimize spectral norm, we want the operator to be as "flat" as possible
        c = np.ones(L + 1) / (L + 1)
        
        # Build the integral operator matrix: K = sum_l c_l P_l
        K = np.zeros((N, N))
        for l in range(L + 1):
            K += c[l] * P[l, :, :]
        
        # The spectral norm is the largest singular value of K
        # Since K is symmetric, this is the largest absolute eigenvalue
        try:
            eigvals = np.linalg.eigvalsh(K)
            return float(np.max(np.abs(eigvals)))
        except Exception:
            return float('inf')

    def spectral_gradient(pts):
        """
        Compute gradient of the spectral norm objective w.r.t. point positions.
        Uses the fact that d/dtheta ||K||_spec = v^T (dK/dtheta) v where v is 
        the top eigenvector.
        """
        L = 8
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        
        P = np.zeros((L + 1, N, N))
        P[0, :, :] = 1.0
        if L >= 1:
            P[1, :, :] = dot
        for l in range(1, L):
            P[l + 1, :, :] = ((2 * l + 1) * dot * P[l, :, :] - l * P[l - 1, :, :]) / (l + 1)
        
        c = np.ones(L + 1) / (L + 1)
        K = np.zeros((N, N))
        for l in range(L + 1):
            K += c[l] * P[l, :, :]
        
        try:
            eigvals, eigvecs = np.linalg.eigh(K)
            # Take the eigenvector corresponding to the largest absolute eigenvalue
            idx = np.argmax(np.abs(eigvals))
            v = eigvecs[:, idx]
        except Exception:
            return None
        
        # Gradient of ||K||_spec w.r.t. each point
        # dK/dx_i involves derivatives of P_l(p_i · p_j) w.r.t. p_i
        grad = np.zeros_like(pts)
        for i in range(N):
            dK_di = np.zeros((N, N))
            for j in range(N):
                if i == j:
                    continue
                # d/dx_i P_l(p_i · p_j) = P_l'(p_i · p_j) * p_j
                # P_l'(x) can be computed from recurrence or directly
                # For simplicity, use finite difference approximation
                pass
            
            # Use a simpler approach: the gradient of the spectral norm 
            # w.r.t. point i is proportional to the off-diagonal contributions
            # Simplified: use the repulsion-like force from the top eigenvector
            # v_j * (some function of p_i · p_j)
            for j in range(N):
                if i == j:
                    continue
                cos_ij = np.dot(pts[i], pts[j])
                cos_ij = np.clip(cos_ij, -1.0, 1.0)
                # Derivative of P_l w.r.t. cos
                # P_l'(x) = (l/(x^2-1))(x P_l(x) - P_{l-1}(x)) for |x| < 1
                # This is complex, so use a simpler surrogate
                # The force should push points apart in a way that reduces spectral norm
                weight = v[i] * v[j]
                # Repulsive force weighted by eigenvector components
                grad[i] += weight * (pts[i] - np.dot(pts[i], pts[j]) * pts[j])
        
        # Normalize gradient
        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        grad = grad / norms
        return grad

    def spectral_optimization(points, n_iter=200, lr=0.01):
        """
        Optimize points to minimize the spectral norm of the integral operator.
        Uses gradient descent with the spectral norm objective.
        """
        best = points.copy()
        best_val = spectral_norm_objective(best)
        
        for it in range(n_iter):
            if time.time() > deadline - 10.0:
                break
            
            val = spectral_norm_objective(points)
            if val < best_val:
                best_val = val
                best = points.copy()
            
            # Compute gradient
            grad = spectral_gradient(points)
            if grad is None:
                break
            
            # Gradient descent step
            points = points - lr * grad
            # Project back to sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
            # Decay learning rate
            lr *= 0.999
        
        return best

    def hybrid_optimize(points, n_iter=300):
        """
        Hybrid optimization: alternate between spectral norm minimization 
        and standard repulsion to escape local minima.
        """
        best = points.copy()
        best_rad = compute_covering_radius(best)
        best_spec = spectral_norm_objective(best)
        
        lr = 0.01
        for it in range(n_iter):
            if time.time() > deadline - 10.0:
                break
            
            # Alternate: every 10 steps, do a spectral step; otherwise repulsion
            if it % 10 == 0:
                # Spectral gradient step
                grad = spectral_gradient(points)
                if grad is not None:
                    points = points - lr * grad
                    norms = np.linalg.norm(points, axis=1, keepdims=True)
                    points = points / norms
            else:
                # Standard repulsion step
                dot = np.dot(points, points.T)
                dot = np.clip(dot, -1.0, 1.0)
                np.fill_diagonal(dot, 1.0)
                dist_sq = 2.0 - 2.0 * dot
                dist_sq[dist_sq < 1e-10] = 1e-10
                w = 1.0 / (dist_sq ** 1.5)
                np.fill_diagonal(w, 0.0)
                sum_w = np.sum(w, axis=1)
                sum_p_w = w @ points
                forces = points * sum_w[:, np.newaxis] - sum_p_w
                points = points + lr * forces
                norms = np.linalg.norm(points, axis=1, keepdims=True)
                points = points / norms
            
            # Check improvement
            if it % 5 == 0:
                rad = compute_covering_radius(points)
                spec = spectral_norm_objective(points)
                if rad < best_rad:
                    best_rad = rad
                    best = points.copy()
                    best_spec = spec
            
            lr *= 0.998
            if lr < 1e-6:
                break
        
        return best

    def polish(points, alpha, n_iter=50):
        """High-precision polish."""
        for _ in range(n_iter):
            dot = np.dot(points, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            np.fill_diagonal(dot, 1.0)
            dist_sq = 2.0 - 2.0 * dot
            dist_sq[dist_sq < 1e-10] = 1e-10
            w = 1.0 / (dist_sq ** 1.5)
            np.fill_diagonal(w, 0.0)
            sum_w = np.sum(w, axis=1)
            sum_p_w = w @ points
            forces = points * sum_w[:, np.newaxis] - sum_p_w
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        return points

    best_points = None
    best_rad = np.inf

    n_restarts = 6

    for restart in range(n_restarts):
        if time.time() > deadline - 25.0:
            break

        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            # Fibonacci with perturbation
            points = fibonacci_spiral(N) + 0.08 * np.random.randn(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        else:
            if best_points is not None:
                points = best_points + 0.1 * np.random.randn(N, 3)
            else:
                points = np.random.randn(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms

        # Hybrid optimization
        if time.time() < deadline - 20.0:
            points = hybrid_optimize(points, n_iter=250)

        # Spectral-only polish
        if time.time() < deadline - 15.0:
            points = spectral_optimization(points, n_iter=100, lr=0.005)

        # Standard polish
        if time.time() < deadline - 10.0:
            points = polish(points, 0.005, 40)

        rad = compute_covering_radius(points)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Final polish
    if time.time() < deadline - 5.0 and best_points is not None:
        final = polish(best_points.copy(), 0.002, 50)
        rad_final = compute_covering_radius(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final

        # Final spectral optimization
        if time.time() < deadline - 3.0:
            final2 = spectral_optimization(best_points.copy(), n_iter=80, lr=0.003)
            rad_final2 = compute_covering_radius(final2)
            if rad_final2 < best_rad:
                best_rad = rad_final2
                best_points = final2

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
