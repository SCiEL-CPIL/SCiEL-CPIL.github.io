import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin for 120s limit

def construct():
    # 1. Initial Configuration: Hexagonal Lattice
    # We want to place N points. A hexagonal lattice is a good start.
    # We'll generate a grid and pick the N points closest to the origin 
    # that satisfy the constraint that they are inside the unit circle 
    # (with a buffer).
    
    def get_initial_points():
        # Try different spacings to find a lattice that yields at least N points
        # inside the unit circle.
        best_points = None
        best_radius = 0.0
        
        # Try a range of spacings
        for spacing in np.linspace(0.15, 0.25, 20):
            k = int(1.0 / spacing) + 2
            # Generate hexagonal lattice points
            # x = i * spacing + (j % 2) * spacing / 2
            # y = j * spacing * sqrt(3) / 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            
            # Keep points strictly inside (with buffer) to avoid initial boundary violations
            # Buffer is half the spacing to ensure initial non-overlap in the lattice
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            radius = radius[mask]
            
            if len(points) >= N:
                # Pick the N points closest to the origin to form a compact cluster
                points = points[np.argsort(radius, kind='stable')[:N]]
                r = _compute_radius(points)
                if r > best_radius:
                    best_radius = r
                    best_points = points.copy()
        
        if best_points is not None:
            return best_points
        
        # Fallback: simple grid or random if lattice fails (unlikely)
        spacing = 0.2
        k = int(1.0 / spacing) + 2
        rows = [(i * spacing, j * spacing) for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        radius = radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind='stable')[:N]]
        
        # Last resort: random
        return np.random.rand(N, 2) * 1.5 - 0.75

    x0 = get_initial_points()
    
    # Ensure x0 is valid (finite, correct shape)
    if not np.all(np.isfinite(x0)):
        return np.zeros((N, 2))
        
    best_x = x0.copy()
    best_score = _compute_radius(best_x)

    # 2. Optimization: Maximize the minimum distance
    # We use a smooth surrogate: Softmin of the distances.
    # Objective: Minimize -Softmin(distances)
    # Softmin(d) = -1/beta * log(sum(exp(-beta * d)))
    # This is smooth and approximates min(d) as beta -> inf.
    
    def _softmin_obj_and_grad(flat_centers, beta):
        """
        Objective: minimize -softmin of (pairwise dists/2, boundary dists)
        Gradient computed analytically.
        
        flat_centers: shape (2*N,)
        beta: float, sharpness of the softmin
        """
        centers = flat_centers.reshape(N, 2)
        
        # Boundary distances
        r_norm = np.hypot(centers[:, 0], centers[:, 1])
        boundary_dists = 1.0 - r_norm
        
        # Pairwise distances / 2
        x = centers[:, 0]
        y = centers[:, 1]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.hypot(dx, dy)
        np.fill_diagonal(dists, 0.0)
        
        # We only need upper triangle for unique pairs
        idx_i, idx_j = np.triu_indices(N, k=1)
        pair_dists = dists[idx_i, idx_j] / 2.0
        
        # Collect all constraints
        # all_dists[0:N] are boundary dists
        # all_dists[N:] are pair dists
        all_dists = np.concatenate([boundary_dists, pair_dists])
        
        # Soft-min: softmin(d) = -1/beta * log(sum(exp(-beta * d)))
        # We want to maximize softmin, so minimize f = -softmin(d) = 1/beta * log(sum(exp(-beta * d)))
        
        # Numerical stability: subtract max argument from exponent
        # exp(-beta * d)
        # Let M = max(-beta * d) = -beta * min(d)
        # sum(exp(-beta * d)) = exp(M) * sum(exp(-beta * d - M))
        # log(sum(exp(-beta * d))) = M + log(sum(exp(-beta * d - M)))
        
        shifted = -beta * all_dists
        max_shifted = np.max(shifted)
        exp_shifted = np.exp(shifted - max_shifted)
        sum_exp = np.sum(exp_shifted)
        
        if sum_exp == 0:
            return 0.0, np.zeros(2*N)
            
        log_sum_exp = np.log(sum_exp)
        
        # f = 1/beta * (max_shifted + log_sum_exp)
        obj = (max_shifted + log_sum_exp) / beta
        
        # Gradient:
        # f = 1/beta * (max_shifted + log(sum_exp))
        # Note: max_shifted depends on all_dists, but in the context of the log-sum-exp trick,
        # the gradient of log(sum(exp(-beta*d))) w.r.t d_i is:
        # d/d d_i [ log(sum_j exp(-beta*d_j)) ] = (1/sum_j exp(-beta*d_j)) * (-beta * exp(-beta*d_i))
        # = -beta * (exp(-beta*d_i) / sum_j exp(-beta*d_j))
        # = -beta * w_i  where w_i is the softmax weight.
        
        # So d f / d d_i = 1/beta * (-beta * w_i) = -w_i
        
        weights = exp_shifted / sum_exp
        
        # Now map gradient from all_dists to centers
        grad_centers = np.zeros((N, 2))
        
        # 1. Boundary part: all_dists[0:N]
        # boundary_dists[i] = 1 - hypot(centers[i])
        # d(boundary_dists[i])/d(centers[i]) = -centers[i] / hypot(centers[i])
        for i in range(N):
            if r_norm[i] > 1e-12:
                # grad_all_dists[i] = -weights[i]
                # grad_centers[i] += grad_all_dists[i] * d(boundary_dists[i])/d(centers[i])
                grad_centers[i] += (-weights[i]) * (-centers[i] / r_norm[i])
        
        # 2. Pairwise part: all_dists[N:]
        # pair_dists[k] corresponds to pair (idx_i[k], idx_j[k])
        # pair_dists[k] = hypot(centers[i]-centers[j]) / 2
        # Let d_ij = hypot(centers[i]-centers[j])
        # d(pair_dists[k])/d(centers[i]) = (centers[i]-centers[j]) / (2 * d_ij)
        # d(pair_dists[k])/d(centers[j]) = (centers[j]-centers[i]) / (2 * d_ij) = - (centers[i]-centers[j]) / (2 * d_ij)
        
        for k in range(len(idx_i)):
            i = idx_i[k]
            j = idx_j[k]
            d_ij = dists[i, j]
            if d_ij < 1e-12:
                continue
            
            # grad_all_dists[N+k] = -weights[N+k]
            w = -weights[N + k]
            
            diff = centers[i] - centers[j]
            grad_vec = diff / (2.0 * d_ij)
            
            grad_centers[i] += w * grad_vec
            grad_centers[j] += w * (-grad_vec)
        
        grad_flat = grad_centers.flatten()
        
        return obj, grad_flat

    # We will run optimization with increasing beta
    # Start with beta=10, go up to 100
    
    for beta in [10.0, 50.0, 100.0, 200.0]:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Use current best as start
        x_start = best_x.reshape(-1)
        
        # Bounds: Each coordinate in [-1, 1] (actually constrained by circle, but box is safe)
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            res = minimize(
                _softmin_obj_and_grad, 
                x_start, 
                args=(beta,), 
                method='L-BFGS-B', 
                jac=True, 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
            )
            
            if res.success or res.fun < _softmin_obj_and_grad(x_start, beta)[0]:
                x_opt = res.x.reshape(N, 2)
                score_opt = _compute_radius(x_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_x = x_opt
        except Exception:
            pass

    # Final check and return
    # Ensure the returned points are valid
    final_score = _compute_radius(best_x)
    if final_score < 0 or not np.all(np.isfinite(best_x)):
        # Fallback to initial if something went wrong
        return x0
        
    return best_x

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    
    # Pairwise distances
    x = centers[:, 0]
    y = centers[:, 1]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists = np.hypot(dx, dy)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

if __name__ == "__main__":
    # Simple test
    pts = construct()
    print(pts.shape)
    # Calculate score
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r_bound = np.min(1.0 - np.linalg.norm(pts, axis=1))
    r = min(min_pair / 2.0, r_bound)
    print(f"Score: {r}")
