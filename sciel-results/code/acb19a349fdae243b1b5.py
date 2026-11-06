import numpy as np
from scipy.optimize import minimize
import time

N = 130

def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return points / norms

def _covering_radius(points, test_points):
    # points: (N, 3)
    # test_points: (M, 3)
    diff = test_points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)  # (M, N)
    min_dists = np.min(dists, axis=1)  # (M,)
    return np.max(min_dists)

def _spectral_energy(points):
    # Compute the variance of the pair correlation function.
    # This promotes uniformity by penalizing clustering.
    n = len(points)
    # Pairwise dot products
    dot = points @ points.T
    dot = np.clip(dot, -1.0, 1.0)
    
    # Pairwise angular distances
    # For efficiency, we can work with dot products directly if possible,
    # but let's compute distances for clarity.
    dists = np.arccos(dot)
    np.fill_diagonal(dists, 0.0)
    
    # We want to minimize the variance of the number of points in shells?
    # Or simply minimize the sum of squared distances to a target?
    # A common spectral energy is related to the power spectrum of the point set.
    # Let's use a simpler proxy: minimize the variance of the pairwise distances.
    # If points are uniform, the distribution of pairwise distances should be smooth.
    # Actually, minimizing the sum of squared pairwise distances (Thomson problem) is related.
    # But for covering, we want to avoid large gaps.
    
    # Let's try minimizing the "clumping" by maximizing the minimum distance
    # or minimizing the variance of local density.
    
    # A robust spectral approach:
    # Compute the power spectrum of the point distribution.
    # We can approximate this by binning the pairwise distances.
    
    # However, a simpler and effective strategy for covering is to minimize
    # the maximum distance to the nearest neighbor (which is what we are doing
    # with the test points).
    
    # The directive suggests: "optimizing the coefficients to minimize the variance
    # of the pair correlation function".
    
    # Let's implement a pair correlation function based energy.
    # g(r) = <number of pairs at distance r> / <expected number for uniform>
    # Variance of g(r) should be minimized.
    
    # For simplicity and speed, let's use a Gaussian kernel based repulsion/attraction.
    # Minimize E = sum_{i != j} exp(-alpha * d_ij^2)
    # This pushes points apart (repulsion).
    # To ensure covering, we also need to prevent points from clumping to the center
    # of a hemisphere, which the repulsion handles.
    
    # Let's use a combination of repulsion and a term that encourages uniformity.
    
    # Actually, the most direct spectral method is to minimize the variance of the
    # projection of the points onto spherical harmonics.
    # For l=0, it's trivial (center of mass).
    # For l=1, it's the dipole moment. Minimizing |sum p_i|^2 ensures the points
    # are centered at the origin.
    # For l=2, it's the quadrupole moment.
    
    # Let's minimize the sum of |sum p_i|^2 for low l.
    # This is equivalent to minimizing the variance of the point distribution.
    
    # S = sum_{i=1}^N p_i
    # E = ||S||^2
    # This forces the centroid to be at the origin.
    
    # To go further, we can look at higher moments.
    # But for N=130, just centering might not be enough to optimize covering radius.
    
    # Let's stick to the test-point based objective but use a better optimization
    # strategy. The "spectral" idea might be implemented as a preprocessing step
    # or a regularizer.
    
    # Let's try a different structural change:
    # Use a "soft" covering radius objective with a high temperature, then anneal.
    # E = (1/M) * sum_y log( sum_i exp(-beta * d(y, p_i)^2) )
    # This is the free energy of a system of points. Minimizing it minimizes
    # the covering radius in the limit beta -> infinity.
    
    return 0.0 # Placeholder, we will use the test point objective primarily

def _soft_covering_radius(points, test_points, beta):
    # points: (N, 3)
    # test_points: (M, 3)
    # Compute d(y, p_i)^2
    diff = test_points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)  # (M, N)
    
    # exp(-beta * dists_sq)
    # To avoid underflow/overflow, subtract the min for each row
    min_dists_sq = np.min(dists_sq, axis=1, keepdims=True)
    exp_terms = np.exp(-beta * (dists_sq - min_dists_sq))
    
    # sum over i
    sums = np.sum(exp_terms, axis=1)
    
    # log(sums) + beta * min_dists_sq
    # E = (1/M) * sum_y ( -1/beta * log(sums) + min_dists_sq )
    # Actually, the free energy is F = -1/beta * log(Z)
    # Z = sum_y sum_i exp(-beta d^2) ? No, usually it's per point.
    
    # Let's use the standard soft-min:
    # soft_min_i d(y, p_i) = -1/beta * log( sum_i exp(-beta d(y, p_i)^2) )
    # Then covering radius approx = max_y soft_min(y)
    
    # But max is non-smooth. We can use a soft-max over y as well?
    # Or just minimize the average soft-min? No, we need the max.
    
    # Let's minimize the log-sum-exp of the soft-mins.
    # This approximates the max.
    
    # soft_min_vals = -1/beta * (log(sums) - log(N)) + min_dists_sq  # Approximation
    # Actually:
    # soft_min_d_sq = -1/beta * log( sum_i exp(-beta d_ij^2) )
    
    soft_min_d_sq = -1.0 / beta * np.log(sums + 1e-12) + min_dists_sq[:, 0]
    
    # Now we want to minimize the max of soft_min_d_sq over y.
    # Use log-sum-exp trick for max.
    max_val = np.max(soft_min_d_sq)
    # Soft max
    log_sum_exp = max_val + np.log(np.sum(np.exp(soft_min_d_sq - max_val)))
    
    return log_sum_exp

def _objective(flat_points, test_points, beta):
    points = _normalize(flat_points.reshape(N, 3))
    return _soft_covering_radius(points, test_points, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds to leave margin

    # Generate initial configuration
    points = _fibonacci_spiral(N)
    
    # Generate test points for covering radius approximation
    # Use a finer Fibonacci spiral
    n_test = 20000
    test_points = _fibonacci_spiral(n_test)
    
    # Initial objective
    beta = 10.0
    current_obj = _objective(points.flatten(), test_points, beta)
    best_points = points.copy()
    best_obj = current_obj
    
    # Optimization
    # We will use L-BFGS-B with the soft covering radius objective.
    # We will anneal beta to make the objective smoother and easier to optimize.
    
    try:
        # Phase 1: Low beta (smooth)
        beta = 5.0
        res = minimize(
            _objective,
            points.flatten(),
            args=(test_points, beta),
            method='L-BFGS-B',
            options={'maxiter': 50, 'maxfun': 100, 'gtol': 1e-4}
        )
        
        if res.fun < best_obj:
            best_points = _normalize(res.x.reshape(N, 3))
            best_obj = res.fun
            
        if time.time() > deadline:
            return best_points
            
        # Phase 2: Higher beta (sharper)
        beta = 20.0
        res2 = minimize(
            _objective,
            best_points.flatten(),
            args=(test_points, beta),
            method='L-BFGS-B',
            options={'maxiter': 50, 'maxfun': 100, 'gtol': 1e-5}
        )
        
        if res2.fun < best_obj:
            best_points = _normalize(res2.x.reshape(N, 3))
            best_obj = res2.fun
            
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    # Check time
    if time.time() > deadline:
        return best_points
    
    # Refine with the actual covering radius (hard max) if time permits
    # This ensures we are optimizing the true metric.
    if time.time() < deadline - 20:
        n_test_fine = 50000
        test_points_fine = _fibonacci_spiral(n_test_fine)
        
        def _hard_objective(flat_points):
            pts = _normalize(flat_points.reshape(N, 3))
            return _covering_radius(pts, test_points_fine)
            
        current_obj_hard = _hard_objective(best_points.flatten())
        
        try:
            res3 = minimize(
                _hard_objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 30, 'maxfun': 50, 'gtol': 1e-6}
            )
            
            if res3.fun < current_obj_hard:
                best_points = _normalize(res3.x.reshape(N, 3))
                
        except Exception as e:
            print(f"Refinement failed: {e}")
    
    return best_points
