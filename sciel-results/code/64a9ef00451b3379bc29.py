import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """
    Compute the actual score (max radius) for a set of points.
    """
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Boundary constraint
    norms = np.linalg.norm(points, axis=1)
    boundary_r = np.min(1.0 - norms)
    
    if boundary_r < 0:
        return 0.0
        
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(points)
    if len(d) == 0:
        pair_r = float('inf')
    else:
        pair_r = np.min(d) / 2.0

    r = min(pair_r, boundary_r)
    return max(0.0, r)

def _generate_hexagonal_lattice(spacing, n_target):
    """
    Generate a hexagonal lattice of points inside the unit disk.
    Select the n_target points closest to the origin.
    """
    k = int(1.0 / spacing) + 5
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                pts.append((x, y))
    
    if len(pts) < n_target:
        return None
    
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:n_target]
    return pts[idx]

def _spectral_params_to_points(params, K):
    """
    Convert spectral parameters to circle centers.
    params: shape (2*K+1, 2)
    K: number of harmonic modes to use (excluding DC)
    
    The centers are formed by summing K Fourier modes:
    c_n = sum_{k=1}^K a_k * exp(2*pi*i*k*n/N)
    
    We also include a DC offset term to allow global translation.
    """
    n = np.arange(N)
    # Compute frequencies: 2*pi*k*n/N
    # We want to compute sum_{k=1}^K a_k * exp(2*pi*i*k*n/N)
    
    # a is complex vector of shape (K, 2) essentially, but stored as real/imag parts in params
    # Let's structure params as:
    # params[0]: DC offset (real, imag)
    # params[1:K+1]: Coefficients for k=1..K (real, imag)
    
    dc = params[0]
    centers = np.tile(dc, (N, 1))
    
    # Precompute exponentials for speed? 
    # exp(2*pi*i*k*n/N)
    # We can compute this iteratively or vectorized.
    # Vectorized: 
    # k_values = np.arange(1, K+1)
    # n_values = n
    # phases = 2*np.pi * np.outer(n_values, k_values) / N
    # exps = np.exp(1j * phases) # Shape (N, K)
    
    # coeffs = params[1:K+1] # Shape (K, 2)
    # real_coeffs = coeffs[:, 0]
    # imag_coeffs = coeffs[:, 1]
    
    # contribution = exps * (real_coeffs + 1j*imag_coeffs)
    # centers += np.real(contribution).T + 1j*np.imag(contribution).T ... wait, centers are real 2D.
    
    # Let's do it carefully.
    # c_n = dc + Re( sum_k (a_k + i b_k) * (cos(theta_kn) + i sin(theta_kn)) )
    # c_n_x = dc_x + sum_k (a_k cos(theta) - b_k sin(theta))
    # c_n_y = dc_y + sum_k (a_k sin(theta) + b_k cos(theta))
    
    k_vals = np.arange(1, K+1)
    # phases shape (N, K)
    phases = 2 * np.pi * np.outer(n, k_vals) / N
    cos_phases = np.cos(phases) # (N, K)
    sin_phases = np.sin(phases) # (N, K)
    
    coeffs = params[1:K+1] # (K, 2)
    a = coeffs[:, 0] # (K,)
    b = coeffs[:, 1] # (K,)
    
    # c_x = dc_x + cos_phases @ a - sin_phases @ b
    # c_y = dc_y + sin_phases @ a + cos_phases @ b
    
    centers[:, 0] += cos_phases @ a - sin_phases @ b
    centers[:, 1] += sin_phases @ a + cos_phases @ b
    
    # Enforce hard radial cutoff to guarantee boundary containment
    # This is crucial for the spectral method as it might push points out.
    norms = np.linalg.norm(centers, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        centers[mask] = centers[mask] / norms[mask, np.newaxis]
        
    return centers

def _points_to_spectral_params(points, K):
    """
    Invert the spectral representation to get initial parameters.
    This is a least squares problem: minimize || P(params) - points ||^2
    Since the mapping is linear (plus the non-linear cutoff which we ignore for init),
    we can solve it directly if we ignore the cutoff or assume points are inside.
    
    We solve: points ~= DFT_matrix @ coeffs
    """
    n = np.arange(N)
    k_vals = np.arange(1, K+1)
    phases = 2 * np.pi * np.outer(n, k_vals) / N
    cos_phases = np.cos(phases)
    sin_phases = np.sin(phases)
    
    # Model:
    # p_x = dc_x + cos @ a - sin @ b
    # p_y = dc_y + sin @ a + cos @ b
    
    # This is a linear system for [dc_x, a, b] for x-coords and [dc_y, a, b] for y-coords?
    # No, a and b are shared between x and y?
    # Wait, the complex coefficient c_k = a_k + i b_k.
    # The contribution to x is Re(c_k * e^{i theta}) = a_k cos - b_k sin.
    # The contribution to y is Im(c_k * e^{i theta}) = a_k sin + b_k cos.
    # So yes, a and b are the same for both x and y components of the center.
    # This means the x and y coordinates are coupled.
    
    # Let's set up the full linear system.
    # Unknowns: dc_x, dc_y, a_1..a_K, b_1..b_K
    # Total unknowns: 2 + 2K
    # Equations: 2N (x and y for each point)
    
    # Matrix A shape (2N, 2+2K)
    # Row for point n, x-coord:
    # [1, 0, cos(1), -sin(1), cos(2), -sin(2), ...]
    # Row for point n, y-coord:
    # [0, 1, sin(1), cos(1), sin(2), cos(2), ...]
    
    # This matrix is large (600 x (2+2K)). If K=10, size is 600x22. Very small.
    
    # Construct A
    # We can use block structure.
    # A_x: N x (1+2K)
    # A_y: N x (1+2K)
    
    # Let's just build the full 2N x (2+2K) matrix.
    # Actually, it's easier to solve two separate least squares if we decouple?
    # No, a and b are shared.
    
    # Let's use np.linalg.lstsq.
    
    # Build A
    # Col 0: 1 for x-rows, 0 for y-rows
    # Col 1: 0 for x-rows, 1 for y-rows
    # Cols 2..2+K: a_k terms
    # Cols 2+K..2+2K: b_k terms
    
    # For x-row n:
    # a_k term: cos(theta_kn)
    # b_k term: -sin(theta_kn)
    
    # For y-row n:
    # a_k term: sin(theta_kn)
    # b_k term: cos(theta_kn)
    
    A = np.zeros((2*N, 2 + 2*K))
    
    # X rows (0 to N-1)
    A[0:N, 0] = 1.0
    A[0:N, 2:2+K] = cos_phases
    A[0:N, 2+K:2+2*K] = -sin_phases
    
    # Y rows (N to 2N-1)
    A[N:2*N, 1] = 1.0
    A[N:2*N, 2:2+K] = sin_phases
    A[N:2*N, 2+K:2+2*K] = cos_phases
    
    # RHS vector
    rhs = np.concatenate([points[:, 0], points[:, 1]])
    
    # Solve
    params, residuals, rank, s = np.linalg.lstsq(A, rhs, rcond=None)
    
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization with Hexagonal Lattice
    best_points = None
    best_score = 0.0
    
    spacings = [0.100, 0.104, 0.106, 0.108, 0.110]
    
    for spacing in spacings:
        if time.time() > deadline - 80.0:
            break
        pts = _generate_hexagonal_lattice(spacing, N)
        if pts is not None:
            score = _compute_score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                
    if best_points is None:
        best_points = np.random.uniform(-0.9, 0.9, size=(N, 2))
        best_score = _compute_score(best_points)

    # 2. Spectral Optimization
    # Choose K (number of harmonics). 
    # Too low K -> smooth, hard to fit tight packing.
    # Too high K -> approaches full dimensionality, loses spectral benefit.
    # Let's try K=15 or 20. 2+2K = 32 or 42 params.
    
    K = 20
    
    # Initialize params from best_points
    try:
        init_params = _points_to_spectral_params(best_points, K)
    except Exception:
        init_params = np.zeros(2 + 2*K)
        # If init fails, fallback to standard optimization? Or just return best_points.
        # Let's try to optimize anyway with zero init? No, that's bad.
        # Let's stick to the best_points found so far if spectral init fails.
        return best_points

    def spectral_objective_and_grad(params):
        points = _spectral_params_to_points(params, K)
        
        # Compute score components
        norms = np.linalg.norm(points, axis=1)
        boundary_vals = 1.0 - norms
        
        # Pairwise distances
        # Use full matrix for gradient ease
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx] / 2.0
        
        all_vals = np.concatenate([pair_dists, boundary_vals])
        
        # Softmin
        beta = 50.0
        neg_beta_v = -beta * all_vals
        max_neg_beta_v = np.max(neg_beta_v)
        exp_vals = np.exp(neg_beta_v - max_neg_beta_v)
        sum_exp = np.sum(exp_vals)
        
        softmin_val = -(max_neg_beta_v / beta + np.log(sum_exp) / beta)
        weights = exp_vals / sum_exp
        
        obj_val = -softmin_val
        
        # Gradient w.r.t points
        grad_points = np.zeros_like(points)
        
        n_pairs = len(pair_dists)
        w_pairs = weights[:n_pairs]
        w_bounds = weights[n_pairs:]
        
        # Pair gradients
        dists_safe = np.where(dists < 1e-8, 1e-8, dists)
        weight_matrix = np.zeros((N, N))
        weight_matrix[i_idx, j_idx] = w_pairs
        weight_matrix[j_idx, i_idx] = w_pairs
        
        term_pairs = weight_matrix[:, :, np.newaxis] * diff / (2.0 * dists_safe[:, :, np.newaxis])
        grad_points += -np.sum(term_pairs, axis=1)
        
        # Boundary gradients
        norms_safe = np.where(norms < 1e-8, 1e-8, norms)
        grad_points += w_bounds[:, np.newaxis] * points / norms_safe[:, np.newaxis]
        
        # Gradient w.r.t params
        # Chain rule: grad_params = J^T @ grad_points
        # J is the Jacobian of points w.r.t params.
        # From _spectral_params_to_points:
        # points[:, 0] = dc_x + cos @ a - sin @ b
        # points[:, 1] = dc_y + sin @ a + cos @ b
        
        # Params structure: [dc_x, dc_y, a_1..a_K, b_1..b_K]
        
        # d(points_x)/d(dc_x) = 1
        # d(points_y)/d(dc_y) = 1
        
        # d(points_x)/d(a_k) = cos(theta_kn)
        # d(points_x)/d(b_k) = -sin(theta_kn)
        
        # d(points_y)/d(a_k) = sin(theta_kn)
        # d(points_y)/d(b_k) = cos(theta_kn)
        
        # Note: This assumes the radial cutoff is not active.
        # If cutoff is active, the gradient is modified.
        # For now, we ignore the cutoff gradient in the chain rule, 
        # relying on the fact that if a point is on the boundary, 
        # the boundary constraint in the objective will push it in, 
        # and the cutoff just clips it. 
        # This is an approximation but often works well if the initial points are inside.
        # To be more robust, we could check if cutoff was active and adjust.
        # For simplicity and speed, we use the linear Jacobian.
        
        n = np.arange(N)
        k_vals = np.arange(1, K+1)
        phases = 2 * np.pi * np.outer(n, k_vals) / N
        cos_phases = np.cos(phases) # (N, K)
        sin_phases = np.sin(phases) # (N, K)
        
        grad_params = np.zeros_like(params)
        
        # DC terms
        grad_params[0] = np.sum(grad_points[:, 0])
        grad_params[1] = np.sum(grad_points[:, 1])
        
        # a terms (indices 2 to 2+K-1)
        # grad_params[2+k] = sum_n (cos_phases[n,k] * grad_points[n,0] + sin_phases[n,k] * grad_points[n,1])
        grad_a = cos_phases.T @ grad_points[:, 0] + sin_phases.T @ grad_points[:, 1]
        grad_params[2:2+K] = grad_a
        
        # b terms (indices 2+K to 2+2K-1)
        # grad_params[2+K+k] = sum_n (-sin_phases[n,k] * grad_points[n,0] + cos_phases[n,k] * grad_points[n,1])
        grad_b = -sin_phases.T @ grad_points[:, 0] + cos_phases.T @ grad_points[:, 1]
        grad_params[2+K:2+2*K] = grad_b
        
        return obj_val, grad_params

    # Run Optimization
    # L-BFGS-B is good for box constraints, but we have no box constraints on params directly.
    # However, we might want to bound them to prevent explosion.
    # Let's use L-BFGS-B with generous bounds.
    
    try:
        bounds = [(-1, 1)] * 2 + [(-1, 1)] * (2*K)
        
        result = minimize(
            spectral_objective_and_grad, 
            init_params, 
            method='L-BFGS-B', 
            jac=True, 
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        if result.success:
            new_params = result.x
            new_points = _spectral_params_to_points(new_params, K)
            new_score = _compute_score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                
    except Exception as e:
        pass

    # 3. Final Local Refinement with SLSQP on Cartesian coordinates
    # The spectral method might get stuck or lose precision.
    # A final short SLSQP run on the best spectral result can polish it.
    
    if time.time() < deadline - 20.0:
        try:
            x_current = best_points.flatten().copy()
            
            def cart_obj_and_grad(x):
                points = x.reshape(-1, 2)
                norms = np.linalg.norm(points, axis=1)
                boundary_vals = 1.0 - norms
                
                diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, np.inf)
                
                i_idx, j_idx = np.triu_indices(N, k=1)
                pair_dists = dists[i_idx, j_idx] / 2.0
                
                all_vals = np.concatenate([pair_dists, boundary_vals])
                
                beta = 100.0 # Higher beta for final polish
                neg_beta_v = -beta * all_vals
                max_neg_beta_v = np.max(neg_beta_v)
                exp_vals = np.exp(neg_beta_v - max_neg_beta_v)
                sum_exp = np.sum(exp_vals)
                
                softmin_val = -(max_neg_beta_v / beta + np.log(sum_exp) / beta)
                weights = exp_vals / sum_exp
                
                obj_val = -softmin_val
                
                grad_points = np.zeros_like(points)
                
                n_pairs = len(pair_dists)
                w_pairs = weights[:n_pairs]
                w_bounds = weights[n_pairs:]
                
                dists_safe = np.where(dists < 1e-8, 1e-8, dists)
                weight_matrix = np.zeros((N, N))
                weight_matrix[i_idx, j_idx] = w_pairs
                weight_matrix[j_idx, i_idx] = w_pairs
                
                term_pairs = weight_matrix[:, :, np.newaxis] * diff / (2.0 * dists_safe[:, :, np.newaxis])
                grad_points += -np.sum(term_pairs, axis=1)
                
                norms_safe = np.where(norms < 1e-8, 1e-8, norms)
                grad_points += w_bounds[:, np.newaxis] * points / norms_safe[:, np.newaxis]
                
                return obj_val, grad_points.flatten()

            constraints = []
            for i in range(N):
                def c(x, i=i):
                    points = x.reshape(-1, 2)
                    return 1.0 - np.linalg.norm(points[i])
                constraints.append({'type': 'ineq', 'fun': c})
                
            result2 = minimize(
                cart_obj_and_grad, 
                x_current, 
                method='SLSQP', 
                jac=True, 
                constraints=constraints, 
                options={'maxiter': 50, 'ftol': 1e-10, 'disp': False}
            )
            
            if result2.success:
                new_points = result2.x.reshape(-1, 2)
                new_score = _compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    
        except Exception:
            pass

    # Final sanity check
    if best_points is None or not np.all(np.isfinite(best_points)):
         angles = 2 * np.pi * np.arange(N) / N
         best_points = np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

    return best_points

# EVOLVE-BLOCK-END
