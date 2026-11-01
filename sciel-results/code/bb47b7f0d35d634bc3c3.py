import numpy as np
import time
import math

N = 150

def _energy_and_grad(X):
    """
    Compute logarithmic energy and Riemannian gradient.
    Energy = sum_{i<j} log(1/|x_i - x_j|)
    """
    n = X.shape[0]
    # Pairwise differences
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    # Avoid division by zero
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient
    # grad_eucl[i] = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # So (x_j - x_i) = -diff[i,j]
    # grad_eucl[i] = - sum_j diff[i,j] / dist2[i,j]
    
    inv_dist2 = 1.0 / dist2
    # Zero out diagonal to be safe, though diff is 0 there
    np.fill_diagonal(inv_dist2, 0.0)
    
    eucl_grad = -np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    # P_i = I - x_i x_i^T
    # grad_riem[i] = P_i @ eucl_grad[i]
    dot = np.sum(eucl_grad * X, axis=1, keepdims=True)
    riem_grad = eucl_grad - dot * X
    
    return energy, riem_grad

def _project_to_sphere(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _fibonacci_init(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _halton_init(n, num_dims=3):
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, 3))
    for i in range(1, n + 1):
        points[i - 1] = [halton(i, 2), halton(i, 3), halton(i, 5)]
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    return points

def _riemannian_cg(X0, max_iter=2000, tol=1e-12, time_limit=10.0):
    """
    Riemannian Conjugate Gradient with Polak-Ribiere beta and Backtracking Line Search.
    """
    start_time = time.time()
    X = X0.copy()
    energy, grad = _energy_and_grad(X)
    
    # Initial search direction
    d = -grad
    d_norm = np.linalg.norm(d)
    
    if d_norm < tol:
        return X, energy
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Backtracking Line Search
        alpha = 1.0
        c1 = 1e-4  # Armijo
        rho = 0.5
        
        accepted = False
        for _ in range(30):
            # Move in direction d
            # X_new = exp_X(alpha * d)
            # For small alpha, X_new approx X + alpha * d, then project.
            # Better: Use geodesic step.
            # exp_X(v) = X cos(||v||) + (X^T v / ||v||) sin(||v||)
            # Here v = alpha * d.
            # ||v|| = alpha * ||d||
            
            v_norm = alpha * d_norm
            if v_norm < 1e-10:
                X_new = X
            else:
                cos_v = np.cos(v_norm)
                sin_v = np.sin(v_norm)
                # (X^T v) / ||v|| = (X^T (alpha d)) / (alpha ||d||) = (X^T d) / ||d||
                # Note: X^T d is scalar for each point i.
                # X is (n,3), d is (n,3).
                # X^T d for each i is sum(X[i] * d[i])
                x_dot_d = np.sum(X * d, axis=1) # shape (n,)
                # We need to divide by ||d|| (which is d_norm, scalar? No, d_norm is global norm? No, d is a matrix of vectors.
                # Wait, d is a collection of tangent vectors.
                # The step is applied independently to each point.
                # So for each point i, we move along d[i].
                # v_i = alpha * d[i].
                # ||v_i|| = alpha * ||d[i]||.
                
                d_norms = np.linalg.norm(d, axis=1) # shape (n,)
                v_norms = alpha * d_norms
                
                # Handle zeros
                safe_v_norms = np.where(v_norms > 1e-10, v_norms, 1.0)
                cos_v = np.cos(safe_v_norms)
                sin_v = np.sin(safe_v_norms)
                
                # term2 = (X^T d) / ||d|| * sin(v)
                # X^T d is sum(X[i] * d[i])
                # d_norm is ||d[i]||
                # term2[i] = (sum(X[i]*d[i]) / ||d[i]||) * sin(alpha * ||d[i]||)
                
                # We must be careful if d[i] is 0.
                # If d[i] is 0, v[i] is 0, cos=1, sin=0. X_new = X.
                
                coeff = np.zeros_like(d_norms)
                mask = d_norms > 1e-10
                coeff[mask] = (x_dot_d[mask] / d_norms[mask]) * sin_v[mask]
                
                X_new = X * cos_v[:, np.newaxis] + d * (coeff / d_norms[:, np.newaxis])[:, np.newaxis]
                # Wait, the formula is X cos(v) + (X^T v / ||v||) sin(v)
                # v = alpha d.
                # X^T v / ||v|| = (alpha X^T d) / (alpha ||d||) = X^T d / ||d||
                # So term is (X^T d / ||d||) sin(alpha ||d||)
                # My coeff above is (X^T d / ||d||) * sin(alpha ||d||).
                # So X_new = X * cos(alpha ||d||) + d * (coeff / ||d||)
                
                # Let's re-verify the vectorized geodesic step.
                # x_new_i = x_i cos(theta_i) + (x_i^T d_i / ||d_i||) sin(theta_i)
                # where theta_i = alpha ||d_i||
                # This is correct.
                
            # Project to sphere just in case of numerical drift
            X_new = _project_to_sphere(X_new)
            
            energy_new, _ = _energy_and_grad(X_new)
            
            # Armijo condition: f(x_new) <= f(x) + c1 * alpha * grad^T d
            # grad^T d = sum_i grad[i] . d[i]
            grad_dot_d = np.sum(grad * d)
            
            if energy_new <= energy + c1 * alpha * grad_dot_d:
                accepted = True
                break
            else:
                alpha *= rho
        
        if not accepted:
            # If line search fails, break or reduce step drastically
            break
            
        # Compute new gradient
        energy_prev = energy
        energy = energy_new
        grad_prev = grad
        grad, _ = _energy_and_grad(X_new)
        
        # Polak-Ribiere Beta
        # beta = (grad_new^T (grad_new - grad_prev)) / (grad_prev^T grad_prev)
        # Clip beta to [0, 1]
        
        grad_diff = grad - grad_prev
        numerator = np.sum(grad * grad_diff)
        denominator = np.sum(grad_prev * grad_prev)
        
        if denominator > 1e-16:
            beta = numerator / denominator
        else:
            beta = 0.0
            
        beta = max(0.0, min(beta, 1.0))
        
        # New direction
        d = -grad + beta * d
        
        # Check convergence
        d_norm = np.linalg.norm(d)
        if d_norm < tol:
            break
            
        X = X_new
        
    return X, energy

def _farthest_point_init(n, rng_seed=42):
    """Farthest-point sampling initialization for better diversity."""
    rng = np.random.RandomState(rng_seed)
    points = np.zeros((n, 3))
    
    # Start with a random point
    points[0] = rng.randn(3)
    points[0] /= np.linalg.norm(points[0])
    
    min_dists = np.full(n, np.inf)
    
    for i in range(1, n):
        # Compute distance from each candidate to all existing points
        # Use the point farthest from all existing points
        best_idx = 0
        best_min_dist = -1
        
        for _ in range(100):  # Try 100 random candidates
            candidate = rng.randn(3)
            candidate /= np.linalg.norm(candidate)
            
            # Compute minimum distance to existing points
            dists = np.linalg.norm(points[:i] - candidate, axis=1)
            min_dist = np.min(dists)
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = candidate
        
        points[i] = best_idx
    
    return points

def _simulated_annealing(X0, time_limit=5.0, rng_seed=42):
    """
    Simulated Annealing on the sphere.
    Uses a distance matrix to avoid redundant calculations.
    """
    start_time = time.time()
    rng = np.random.RandomState(rng_seed)
    X = X0.copy()
    
    # Precompute pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    
    # Energy
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Cooling schedule
    T_start = 0.1
    T_end = 0.0001
    n_iters = 5000
    
    for k in range(n_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Temperature
        t = k / n_iters
        T = T_start * (T_end / T_start) ** t
        
        # Pick two random points to swap or move
        i, j = rng.randint(0, N, size=2)
        if i == j:
            continue
            
        # Move point i towards point j or away
        # Random perturbation
        noise = rng.randn(3) * 0.01
        X_new = X.copy()
        X_new[i] = X[i] + noise
        X_new[i] = X_new[i] / np.linalg.norm(X_new[i])
        
        # Update distance matrix incrementally
        # Only row i and column i change
        diff_i = X_new[i] - X
        new_dist2_i = np.sum(diff_i * diff_i, axis=1)
        new_dist_i = np.sqrt(new_dist2_i)
        
        # Compute new energy
        # Old energy contribution of i: -sum_{j!=i} log(dist[i,j])
        old_contrib = -np.sum(np.log(np.where(i != np.arange(N), dist[i], 1.0)))
        
        # New energy contribution of i
        new_contrib = -np.sum(np.log(np.where(i != np.arange(N), new_dist_i, 1.0)))
        
        delta_e = new_contrib - old_contrib
        
        # Metropolis criterion
        if delta_e < 0 or rng.rand() < math.exp(-delta_e / T):
            # Accept
            X = X_new
            energy += delta_e
            
            # Update distance matrix
            dist[i, :] = new_dist_i
            dist[:, i] = new_dist_i
            np.fill_diagonal(dist, 0.0) # Keep diagonal 0 for safety, though we use 1.0 in log? No, log(1) = 0.
            # Actually, we need to be careful with the diagonal.
            # In _energy_and_grad we set diagonal to 1.0.
            # Here, we only sum over i!=j.
            # So diagonal value doesn't matter for energy, but for consistency let's keep it 0 or 1.
            # Let's set it to 1.0 to match _energy_and_grad if we were to call it.
            np.fill_diagonal(dist, 1.0)

    return X, energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    initializations = []
    
    # 1. Fibonacci
    initializations.append(_fibonacci_init(N))
    
    # 2. Halton
    initializations.append(_halton_init(N))
    
    # 3. Farthest Point Sampling
    initializations.append(_farthest_point_init(N, rng_seed=42))
    
    # 4. Random
    rng = np.random.RandomState(42)
    rand_pts = rng.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts))
    
    # 5. Random 2
    rng2 = np.random.RandomState(123)
    rand_pts2 = rng2.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts2))
    
    best_energy = np.inf
    best_points = initializations[0]
    
    # Phase 1: RCG from multiple starts
    for idx, X0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
            
        time_budget = (deadline - 20.0 - time.time()) / max(1, len(initializations) - idx)
        time_budget = max(time_budget, 5.0)
        
        try:
            X_opt, e_opt = _riemannian_cg(X0, max_iter=1000, tol=1e-12, time_limit=time_budget)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = X_opt.copy()
                
        except Exception:
            pass

    # Phase 2: Simulated Annealing Polish
    # Use SA to escape local minima, then RCG again
    if time.time() < deadline - 10.0:
        try:
            sa_budget = min(10.0, deadline - 10.0 - time.time())
            X_sa, e_sa = _simulated_annealing(best_points, time_limit=sa_budget, rng_seed=42)
            
            # RCG from SA result
            X_opt, e_opt = _riemannian_cg(X_sa, max_iter=1000, tol=1e-12, time_limit=10.0)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = X_opt.copy()
                
        except Exception:
            pass

    # Phase 3: Final Polish
    # Use L-BFGS-B on the best result found by RCG to squeeze out more precision
    if time.time() < deadline - 5.0:
        try:
            from scipy.optimize import minimize
            
            def objective(x_flat):
                X = x_flat.reshape(N, 3)
                X = _project_to_sphere(X)
                e, _ = _energy_and_grad(X)
                return e
            
            def gradient(x_flat):
                X = x_flat.reshape(N, 3)
                X = _project_to_sphere(X)
                e, g = _energy_and_grad(X)
                return g.flatten()
                
            result = minimize(
                objective,
                best_points.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14, 'm': 20}
            )
            
            X_polished = _project_to_sphere(result.x.reshape(N, 3))
            e_polished, _ = _energy_and_grad(X_polished)
            
            if e_polished < best_energy:
                best_energy = e_polished
                best_points = X_polished
                
        except Exception:
            pass

    # Final validation
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.sqrt(dist2[iu]).min()
    
    if min_dist < 1e-9:
        return _fibonacci_init(N)
        
    return best_points
