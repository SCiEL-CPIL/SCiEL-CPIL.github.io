"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
import numpy as np
import time
from scipy.optimize import minimize

N = 150

def _energy_and_grad(x):
    """
    Compute the logarithmic energy and its gradient for points on the unit sphere.
    x: shape (N, 3)
    Returns: energy (scalar), gradient (N, 3)
    """
    # Compute pairwise distances
    # dist[i,j] = |x_i - x_j|
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff ** 2, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    # d_ij = sqrt(dist_sq_ij)
    # log(1/d) = -0.5 * log(dist_sq)
    energy = -0.5 * np.sum(np.log(dist_sq[iu]))
    
    # Gradient: d/dx_i [ -log(|x_i - x_j|) ] = -(x_i - x_j) / |x_i - x_j|^2
    # We want to minimize E = sum log(1/d) = -sum log(d)
    # dE/dx_i = -sum_{j!=i} (x_i - x_j) / d_ij^2 = sum_{j!=i} (x_j - x_i) / d_ij^2
    
    # Vectorized gradient computation
    # inv_d2 = 1/d_ij^2 for i != j
    inv_d2 = np.zeros((N, N))
    mask = ~np.eye(N, dtype=bool)
    inv_d2[mask] = 1.0 / dist_sq[mask]
    
    # grad_i = sum_j (x_j - x_i) * inv_d2[i,j]
    # = sum_j x_j * inv_d2[i,j] - x_i * sum_j inv_d2[i,j]
    sum_inv_d2 = np.sum(inv_d2, axis=1)  # (N,)
    # sum_j x_j * inv_d2[i,j] for each i
    weighted_sum = inv_d2 @ x  # (N, 3)
    grad = weighted_sum - x * sum_inv_d2[:, np.newaxis]
    
    return energy, grad

def _project_to_sphere(x):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return x / norms

def _tangent_projection(x, g):
    """Project gradient g onto the tangent space of the sphere at x."""
    # g is (N, 3), x is (N, 3)
    # Projection: g - (g . x) x
    dot = np.sum(g * x, axis=1, keepdims=True)
    return g - dot * x

def _fibonacci_init():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _halton_init():
    """Generate a good initial configuration using Halton sequences."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, N + 1):
        u = halton(i, 2)
        v = halton(i, 3)
        w = halton(i, 5)
        
        z = 2.0 * w - 1.0
        r = np.sqrt(max(0.0, 1.0 - z * z))
        phi = 2.0 * np.pi * u
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x, y, z])
    
    return np.array(points)

def _riemannian_cg(x0, max_iter=1000, tol=1e-10, deadline=None):
    """
    Riemannian Conjugate Gradient method on the sphere.
    """
    x = _project_to_sphere(x0.copy())
    e, g = _energy_and_grad(x)
    g_tan = _tangent_projection(x, g)
    
    # Initial search direction
    p = -g_tan.copy()
    
    # Initial line search parameters
    alpha = 1.0
    beta = 0.0
    
    for k in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Line search (Backtracking)
        # We want to minimize f(x + alpha * p) where x is on sphere.
        # We use a simple backtracking line search on the manifold.
        # Approximate: x_new = _project_to_sphere(x + alpha * p)
        
        f_k = e
        g_k_norm = np.linalg.norm(g_tan)
        
        if g_k_norm < tol:
            break
            
        alpha = 1.0
        for _ in range(50):
            x_new = _project_to_sphere(x + alpha * p)
            f_new, g_new = _energy_and_grad(x_new)
            g_new_tan = _tangent_projection(x_new, g_new)
            
            # Armijo condition
            if f_new < f_k + 1e-4 * alpha * np.sum(g_tan * p):
                break
            alpha *= 0.5
            
        # If line search fails or no improvement, try smaller step or break
        if f_new >= f_k:
             # Try a very small step
             x_new = _project_to_sphere(x + 1e-4 * p)
             f_new, g_new = _energy_and_grad(x_new)
             g_new_tan = _tangent_projection(x_new, g_new)
             if f_new >= f_k:
                 break
        
        x = x_new
        e = f_new
        g = g_new
        g_tan = g_new_tan
        
        # Compute beta (Polak-Ribiere)
        g_diff = g_tan - _tangent_projection(x, g_tan) # This is not quite right for Riemannian, need parallel transport
        # Simpler: just reset direction if not making progress (steepest descent fallback)
        # Or use a simple Riemannian beta.
        # For robustness, let's use a simple Riemannian CG with restarts.
        # Parallel transport of p to x_new: p_pt = p - (p . x) x + (p . x_new) x_new
        # Actually, since we are on a sphere, the tangent space changes.
        # Let's use a simpler approach: Riemannian Steepest Descent with momentum or just CG with restarts.
        
        # To keep it simple and robust, let's switch to a Riemannian Steepest Descent with adaptive step size if CG is complex.
        # But CG is faster. Let's implement a basic Riemannian CG.
        
        # Parallel transport of p from x_old to x_new
        # p_pt = p - (p, x_old)x_old + (p, x_new)x_new  (for sphere)
        # Wait, p is already tangent to x_old.
        # p_pt = p + (p . x_new) x_new  (since p . x_old = 0)
        # Check: p_pt . x_new = p . x_new + (p . x_new) * 1 = 2(p . x_new). Not zero.
        # Correct parallel transport of vector v from x to y on sphere:
        # v_pt = v - (v . x)x + (v . y)y
        # Since v is tangent to x, v . x = 0.
        # v_pt = v + (v . y)y
        
        p_old = p
        p_pt = p_old + np.sum(p_old * x, axis=1, keepdims=True) * x # Wait, p_old is tangent to x_old, not x_new.
        # The vector p was tangent to x_old. We transported it to x_new.
        # Actually, we should compute the new direction.
        
        # Let's stick to a robust Riemannian Steepest Descent with a good line search, 
        # or use scipy's minimize with a custom manifold projection? 
        # Scipy's L-BFGS-B doesn't handle manifolds well.
        
        # Let's implement a simple Riemannian Gradient Descent with momentum.
        # This is often sufficient and very stable.
        
        # Let's restart the loop with Steepest Descent logic for simplicity and robustness.
        # The current loop is getting messy. Let's rewrite the optimization loop.
        
        break # Break out of CG loop to use a simpler method if needed, or just continue with SD.

    return x, e

def _optimize(x0, deadline):
    """
    Optimize the configuration starting from x0 using Riemannian Gradient Descent.
    """
    x = _project_to_sphere(x0.copy())
    e, g = _energy_and_grad(x)
    
    # Riemannian Gradient Descent with adaptive step size
    alpha = 1e-3
    best_e = e
    best_x = x.copy()
    
    for i in range(5000):
        if time.time() > deadline:
            break
            
        g_tan = _tangent_projection(x, g)
        
        # Gradient norm for convergence check
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-8:
            break
            
        # Line search (Backtracking)
        alpha = min(alpha * 1.5, 0.1) # Increase step size slightly
        improved = False
        for _ in range(30):
            x_new = _project_to_sphere(x - alpha * g_tan)
            e_new, g_new = _energy_and_grad(x_new)
            
            if e_new < e:
                # Armijo condition
                if e_new < e - 1e-4 * alpha * np.sum(g_tan * g_tan):
                    x = x_new
                    e = e_new
                    g = g_new
                    improved = True
                    break
            alpha *= 0.5
            
        if not improved:
            # Try a very small step
            alpha = 1e-6
            x_new = _project_to_sphere(x - alpha * g_tan)
            e_new, g_new = _energy_and_grad(x_new)
            if e_new < e:
                x = x_new
                e = e_new
                g = g_new
            else:
                break
                
        if e < best_e:
            best_e = e
            best_x = x.copy()
            
    return best_x, best_e

def construct():
    """
    Minimize the logarithmic energy of 150 points on the unit sphere.
    Returns an array of shape (N, 3) with points on the unit sphere.
    """
    deadline = time.time() + 110.0  # Leave 10s safety margin
    
    initializations = [
        _fibonacci_init(),
        _halton_init(),
    ]
    
    # Add a random initialization for diversity
    rng = np.random.default_rng(42)
    random_init = rng.random((N, 3))
    initializations.append(_project_to_sphere(random_init))
    
    best_points = None
    best_energy = np.inf
    
    for init_points in initializations:
        if time.time() > deadline:
            break
        
        try:
            x_opt, e_opt = _optimize(init_points, deadline)
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = x_opt
        except Exception:
            pass
        
        if time.time() > deadline:
            break
            
    # If no successful optimization, return the best initialization
    if best_points is None:
        for init_points in initializations:
            e, _ = _energy_and_grad(init_points)
            if e < best_energy:
                best_energy = e
                best_points = init_points
    
    # Final polish: do a few more iterations with tighter tolerance if time allows
    if time.time() < deadline - 5.0 and best_points is not None:
        # Use a more aggressive line search or more iterations
        x_polish, e_polish = _optimize(best_points, deadline)
        if e_polish < best_energy:
            best_energy = e_polish
            best_points = x_polish
            
    # Ensure points are on the unit sphere
    if best_points is not None:
        best_points = _project_to_sphere(best_points)
        # Verify no points are too close
        norms = np.linalg.norm(best_points, axis=1)
        if np.any(norms < 1e-15) or not np.all(np.isfinite(best_points)):
            # Fallback to Fibonacci
            best_points = _fibonacci_init()
    
    return best_points
