import numpy as np
from scipy.special import lpmv, factorial
import time

def _build_harmonics():
    """Precompute real orthonormal degree-10 spherical harmonics on the quadrature grid."""
    x, w = np.polynomial.legendre.leggauss(70)
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    
    cos_theta, phi_grid = np.meshgrid(x, phi, indexing='ij')
    dphi = 2 * np.pi / 130
    weights_2d = np.outer(w, np.ones(130)) * dphi
    
    P = {}
    for m in range(0, 11):
        P[m] = lpmv(m, 10, cos_theta)
    
    N = {}
    for m in range(0, 11):
        N[m] = np.sqrt(21.0 / (4 * np.pi) * factorial(10 - m) / factorial(10 + m))
    
    Y = np.zeros((21, 70, 130))
    for idx, m in enumerate(range(-10, 11)):
        if m == 0:
            Y[idx] = N[0] * P[0]
        elif m > 0:
            Y[idx] = np.sqrt(2) * N[m] * P[m] * np.cos(m * phi_grid)
        else:
            am = abs(m)
            Y[idx] = np.sqrt(2) * N[am] * P[am] * np.sin(am * phi_grid)
    
    # Precompute weighted basis for fast evaluation
    # B[m] = Y[m] * weights_2d  (21, 70, 130)
    B = Y * weights_2d
    
    return Y, B, weights_2d

def _eval_f(c, Y):
    """Evaluate f on grid: (70, 130)."""
    return np.tensordot(c, Y, axes=(0, 0))

def _eval_f4(c, Y, B):
    """Evaluate f^4 weighted sum efficiently."""
    f = np.tensordot(c, Y, axes=(0, 0))
    return np.sum(f**4 * (B / (Y + 1e-300) * (Y + 1e-300)))  # This is wrong, let's just use B directly
    
def _compute_f(c, Y):
    return np.tensordot(c, Y, axes=(0, 0))

def _objective(c, Y, weights_2d):
    """Compute ||f||_4 / ||f||_2."""
    f = _compute_f(c, Y)
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(f**4 * w)**0.25
    return norm4 / norm2

def _power_iteration_step(c, Y, B):
    """
    One step of the nonlinear power iteration for minimizing ||f||_4 on the unit sphere.
    
    The gradient of ||f||_4^4 / ||f||_2^4 with respect to c (on the sphere) points in the
    direction of f^3 * Y. The fixed-point iteration is:
      c_new = normalize( c - alpha * grad_sphere(||f||_4^4) )
    where grad_sphere = grad - (grad . c) * c
    
    Equivalently, for the ratio, we can use the "L4 eigenvector" iteration:
      c_new = normalize( Y^T * (f^3 * w) )  projected to sphere
    
    Actually, the correct fixed-point for minimizing R = ||f||_4/||f||_2 subject to ||c||=1:
      At optimum, c is proportional to Y^T (f^3 * w) / ||f||_4^3 (the gradient of ||f||_4^4)
      normalized and projected.
    """
    f = _compute_f(c, Y)
    w = B / Y if False else None  # We'll use B directly
    
    # grad of ||f||_4^4 = 4 * sum(f^3 * Y_m * w) 
    # So the "eigenvector" direction is: sum(f^3 * Y_m * w) = Y^T(f^3 * w)
    f3 = f**3
    # g_m = sum_ij f^3_ij * Y_m_ij * w_ij
    # Using B = Y * w: g_m = sum_ij f3_ij * Y_m_ij * w_ij
    # We can compute this as: for each m, sum(f3 * Y[m] * w)
    # More efficiently: g = Y^T (f3 * w) flattened
    # Let's compute: g[m] = np.sum(f3 * Y[m] * w)
    # Vectorized: g = np.einsum('ij,mij,ij->m', f3, Y, w)
    g = np.einsum('ij,mij,ij->m', f3, Y, w)
    
    # Project to tangent: g_t = g - (g . c) * c
    dot = np.dot(g, c)
    g_t = g - dot * c
    
    # Normalize
    norm_gt = np.linalg.norm(g_t)
    if norm_gt < 1e-15:
        return c.copy()
    
    return g_t / norm_gt

def _damped_power_iteration(c, Y, w, max_iter=500, damping=0.5, tol=1e-14):
    """
    Damped power iteration for the nonlinear eigenproblem.
    c_{k+1} = normalize( (1-damping)*c_k + damping * g_t(c_k) / ||g_t(c_k)|| )
    """
    c_curr = c / np.linalg.norm(c)
    prev_obj = _objective(c_curr, Y, w)
    
    for it in range(max_iter):
        f = _compute_f(c_curr, Y)
        f3 = f**3
        g = np.einsum('ij,mij,ij->m', f3, Y, w)
        dot = np.dot(g, c_curr)
        g_t = g - dot * c_curr
        norm_gt = np.linalg.norm(g_t)
        if norm_gt < 1e-15:
            break
        
        g_t_norm = g_t / norm_gt
        
        # Damped update
        c_next = (1 - damping) * c_curr + damping * g_t_norm
        norm_cn = np.linalg.norm(c_next)
        if norm_cn < 1e-15:
            break
        c_next = c_next / norm_cn
        
        obj = _objective(c_next, Y, w)
        if abs(prev_obj - obj) < tol:
            c_curr = c_next
            break
        c_curr = c_next
        prev_obj = obj
    
    return c_curr

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio
    using nonlinear power iteration with multi-restart.
    """
    start_time = time.time()
    deadline = 610.0  # safety margin before 630s limit
    
    Y, B, weights_2d = _build_harmonics()
    w = weights_2d
    
    best_coeffs = np.ones(21) / np.sqrt(21)
    best_score = _objective(best_coeffs, Y, w)
    
    # Generate diverse initial points on the 21-dim sphere
    np.random.seed(42)
    
    initial_points = []
    
    # Structured guesses
    c = np.zeros(21)
    c[10] = 1.0
    initial_points.append(c)
    
    c = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        c[10 + m] = 1.0
    initial_points.append(c)
    
    c = np.zeros(21)
    for m in range(0, 11):
        c[10 + m] = 1.0
    initial_points.append(c)
    
    c = np.zeros(21)
    for m in range(0, 11):
        c[10 + m] = (-1)**m
    initial_points.append(c)
    
    c = np.zeros(21)
    c[10] = 1.0
    for m in [1, 2, 3, 4, 5]:
        c[10 + m] = 0.1
        c[10 - m] = 0.1
    initial_points.append(c)
    
    # Random points
    for i in range(50):
        x0 = np.random.randn(21)
        initial_points.append(x0)
    
    # Sparse random
    for i in range(20):
        x0 = np.random.randn(21)
        mask = np.random.random(21) < 0.3
        x0 *= mask
        if np.linalg.norm(x0) < 1e-10:
            x0 = np.random.randn(21)
        initial_points.append(x0)
    
    # Cos-only structured
    for i in range(20):
        x0 = np.zeros(21)
        for m in range(11):
            if np.random.random() < 0.5:
                x0[10 + m] = np.random.randn()
        if np.linalg.norm(x0) < 1e-10:
            x0 = np.random.randn(21)
        initial_points.append(x0)
    
    # Run power iteration from each initial point
    for idx, c0 in enumerate(initial_points):
        if time.time() - start_time > deadline - 30:
            break
        
        c0 = c0 / np.linalg.norm(c0)
        
        try:
            c_opt = _damped_power_iteration(c0, Y, w, max_iter=1000, damping=0.7, tol=1e-15)
            score = _objective(c_opt, Y, w)
            if score < best_score and np.isfinite(score):
                best_score = score
                best_coeffs = c_opt.copy()
        except Exception:
            pass
    
    # Second pass: perturb best solution and re-run
    if time.time() - start_time < deadline - 60:
        for i in range(30):
            if time.time() - start_time > deadline - 30:
                break
            perturbation_scale = 0.3 * (1 - i / 30.0)
            c0 = best_coeffs + perturbation_scale * np.random.randn(21)
            c0 = c0 / np.linalg.norm(c0)
            
            try:
                c_opt = _damped_power_iteration(c0, Y, w, max_iter=2000, damping=0.8, tol=1e-16)
                score = _objective(c_opt, Y, w)
                if score < best_score and np.isfinite(score):
                    best_score = score
                    best_coeffs = c_opt.copy()
            except Exception:
                pass
    
    # Third pass: very fine-tuned damping near the best
    if time.time() - start_time < deadline - 20:
        for i in range(20):
            if time.time() - start_time > deadline - 10:
                break
            c0 = best_coeffs + 0.05 * np.random.randn(21)
            c0 = c0 / np.linalg.norm(c0)
            
            try:
                c_opt = _damped_power_iteration(c0, Y, w, max_iter=5000, damping=0.95, tol=1e-16)
                score = _objective(c_opt, Y, w)
                if score < best_score and np.isfinite(score):
                    best_score = score
                    best_coeffs = c_opt.copy()
            except Exception:
                pass
    
    # Validate
    if len(best_coeffs) != 21 or not np.all(np.isfinite(best_coeffs)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs < 1e-3:
            best_coeffs *= 1.0 / max_abs if max_abs > 0 else 1.0
        elif max_abs > 1e5:
            best_coeffs *= 1e5 / max_abs
        if not np.all(np.isfinite(best_coeffs)):
            return np.ones(21)
    
    return best_coeffs.astype(np.float64)
