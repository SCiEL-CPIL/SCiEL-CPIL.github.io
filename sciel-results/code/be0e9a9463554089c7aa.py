# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

_DEADLINE = time.time() + 620.0
_K = 10
_NTH = 70
_NPHI = 130

# Precompute harmonics and weights
def _build_harmonics():
    """
    Returns:
    Y: (NPHI, NTH, 2K+1) array of spherical harmonic values
    w: (NTH,) weights for Gauss-Legendre quadrature
    dphi: (NPHI,) spacing in phi (constant)
    """
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    theta = np.arccos(mu)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    
    # Note: The quadrature weights for the double integral are w_theta * dphi
    # We will combine them into a single weight vector for each (theta, phi) point
    # But since dphi is constant, we can just multiply w by dphi later or now.
    # Let's create a full weight matrix W of shape (NPHI, NTH) where W[i, j] = w[j] * dphi
    W = np.outer(np.ones(_NPHI), w * dphi)

    Y = np.zeros((_NPHI, _NTH, 2 * k + 1), dtype=np.float64)
    for idx, m in enumerate(range(-k, k + 1)):
        am = abs(m)
        # Normalization constant N_m
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi)
                    * float(factorial(k - am, exact=True))
                    / float(factorial(k + am, exact=True)))
        
        # Associated Legendre polynomial P_l^m(x) with Condon-Shortley phase
        # scipy.special.lpmv(m, l, x) includes the Condon-Shortley phase (-1)^m
        P = lpmv(am, k, mu)
        
        if m == 0:
            Y[:, :, idx] = N * P
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * N * P * np.cos(m * phi)[:, None]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * N * P * np.sin(am * phi)[:, None]
            
    return Y, W

Y_GLOBAL, W_GLOBAL = _build_harmonics()

def _objective_and_grad(c):
    """
    Computes R(c) = ||f||_4 / ||f||_2 and its gradient.
    Since we constrain ||c||_2 = 1, ||f||_2 = 1, so we minimize ||f||_4.
    Returns:
    val: float, the value of ||f||_4
    grad: np.ndarray, the Euclidean gradient of ||f||_4
    """
    f = Y_GLOBAL @ c
    f2 = f * f
    f4 = f2 * f2
    
    S4 = np.sum(W_GLOBAL * f4)
    if S4 < 1e-12:
        return 1e12, np.zeros_like(c)
    
    val = S4 ** 0.25
    
    # Gradient of S4 w.r.t c
    # dS4/dc = sum_q W_q * 4 * f_q^3 * Y_q
    # grad_S4 = 4 * Y^T (W * f^3)
    f3 = f * f2
    grad_S4 = 4.0 * (Y_GLOBAL.T @ (W_GLOBAL * f3))
    
    # d(||f||_4)/dc = (1/4) * S4^(-3/4) * dS4/dc
    grad = (0.25 * S4 ** -0.75) * grad_S4
    
    return val, grad

def _riemannian_minimize(c0, max_iter=500, tol=1e-10):
    """
    Minimizes ||f||_4 on the unit sphere using Riemannian gradient descent
    with backtracking line search along geodesics.
    """
    c = c0 / np.linalg.norm(c0)
    val, grad = _objective_and_grad(c)
    
    # Project gradient onto tangent space
    # P_c(g) = g - (g^T c) c
    grad_proj = grad - np.dot(grad, c) * c
    grad_norm = np.linalg.norm(grad_proj)
    
    if grad_norm < tol:
        return c, val

    best_c = c.copy()
    best_val = val

    for i in range(max_iter):
        if time.time() > _DEADLINE - 30.0:
            break
            
        # Search direction is negative projected gradient
        d = -grad_proj
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-12:
            break
            
        d = d / d_norm
        
        # Line search along geodesic
        # c(t) = cos(t) * c - sin(t) * d
        # We want to find t > 0 that minimizes val(c(t))
        t = 1.0
        c_curr = c
        val_curr = val
        
        # Backtracking
        for _ in range(20):
            c_new = np.cos(t) * c_curr - np.sin(t) * d
            # Re-normalize to ensure we are on the sphere (numerical safety)
            c_new = c_new / np.linalg.norm(c_new)
            
            val_new, _ = _objective_and_grad(c_new)
            
            if val_new < val_curr:
                break
            t *= 0.5
            
        if t < 1e-8:
            break
            
        c = c_new
        val = val_new
        
        if val < best_val:
            best_val = val
            best_c = c.copy()
            
        # Check convergence
        val, grad = _objective_and_grad(c)
        grad_proj = grad - np.dot(grad, c) * c
        grad_norm = np.linalg.norm(grad_proj)
        
        if grad_norm < tol:
            break
            
    return best_c, best_val

def _make_inits(rng, k):
    """Generate diverse initializations."""
    inits = []
    dim = 2 * k + 1
    
    # 1. Single modes
    for m in range(-k, k + 1):
        v = np.zeros(dim)
        v[m + k] = 1.0
        inits.append(v)
        
    # 2. Pairs of conjugate modes (cos/sin)
    for m in range(1, k + 1):
        v = np.zeros(dim)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v / np.linalg.norm(v))
        
    # 3. m=0 mixed with high m
    for m in [k, k-1, k-2]:
        for weight in [0.1, 0.3, 0.5, 0.7, 0.9]:
            v = np.zeros(dim)
            v[0 + k] = weight
            v[m + k] = np.sqrt(1 - weight**2)
            v[-m + k] = np.sqrt(1 - weight**2)
            inits.append(v / np.linalg.norm(v))
            
    # 4. Random
    for _ in range(50):
        v = rng.standard_normal(dim)
        inits.append(v / np.linalg.norm(v))
        
    # 5. Sparse random
    for _ in range(30):
        v = np.zeros(dim)
        idx = rng.choice(dim, size=rng.integers(2, 8), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        inits.append(v / np.linalg.norm(v))
        
    # 6. Alternating signs
    v = np.ones(dim)
    v[::2] = -1.0
    inits.append(v / np.linalg.norm(v))
    
    # 7. Dominant high m with small noise
    for m in [k, k-1, -(k-1), -k]:
        v = np.zeros(dim)
        v[m + k] = 1.0
        v += 0.1 * rng.standard_normal(dim)
        inits.append(v / np.linalg.norm(v))

    return inits

def rudin_problem10() -> np.ndarray:
    k = _K
    rng = np.random.default_rng(42)
    inits = _make_inits(rng, k)

    best_c = inits[0].copy()
    best_val = _objective_and_grad(best_c)[0]
    
    # Phase 1: Optimize from diverse starts
    scored = []
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 60.0:
            break
        x0 = x0 / np.linalg.norm(x0)
        c, val = _riemannian_minimize(x0, max_iter=300, tol=1e-9)
        scored.append((val, c))
        if val < best_val:
            best_val = val
            best_c = c
            
    # Sort and keep top candidates
    scored.sort(key=lambda t: t[0])
    top_candidates = [c for _, c in scored[:10]]
    if not top_candidates:
        top_candidates = [best_c]

    # Phase 2: Refine top candidates with higher precision
    for x0 in top_candidates:
        if time.time() > _DEADLINE - 30.0:
            break
        c, val = _riemannian_minimize(x0, max_iter=1000, tol=1e-12)
        if val < best_val:
            best_val = val
            best_c = c
            
    # Phase 3: Random restarts around best
    while time.time() < _DEADLINE - 10.0:
        noise_scale = 0.2
        x0 = best_c + noise_scale * rng.standard_normal(2 * k + 1)
        x0 = x0 / np.linalg.norm(x0)
        c, val = _riemannian_minimize(x0, max_iter=500, tol=1e-11)
        if val < best_val:
            best_val = val
            best_c = c

    out = best_c.astype(np.float64)
    out = out / np.linalg.norm(out)
    
    # Final high-precision polish using scipy trust-constr if time permits
    if time.time() < _DEADLINE - 5.0:
        try:
            # Define objective and gradient for scipy
            def obj_scipy(c):
                return _objective_and_grad(c)[0]
            
            def grad_scipy(c):
                return _objective_and_grad(c)[1]
            
            # Add equality constraint ||c||^2 = 1
            cons = ({'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0, 
                     'jac': lambda c: 2.0 * c})
            
            res = minimize(obj_scipy, out, jac=grad_scipy, method='trust-constr', 
                           constraints=cons, options={'maxiter': 100, 'gtol': 1e-12})
            
            if res.success:
                res_c = res.x
                res_val = obj_scipy(res_c)
                if res_val < best_val:
                    best_val = res_val
                    best_c = res_c
        except Exception:
            pass

    out = best_c.astype(np.float64)
    out = out / np.linalg.norm(out)
    
    # Final validity check
    if np.any(np.isnan(out)) or np.any(np.isinf(out)):
        out = np.ones(21) / np.sqrt(21)
    if np.max(np.abs(out)) < 1e-3 or np.max(np.abs(out)) > 1e5:
        out = np.ones(21) / np.sqrt(21)
        
    return out

# EVOLVE-BLOCK-END
