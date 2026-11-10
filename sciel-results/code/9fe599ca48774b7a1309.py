# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

_DEADLINE = time.time() + 620.0
_K = 10
_NTH = 70
_NPHI = 130


def _build_harmonics():
    """Return Y flattened (N, 2k+1), G matrix, and weights for quadrature."""
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    theta = np.arccos(mu)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    # Build Y as (nphi*nth, 2k+1)
    Y = np.zeros((_NPHI * _NTH, 2 * k + 1), dtype=np.float64)
    for idx, m in enumerate(range(-k, k + 1)):
        am = abs(m)
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi)
                    * float(factorial(k - am, exact=True))
                    / float(factorial(k + am, exact=True)))
        P = lpmv(am, k, mu)
        if m == 0:
            Y[:, idx] = N * P[np.newaxis, :] * np.ones(_NPHI)[:, np.newaxis]
        elif m > 0:
            cos_term = np.cos(m * phi)
            Y[:, idx] = (np.sqrt(2.0) * N * P[np.newaxis, :] * cos_term[:, np.newaxis]).ravel()
        else:
            sin_term = np.sin(am * phi)
            Y[:, idx] = (np.sqrt(2.0) * N * P[np.newaxis, :] * sin_term[:, np.newaxis]).ravel()
    
    # Build quadrature weight matrix: each (phi, theta) point has weight w_theta * dphi
    # Y is (nphi*nth, dim), weights are (nphi*nth,)
    weights = np.tile(w, _NPHI) * dphi
    
    # G = Y^T W Y where W = diag(weights)
    YW = Y * weights[:, np.newaxis]
    G = Y.T @ YW
    
    return Y, weights, G


def _objective_and_grad(c, Y, weights):
    """Compute R(c) = ||f||_4 / ||f||_2 and its gradient.
    
    Since the basis is orthonormal, ||f||_2^2 = c^T G c.
    On the unit sphere (c^T G c = 1), R = ||f||_4.
    We minimize ||f||_4^4 = sum(f^4 * weights) for numerical stability.
    """
    f = Y @ c
    f2 = f * f
    f4 = f2 * f2
    
    n4 = np.sum(f4 * weights)
    n2 = np.sum(f2 * weights)
    
    if n2 < 1e-15:
        return 1e12, np.zeros_like(c)
    
    R = (n4 ** 0.25) / np.sqrt(n2)
    
    # Gradient of R with respect to c
    # R = (sum f^4 w)^(1/4) / (sum f^2 w)^(1/2)
    # dR/dc = (1/4) (sum f^4 w)^(-3/4) * 4 f^3 w Y^T / (sum f^2 w)^(1/2)
    #        - (sum f^4 w)^(1/4) * (1/2) (sum f^2 w)^(-3/2) * 2 f w Y^T
    #       = (sum f^4 w)^(-3/4) * (sum f^2 w)^(-1/2) * f^3 w Y^T
    #         - 0.5 * (sum f^4 w)^(1/4) * (sum f^2 w)^(-3/2) * f w Y^T
    
    n4_quarter = n4 ** 0.25
    n2_half = np.sqrt(n2)
    
    # dR/dc = n4_quarter^(-3) * n2_half^(-1) * Y^T (f^3 * w) - 0.5 * n4_quarter * n2_half^(-3) * Y^T (f * w)
    #       = (1/(n4^0.75 * n2^0.5)) * Y^T (f^3 * w) - (0.5 * n4^0.25 / n2^1.5) * Y^T (f * w)
    
    term1 = (1.0 / (n4 ** 0.75 * n2_half)) * (Y.T @ (f * f * f * weights))
    term2 = (0.5 * n4_quarter / (n2 * n2_half)) * (Y.T @ (f * weights))
    
    grad = term1 - term2
    
    return R, grad


def _trust_region_sphere(c0, Y, weights, G, maxiter=500, delta_init=1.0):
    """Trust-region optimization on the unit sphere (c^T G c = 1).
    
    Uses a Newton-like step with line search, projecting back to the sphere.
    """
    c = c0.copy()
    # Ensure c is on the sphere: c^T G c = 1
    c_norm = np.sqrt(c @ G @ c)
    if c_norm > 1e-12:
        c = c / c_norm
    
    best_c = c.copy()
    best_val, _ = _objective_and_grad(c, Y, weights)
    
    delta = delta_init
    
    for it in range(maxiter):
        if time.time() > _DEADLINE - 10.0:
            break
        
        val, grad = _objective_and_grad(c, Y, weights)
        if val < best_val:
            best_val = val
            best_c = c.copy()
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-10:
            break
        
        # Compute Hessian approximation using BFGS-like update
        # For simplicity, use gradient descent with adaptive step size
        # Project gradient to tangent space of the sphere
        # Tangent space: {v : v^T G c = 0}
        # Projected gradient: P * grad where P = I - G c (c^T G) / (c^T G c)
        # Since c^T G c = 1: P = I - G c c^T
        
        Gc = G @ c
        proj_grad = grad - Gc * (c @ grad)  # Project to tangent space
        
        # Adaptive step size with backtracking line search
        step = delta
        improved = False
        for _ in range(30):
            c_new = c + step * proj_grad
            # Project back to sphere
            c_new_norm = np.sqrt(c_new @ G @ c_new)
            if c_new_norm > 1e-12:
                c_new = c_new / c_new_norm
            
            new_val, _ = _objective_and_grad(c_new, Y, weights)
            if new_val < val - 1e-12 * step * np.linalg.norm(proj_grad):
                c = c_new
                improved = True
                break
            step *= 0.5
        
        if not improved:
            # Try smaller delta
            delta *= 0.5
            if delta < 1e-10:
                break
        else:
            delta = min(delta * 1.5, 2.0)
    
    return best_c, best_val


def _make_inits(rng, k, G):
    """Generate diverse initializations on the unit sphere (c^T G c = 1)."""
    inits = []
    dim = 2 * k + 1
    
    def normalize(c):
        norm = np.sqrt(c @ G @ c)
        if norm > 1e-12:
            return c / norm
        return c
    
    # 1. Single modes
    for m in range(-k, k + 1):
        v = np.zeros(dim)
        v[m + k] = 1.0
        inits.append(normalize(v))
        
    # 2. Pairs of conjugate modes
    for m in range(1, k + 1):
        v = np.zeros(dim)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(normalize(v))
        
    # 3. m=0 mixed with high m
    for m in [k, k-1, k-2]:
        for weight in [0.1, 0.3, 0.5, 0.7, 0.9]:
            v = np.zeros(dim)
            v[0 + k] = weight
            v[m + k] = np.sqrt(1 - weight**2)
            v[-m + k] = np.sqrt(1 - weight**2)
            inits.append(normalize(v))
            
    # 4. Random
    for _ in range(50):
        v = rng.standard_normal(dim)
        inits.append(normalize(v))
        
    # 5. Sparse random
    for _ in range(40):
        v = np.zeros(dim)
        idx = rng.choice(dim, size=rng.integers(2, 8), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        inits.append(normalize(v))
        
    # 6. Dominant high m with small noise
    for m in [k, k-1, -(k-1), -k]:
        v = np.zeros(dim)
        v[m + k] = 1.0
        v += 0.1 * rng.standard_normal(dim)
        inits.append(normalize(v))

    return inits


def rudin_problem10() -> np.ndarray:
    Y, weights, G = _build_harmonics()
    k = _K
    rng = np.random.default_rng(42)
    inits = _make_inits(rng, k, G)

    best_c = inits[0].copy()
    best_val, _ = _objective_and_grad(best_c, Y, weights)
    
    # Phase 1: Optimize from diverse starts with trust-region
    scored = []
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 150.0:
            break
        c, val = _trust_region_sphere(x0, Y, weights, G, maxiter=300, delta_init=0.5)
        scored.append((val, c))
        if val < best_val:
            best_val = val
            best_c = c
            
    # Sort and keep top candidates for refinement
    scored.sort(key=lambda t: t[0])
    top_candidates = [c for _, c in scored[:15]]
    if not top_candidates:
        top_candidates = [best_c]

    # Phase 2: Refine top candidates with more iterations
    for x0 in top_candidates:
        if time.time() > _DEADLINE - 60.0:
            break
        c, val = _trust_region_sphere(x0, Y, weights, G, maxiter=1000, delta_init=0.1)
        if val < best_val:
            best_val = val
            best_c = c
            
    # Phase 3: Random restarts around best
    while time.time() < _DEADLINE - 20.0:
        noise_scale = 0.2
        x0 = best_c + noise_scale * rng.standard_normal(2 * k + 1)
        # Normalize
        norm = np.sqrt(x0 @ G @ x0)
        if norm > 1e-12:
            x0 = x0 / norm
        c, val = _trust_region_sphere(x0, Y, weights, G, maxiter=200, delta_init=0.3)
        if val < best_val:
            best_val = val
            best_c = c

    out = best_c.astype(np.float64)
    
    # Final validity check
    if np.any(np.isnan(out)) or np.any(np.isinf(out)):
        out = np.ones(21) / np.sqrt(21)
    if np.max(np.abs(out)) < 1e-3 or np.max(np.abs(out)) > 1e5:
        out = np.ones(21) / np.sqrt(21)
        
    return out


# EVOLVE-BLOCK-END
