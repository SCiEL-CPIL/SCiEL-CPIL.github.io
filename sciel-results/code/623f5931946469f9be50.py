# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, factorial
import time

def _precompute_basis():
    k = 10
    n_theta = 70
    n_phi = 130
    dim = 21

    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    n_pts = n_theta * n_phi

    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, k, x)

    N = np.zeros(11)
    for m in range(11):
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) *
                       float(factorial(k - m) / factorial(k + m)))

    B = np.zeros((n_pts, dim))
    weights = np.repeat(w, n_phi) * (2.0 * np.pi / n_phi)
    sqrt2 = np.sqrt(2.0)

    B[:, 10] = np.repeat(N[0] * P[0, :], n_phi)
    for m in range(1, 11):
        val = sqrt2 * N[m] * P[m, :]
        cos_t = np.cos(m * phi)
        sin_t = np.sin(m * phi)
        B[:, 10 + m] = (val[:, None] * cos_t[None, :]).ravel()
        B[:, 10 - m] = (val[:, None] * sin_t[None, :]).ravel()

    return B, weights

B_G, W_G = _precompute_basis()

def _precompute_S():
    N_pts, dim = B_G.shape
    B_reshaped = B_G[:, :, None] * B_G[:, None, :]
    Z = B_reshaped.reshape(N_pts, dim * dim)
    S = Z.T @ (W_G[:, None] * Z)
    return S

S_G = _precompute_S()
DIM = 21

def _compute_ratio(c):
    norm2_sq = np.dot(c, c)
    if norm2_sq < 1e-30:
        return np.inf
    u = (c[:, None] * c[None, :]).ravel()
    norm4_4 = u @ S_G @ u
    if norm4_4 < 1e-30:
        return np.inf
    norm4 = norm4_4 ** 0.25
    norm2 = np.sqrt(norm2_sq)
    return norm4 / norm2

def _obj_grad(c):
    norm2_sq = np.dot(c, c)
    if norm2_sq < 1e-30:
        return np.inf, np.zeros_like(c)
    u = (c[:, None] * c[None, :]).ravel()
    norm4_4 = u @ S_G @ u
    if norm4_4 < 1e-30:
        return np.inf, np.zeros_like(c)
    norm4 = norm4_4 ** 0.25
    norm2 = np.sqrt(norm2_sq)
    R = norm4 / norm2
    V = S_G @ u
    V_reshaped = V.reshape(DIM, DIM)
    grad_Q = 4.0 * (V_reshaped @ c)
    grad_R = grad_Q / (4.0 * norm4**3 * norm2**2) - (norm4 * c) / (norm2**3)
    return R, grad_R

def _normalize_unit(c):
    n2 = np.dot(c, c)
    if n2 < 1e-30:
        return c
    return c / np.sqrt(n2)

def _project_onto_sphere(c, r=1.0):
    n = np.linalg.norm(c)
    if n < 1e-15:
        c = np.random.randn(len(c))
        return c / np.linalg.norm(c) * r
    return c / n * r

def _riemannian_newton(c, deadline, max_iter=200, delta0=1.0):
    """Trust-region Newton method on the unit sphere using projected Hessian."""
    c = _normalize_unit(c.copy())
    cur = _compute_ratio(c)
    delta = delta0
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        val, grad = _obj_grad(c)
        if not np.isfinite(val):
            break
        
        # Project gradient to tangent space
        grad_t = grad - (grad @ c) * c
        gnorm = np.linalg.norm(grad_t)
        if gnorm < 1e-14:
            break
        
        # Compute Riemannian Hessian via finite differences of the projected gradient
        # H_ij = d/dx_j (grad_i - (grad@c)c_i) projected
        # Use the 4th-order structure: H = (1/4) * (S-related terms) - curvature terms
        # For efficiency, compute Hessian by applying the 4-index tensor
        
        # Alternative: use a 2-point finite difference of the projected gradient
        # This is O(dim) extra objective evaluations but gives accurate Hessian
        eps = 1e-7
        H = np.zeros((DIM, DIM))
        for i in range(DIM):
            ei = np.zeros(DIM)
            ei[i] = 1.0
            ei = ei - (ei @ c) * c  # project to tangent
            ei = ei / np.linalg.norm(ei)
            
            cp = _normalize_unit(c + eps * ei)
            cm = _normalize_unit(c - eps * ei)
            _, gp = _obj_grad(cp)
            _, gm = _obj_grad(cm)
            
            gp_t = gp - (gp @ cp) * cp
            gm_t = gm - (gm @ cm) * cm
            
            H[:, i] = (gp_t - gm_t) / (2 * eps)
        
        # Symmetrize
        H = 0.5 * (H + H.T)
        
        # Solve trust-region subproblem: min 0.5*d'H*d + grad_t'd s.t. ||d|| <= delta
        # Use eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(H)
        
        # If H is PSD, d = -delta * eigvec of smallest eigenvalue (if grad aligns)
        # Otherwise, use the standard trust-region solution
        # Simplified: if H is positive definite, d = -H^{-1} grad_t, clipped to delta
        # If not, use steepest descent direction
        
        # Check if H is positive definite
        if np.all(eigvals > 1e-10):
            # Newton step
            d = -np.linalg.solve(H, grad_t)
            dnorm = np.linalg.norm(d)
            if dnorm > delta:
                d = d / dnorm * delta
        else:
            # Use steepest descent with trust region
            d = -grad_t / gnorm * delta
        
        # Line search
        improved = False
        for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
            c_try = _normalize_unit(c + alpha * d)
            v_try = _compute_ratio(c_try)
            if v_try < cur - 1e-15:
                c = c_try
                cur = v_try
                improved = True
                delta = min(2.0 * delta, 5.0)
                break
            elif v_try < cur:
                # Accept but don't increase trust region
                c = c_try
                cur = v_try
                improved = True
                break
        
        if not improved:
            delta *= 0.5
            if delta < 1e-10:
                break
    
    return cur, c

def _init_pool(dim):
    inits = []
    c = np.zeros(dim); c[10] = 1.0; inits.append(c)
    inits.append(np.ones(dim))
    for i in range(dim):
        c = np.zeros(dim); c[i] = 1.0; inits.append(c)
    for b in [0.3, 0.5, 0.7]:
        c = np.zeros(dim)
        for i in range(dim):
            c[i] = b ** abs(i - 10)
        inits.append(c)
    for fr in np.arange(0.2, 3.0, 0.2):
        c = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c[i] = np.cos(fr * m)
        inits.append(c)
        c2 = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c2[i] = np.sin(fr * m)
        inits.append(c2)
    for s in range(50):
        np.random.seed(s * 12345 + 67890)
        inits.append(np.random.randn(dim))
    for s in range(30):
        np.random.seed(s * 98765 + 43210)
        c = np.random.randn(dim)
        for i in range(dim):
            c[i] *= 0.5 ** abs(i - 10)
        inits.append(c)
    for s in range(30):
        np.random.seed(s * 24680 + 13579)
        c = np.zeros(dim)
        nnz = np.random.randint(1, 8)
        idx = np.random.choice(dim, nnz, replace=False)
        c[idx] = np.random.randn(nnz)
        inits.append(c)
    for p1 in range(-10, 11, 3):
        for p2 in range(p1 + 2, 11, 3):
            c = np.zeros(dim)
            for i in range(dim):
                m = i - 10
                c[i] = np.exp(-(m - p1)**2 / 3.0) + np.exp(-(m - p2)**2 / 3.0)
            inits.append(c)
    return inits

def _grad_descent_polish(c, deadline, max_iter=5000):
    c = _project_onto_sphere(c.copy())
    cur = _compute_ratio(c)
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
        val, grad = _obj_grad(c)
        if not np.isfinite(val):
            break
        grad_t = grad - (grad @ c) * c
        gnorm = np.linalg.norm(grad_t)
        if gnorm < 1e-15:
            break
        grad_t = grad_t / gnorm
        step = 0.05
        improved = False
        for _bt in range(40):
            c_try = _project_onto_sphere(c - step * grad_t)
            vt = _compute_ratio(c_try)
            if vt < cur - 1e-16:
                c = c_try; cur = vt; improved = True
                break
            step *= 0.5
        if not improved:
            break
    return cur, c

def rudin_problem10() -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 600.0
    dim = DIM

    best_c = None
    best_val = np.inf

    inits = _init_pool(dim)
    init_sorted = []
    for c_init in inits:
        cn = _project_onto_sphere(c_init)
        v = _compute_ratio(cn)
        init_sorted.append((v, cn))
        if v < best_val:
            best_val = v
            best_c = cn.copy()
    init_sorted.sort(key=lambda x: x[0])

    # Phase 1: Riemannian Newton from top 30 initial points
    for v, cs in init_sorted[:30]:
        if time.time() > deadline - 200:
            break
        try:
            vo, co = _riemannian_newton(cs, deadline, max_iter=150, delta0=1.0)
            if vo < best_val:
                best_val = vo
                best_c = co.copy()
        except Exception:
            pass

    # Phase 2: Gradient polish from best
    if time.time() < deadline - 50:
        vp, cp = _grad_descent_polish(best_c, deadline, max_iter=5000)
        if vp < best_val:
            best_val = vp
            best_c = cp.copy()

    # Phase 3: Riemannian Newton again from polished point
    if time.time() < deadline - 30:
        try:
            vo, co = _riemannian_newton(best_c, deadline, max_iter=200, delta0=0.5)
            if vo < best_val:
                best_val = vo
                best_c = co.copy()
        except Exception:
            pass

    # Phase 4: Final gradient polish
    if time.time() < deadline - 10:
        vp, cp = _grad_descent_polish(best_c, deadline, max_iter=3000)
        if vp < best_val:
            best_val = vp
            best_c = cp.copy()

    # Phase 5: Try a few more Newton restarts from perturbed best
    if time.time() < deadline - 15:
        for s in range(5):
            if time.time() > deadline - 15:
                break
            np.random.seed(42 + s)
            pert = best_c + 0.1 * np.random.randn(dim)
            pert = _project_onto_sphere(pert)
            try:
                vo, co = _riemannian_newton(pert, deadline, max_iter=100, delta0=0.3)
                if vo < best_val:
                    best_val = vo
                    best_c = co.copy()
            except Exception:
                pass

    coeffs = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(coeffs)):
        coeffs = np.ones(dim)
    mc = np.max(np.abs(coeffs))
    if mc < 1e-3:
        coeffs = coeffs / mc * 1.0 if mc > 0 else np.ones(dim)
    elif mc > 1e5:
        coeffs = coeffs / mc * 1.0
    return coeffs

# EVOLVE-BLOCK-END
