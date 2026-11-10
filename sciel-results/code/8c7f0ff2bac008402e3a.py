# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, factorial2
from scipy.optimize import minimize as scipy_minimize
import time

def _precompute_basis():
    """Precompute real orthonormal degree-10 spherical harmonics on 70x130 grid."""
    k = 10
    n_theta = 70
    n_phi = 130
    n_pts = n_theta * n_phi
    dim = 2 * k + 1

    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, k, x)

    N = np.zeros(11)
    for m in range(11):
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * factorial2(k - m) / factorial2(k + m))

    B = np.zeros((n_pts, dim))
    weights = np.repeat(w, n_phi) * (2.0 * np.pi / n_phi)
    sqrt2 = np.sqrt(2.0)

    B[:, 10] = np.repeat(N[0] * P[0, :], n_phi)
    for m in range(1, 11):
        val_theta = sqrt2 * N[m] * P[m, :]
        B[:, m + 10] = (val_theta[:, None] * np.cos(m * phi)[None, :]).ravel()
        B[:, -m + 10] = (val_theta[:, None] * np.sin(m * phi)[None, :]).ravel()

    return B, weights

B_GLOBAL, W_GLOBAL = _precompute_basis()
N_PTS = B_GLOBAL.shape[0]

def _compute_ratio(c):
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    if norm2_sq < 1e-30:
        return np.inf
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(W_GLOBAL * f2 * f2) ** 0.25
    return norm4 / norm2

def _objective_and_grad(c):
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    if norm2_sq < 1e-30:
        return np.inf, np.zeros_like(c)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(W_GLOBAL * f2 * f2) ** 0.25
    if norm4 < 1e-30:
        return np.inf, np.zeros_like(c)
    R = norm4 / norm2
    G4 = B_GLOBAL.T @ (W_GLOBAL * f * f2)
    G2 = B_GLOBAL.T @ (W_GLOBAL * f)
    grad_R = G4 / (norm2 * norm4**3) - (norm4 * G2) / (norm2**3)
    return R, grad_R

def _hessian(c):
    """Compute Hessian of R via automatic differentiation of the gradient."""
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(W_GLOBAL * f2 * f2) ** 0.25

    a = 1.0 / (norm2 * norm4**3)
    b = norm4 / norm2**3

    # Hessian of G4 = B^T diag(w f^2) B  =>  B^T diag(w (3 f^2)) B + B^T diag(w f^2) B... 
    # Actually d/dx [w f^2] = 2 w f * B, so H_G4 = B^T diag(2 w f) B + B^T diag(w f^2) B... 
    # Let me be more careful. G4_i = sum_j w_j f_j^3 * B_ji.
    # dG4_i/dc_k = sum_j w_j * 3 f_j^2 * B_jk * B_ji
    # So H_G4[i,k] = sum_j 3 w_j f_j^2 B_ji B_jk = 3 * B^T diag(w f^2) B
    H_G4 = 3.0 * (B_GLOBAL.T @ (W_GLOBAL * f2)[:, None] * B_GLOBAL)
    # Wait, that's not right either. Let me think again.
    # G4 = B^T (w * f^3) where f = B c
    # dG4/dc = B^T diag(3 w f^2) B
    H_G4 = B_GLOBAL.T @ (3.0 * W_GLOBAL * f2)[:, None] * B_GLOBAL

    # G2 = B^T (w f), dG2/dc = B^T diag(w) B
    H_G2 = B_GLOBAL.T @ W_GLOBAL[:, None] * B_GLOBAL

    # R = norm4/norm2
    # dR/dc = a * G4 - b * G2  (where a = 1/(norm2*norm4^3), b = norm4/norm2^3)
    # Hessian: d^2R/dc^2 = a * H_G4 - b * H_G2 + (da/dc * G4 + a * dG4/dc... )
    # This is getting complex. Let me use a simpler approach: numerical Hessian via finite differences of grad.
    
    # Actually, let's just use a trust-region / Newton method with numerical Hessian.
    # For 21 variables, a 21x21 numerical Hessian is cheap.
    
    eps = 1e-7
    H = np.zeros((21, 21))
    g0 = _objective_and_grad(c)[1]
    for i in range(21):
        c_plus = c.copy()
        c_plus[i] += eps
        g_plus = _objective_and_grad(c_plus)[1]
        H[i, :] = (g_plus - g0) / eps
    return H

def _newton_polish(c, deadline, max_iter=200):
    """Newton's method with line search for final polish."""
    c = c.copy()
    current_val = _compute_ratio(c)
    
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
        val, grad = _objective_and_grad(c)
        if not np.isfinite(val):
            break
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-14:
            break
        
        H = _hessian(c)
        # Solve H * delta = -grad
        try:
            delta = np.linalg.solve(H + 1e-12 * np.eye(21), -grad)
        except np.linalg.LinAlgError:
            delta = -grad / gnorm
        
        # Line search
        improved = False
        step = 1.0
        for _ in range(30):
            c_try = c + step * delta
            val_try = _compute_ratio(c_try)
            if val_try < current_val - 1e-16:
                c = c_try
                current_val = val_try
                improved = True
                break
            step *= 0.5
        if not improved:
            break
    return current_val, c

def _normalize(c):
    f = B_GLOBAL @ c
    n2 = np.sum(W_GLOBAL * f * f)
    if n2 < 1e-30:
        return c
    return c / np.sqrt(n2)

def _lgbfsb(c0, deadline, norm_func=None):
    def obj_and_grad(c):
        if not np.all(np.isfinite(c)):
            return 1e10, np.zeros_like(c)
        cn = norm_func(c)
        val, grad = _objective_and_grad(cn)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(c)
        return val, grad
    try:
        res = scipy_minimize(obj_and_grad, c0, jac=True, method='L-BFGS-B',
                             options={'maxiter': 8000, 'ftol': 1e-25, 'gtol': 1e-22, 'maxfun': 30000})
        if res.fun < 1e10:
            return res.fun, norm_func(res.x)
    except Exception:
        pass
    return np.inf, c0

def _wigner_d_j10(m, alpha, beta):
    """Compute d^10_{10,m}(beta) * e^{i m alpha} properly using scipy factorial2."""
    j = 10
    mp = 10  # We're rotating Y_{10,10} to get Y_{10,m}
    # d^j_{mp,m}(beta) = sum_k (-1)^k sqrt((mp+mp)! (m+m)! / ((mp+mp-2k)! (m+m-2k)! ... )) 
    # Standard formula: d^j_{m',m}(beta) = sum_k (-1)^{k} * sqrt((m'+m-2k)!... )
    # Let me use the explicit sum formula for d^j_{m',m}(beta):
    # d^j_{m',m}(beta) = sum_{k=kmin}^{kmax} (-1)^k * (m'+m-2k)! / ((m'-m+k)! (m'+m-k)! k! (j-m'-k)! (j+m-k)!)
    # * sqrt((j+m')!(j-m')!(j+m)!(j-m)!) * cos^{2j+m-m'-2k}(beta/2) * sin^{m'-m+2k}(beta/2)
    # where kmin = max(0, m'-m), kmax = min(j-m', j+m)
    
    cb = np.cos(beta / 2.0)
    sb = np.sin(beta / 2.0)
    
    kmin = max(0, mp - m)
    kmax = min(j - mp, j + m)
    if kmax < kmin:
        return 0.0
    
    # Precompute factorials
    log_fact = np.zeros(2 * j + 2)
    for i in range(1, 2 * j + 2):
        log_fact[i] = log_fact[i-1] + np.log(i)
    
    log_num = log_fact[j+mp] + log_fact[j-mp] + log_fact[j+m] + log_fact[j-m]
    
    total = 0.0
    for k in range(kmin, kmax + 1):
        log_term = (log_fact[j + mp - m + k] + log_fact[j - mp - k] + 
                    log_fact[j + m - k] + log_fact[k] +
                    log_fact[2*j + mp - m - 2*k])  # (m'+m-2k)! = (2j + mp - m - 2k)!... 
        # Wait, let me redo. (m'+m-2k)! where m'=mp=10, so (20-m-2k)!
        # Actually the formula is:
        # d^j_{m',m} = sum_k (-1)^k * sqrt( (j+m')!(j-m')!(j+m)!(j-m)! )
        #   * cos^{2j+m-m'-2k}(beta/2) * sin^{m'-m+2k}(beta/2)
        #   / ( (m'-m+k)! (m'+m-k)! k! (j-m'-k)! (j+m-k)! )
        
        # Exponents:
        exp_cos = 2*j + m - mp - 2*k  # 20 + m - 10 - 2k = 10 + m - 2k
        exp_sin = mp - m + 2*k         # 10 - m + 2k
        
        if exp_cos < 0 or exp_sin < 0:
            continue
        
        log_den = (log_fact[mp - m + k] + log_fact[mp + m - k] + 
                   log_fact[k] + log_fact[j - mp - k] + log_fact[j + m - k])
        
        log_val = 0.5 * log_num - log_den + exp_cos * np.log(max(cb, 1e-300)) + exp_sin * np.log(max(sb, 1e-300))
        
        if log_val < -700:
            continue
        
        sign = (-1.0) ** k
        total += sign * np.exp(log_val)
    
    return total

def _generate_inits():
    dim = 21
    inits = []
    
    # Zonal
    c = np.zeros(dim); c[10] = 1.0; inits.append(c)
    
    # m=±10
    c = np.zeros(dim); c[20] = 1.0; c[0] = 1.0; inits.append(c)
    
    # All ones
    inits.append(np.ones(dim))
    
    # Alternating
    c = np.zeros(dim)
    for i in range(dim):
        c[i] = 1.0 if (i-10) % 2 == 0 else -1.0
    inits.append(c)
    
    # Gaussian in m
    for sig in [1.5, 2.5, 3.5, 5.0]:
        c = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c[i] = np.exp(-m**2 / (2*sig**2))
        inits.append(c)
    
    # Cosine patterns
    for freq in [1, 2, 3]:
        c = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c[i] = np.cos(freq * np.pi * m / 20.0)
        inits.append(c)
    
    # Random with symmetry
    rng = np.random.RandomState(7)
    for _ in range(30):
        c = rng.randn(dim)
        for m in range(1, 11):
            c[10-m] = c[10+m]
        inits.append(c)
    
    # Proper Wigner-d rotated Y_{10,10}
    rng2 = np.random.RandomState(42)
    for _ in range(60):
        beta = rng2.uniform(0, np.pi)
        alpha = rng2.uniform(0, 2*np.pi)
        c = np.zeros(dim)
        for m in range(-10, 11):
            d = _wigner_d_j10(m, alpha, beta)
            c[10+m] = d * np.cos(m * alpha)
            if m != 0:
                c[10-m] = d * np.sin(m * alpha)
        nrm = _normalize(c)
        if np.linalg.norm(nrm) > 1e-10:
            inits.append(nrm)
    
    # Pure cos/sin for each m
    for m in range(1, 11):
        c = np.zeros(dim); c[10+m] = 1.0; inits.append(c)
        c = np.zeros(dim); c[10-m] = 1.0; inits.append(c)
    
    # Pairs
    for m1 in range(1, 11, 2):
        for m2 in range(m1+1, 11, 2):
            c = np.zeros(dim); c[10+m1] = 1.0; c[10+m2] = 1.0; inits.append(c)
            c = np.zeros(dim); c[10-m1] = 1.0; c[10-m2] = 1.0; inits.append(c)
    
    return inits

def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 600.0  # Safety margin from 630s
    dim = 21
    
    best_val = np.inf
    best_coeffs = None
    
    inits = _generate_inits()
    
    # Evaluate all inits
    scored = []
    for c in inits:
        cn = _normalize(c)
        v = _compute_ratio(cn)
        scored.append((v, cn))
        if v < best_val:
            best_val = v
            best_coeffs = cn.copy()
    
    scored.sort(key=lambda x: x[0])
    
    # L-BFGS-B from top initial points
    n_starts = min(60, len(scored))
    for i in range(n_starts):
        if time.time() > deadline - 180:
            break
        v, c_opt = _lgbfsb(scored[i][1], deadline, norm_func=_normalize)
        if v < best_val:
            best_val = v
            best_coeffs = c_opt.copy()
    
    # Newton polish
    if time.time() < deadline - 60:
        v, c_opt = _newton_polish(best_coeffs, deadline)
        if v < best_val:
            best_val = v
            best_coeffs = c_opt.copy()
    
    # Final L-BFGS-B
    if time.time() < deadline - 30:
        v, c_opt = _lgbfsb(best_coeffs, deadline, norm_func=_normalize)
        if v < best_val:
            best_val = v
            best_coeffs = c_opt.copy()
    
    # Final Newton
    if time.time() < deadline - 10:
        v, c_opt = _newton_polish(best_coeffs, deadline)
        if v < best_val:
            best_val = v
            best_coeffs = c_opt.copy()
    
    coeffs = best_coeffs.astype(np.float64)
    if not np.all(np.isfinite(coeffs)):
        coeffs = np.ones(dim)
    mc = np.max(np.abs(coeffs))
    if mc < 1e-3:
        coeffs = coeffs / max(mc, 1e-15)
    elif mc > 1e5:
        coeffs = coeffs / mc
    return coeffs

# EVOLVE-BLOCK-END
