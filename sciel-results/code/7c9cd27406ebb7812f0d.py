# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import eval_legendre

_DEADLINE_SECONDS = 620.0

def _build_harmonic_matrix():
    """Build the real orthonormal degree-10 spherical harmonic evaluation matrix."""
    k = 10
    Ntheta = 70
    Nphi = 130

    x, w = np.polynomial.legendre.leggauss(Ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=False)
    dphi = 2.0 * np.pi / Nphi

    H = np.zeros((Ntheta, Nphi, 2 * k + 1), dtype=np.float64)

    P = np.zeros((Ntheta, k + 1), dtype=np.float64)
    for m in range(k + 1):
        P[:, m] = eval_legendre(k, m, x)

    def N_const(m):
        m = abs(int(m))
        lo = k - m + 1
        hi = k + m
        den = 1.0
        for j in range(lo, hi + 1):
            den *= j
        return np.sqrt((2 * k + 1) / (4.0 * np.pi) * (1.0 / den))

    for m in range(-k, k + 1):
        j = m + k
        if m == 0:
            H[:, :, j] = N_const(0) * P[:, 0][:, None] * np.ones((1, Nphi))
        elif m > 0:
            H[:, :, j] = np.sqrt(2.0) * N_const(m) * P[:, m][:, None] * np.cos(m * phi)[None, :]
        else:
            am = -m
            H[:, :, j] = np.sqrt(2.0) * N_const(am) * P[:, am][:, None] * np.sin(am * phi)[None, :]

    weights = np.outer(w, np.full(Nphi, dphi))
    return H, weights

def _ratio_from_coeffs(c, H_flat, w_flat):
    """Compute R = ||f||_4 / ||f||_2."""
    f = H_flat @ c
    f2 = f * f
    f4 = f2 * f2
    A = np.dot(f4, w_flat)
    B = np.dot(f2, w_flat)
    if A <= 0 or B <= 0:
        return np.inf
    return A ** 0.25 / np.sqrt(B)

def _trust_region_method(c, H_flat, w_flat, max_iters=100, deadline=None):
    """
    Riemannian Trust-Region method on the unit sphere.
    Minimizes J(c) = ||f||_4 / ||f||_2.
    Uses exact Hessian projected to tangent space.
    """
    n = len(c)
    c = c.copy()
    cn = np.linalg.norm(c)
    if cn < 1e-15:
        return c
    c = c / cn
    
    best_c = c.copy()
    best_R = _ratio_from_coeffs(c, H_flat, w_flat)
    
    # Precompute W for efficiency
    # H_flat: (N, n), w_flat: (N,)
    # Ht_W = H_flat.T @ (w_flat[:, None] * H_flat) -> (n, n)
    # Ht_W_f = H_flat.T @ (w_flat * (H_flat @ c)) -> (n,)
    # Ht_W_f2 = H_flat.T @ (w_flat * (H_flat @ c)**2) -> (n, n) via Ht_W * f? No.
    # Ht_W_f2 = H_flat.T @ (w_flat * f^2)[:, None] * H_flat? No.
    # Let's just compute on the fly or cache parts.
    
    I = np.eye(n)
    
    for it in range(max_iters):
        if deadline is not None and (time.time() - deadline[0]) > deadline[1]:
            break
            
        f = H_flat @ c
        f2 = f * f
        f4 = f2 * f2
        
        A = np.dot(f4, w_flat)
        B = np.dot(f2, w_flat)
        total_w = np.sum(w_flat)
        
        # Objective: J = A^0.25 / B^0.5
        # Log Objective: L = 0.25 ln A - 0.5 ln B
        # Grad L = 0.25/A * dA/dc - 0.5/B * dB/dc
        # dA/dc = 4 H^T (f^3 w)
        # dB/dc = 2 H^T (f w)
        
        f3w = f * f2 * w_flat
        fw = f * w_flat
        
        grad_A = 4.0 * (H_flat.T @ f3w)
        grad_B = 2.0 * (H_flat.T @ fw)
        
        # Gradient of L
        grad_L = (0.25 / A) * grad_A - (0.5 / B) * grad_B
        
        # Project to tangent space
        c_dot_g = np.dot(c, grad_L)
        grad_tan = grad_L - c_dot_g * c
        
        g_norm = np.linalg.norm(grad_tan)
        if g_norm < 1e-14:
            break
            
        # Hessian of L
        # d2L = 0.25 * d2A - 0.5 * d2B
        # d2A = 12 H^T (f^2 w) H + 4 H^T (f w) H^T (f w)? No.
        # A = sum w_i f_i^4
        # dA = sum 4 w_i f_i^3 df
        # d2A = sum 12 w_i f_i^2 df^2 + 4 w_i f_i d^2f? No, f is linear in c.
        # f = H c. df = H dc.
        # dA = 4 (H^T (f^3 w)) dc
        # d2A = 12 (H^T (f^2 w) H) dc dc
        # So Hess_A = 12 H^T (f^2 w) H
        # Similarly Hess_B = 2 H^T (w) H
        
        f2w = f2 * w_flat
        # Hess_A = 12 * (H_flat.T @ (f2w[:, None] * H_flat))
        # This is 12 * (H_flat.T * f2w) @ H_flat
        # Let's compute Ht_f2w = H_flat.T @ (f2w[:, None] * H_flat)
        # Actually: H_flat.T @ (f2w[:, None] * H_flat) is (n, N) @ (N, n) -> (n, n)
        # Element (i,j) = sum_k H_ki f_k^2 w_k H_kj
        
        Ht_f2w_H = (H_flat.T * f2w) @ H_flat
        Ht_w_H = (H_flat.T * w_flat) @ H_flat
        
        Hess_A = 12.0 * Ht_f2w_H
        Hess_B = 2.0 * Ht_w_H
        
        # Hess_L = 0.25 * d2(ln A) - 0.5 * d2(ln B)
        # d2(ln A) = d2A/A - (dA)^2/A^2
        # dA = grad_A
        # d2(ln B) = d2B/B - (dB)^2/B^2
        
        outer_A = np.outer(grad_A, grad_A)
        outer_B = np.outer(grad_B, grad_B)
        
        Hess_L = (0.25 / A) * (Hess_A - outer_A / A) - (0.5 / B) * (Hess_B - outer_B / B)
        
        # Project Hessian to tangent space
        P = I - np.outer(c, c)
        Hess_tan = P @ Hess_L @ P
        
        # Solve (Hess_tan + mu I) s = -grad_tan
        # Adaptive Levenberg-Marquardt damping
        s = None
        for mu in [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0]:
            try:
                s = np.linalg.solve(Hess_tan + mu * I, -grad_tan)
                if np.linalg.norm(s) <= 1.0:
                    break
            except np.linalg.LinAlgError:
                continue
        
        if s is None:
            # Fallback to gradient descent
            s = -grad_tan / (g_norm + 1e-30)
        
        # Project step to tangent space
        s = s - np.dot(c, s) * c
        
        # Try step
        # Riemannian exponential map: c_new = cos(||s||) c + sin(||s||) (s/||s||)
        s_norm = np.linalg.norm(s)
        if s_norm < 1e-15:
            break
            
        c_trial = c + s
        cn_t = np.linalg.norm(c_trial)
        if cn_t < 1e-15:
            break
        c_trial = c_trial / cn_t
        
        R_trial = _ratio_from_coeffs(c_trial, H_flat, w_flat)
        
        if R_trial < best_R:
            c = c_trial
            best_c = c.copy()
            best_R = R_trial
        else:
            # Try half step
            s_half = 0.5 * s
            c_trial = c + s_half
            cn_t = np.linalg.norm(c_trial)
            if cn_t > 1e-15:
                c_trial = c_trial / cn_t
                R_trial = _ratio_from_coeffs(c_trial, H_flat, w_flat)
                if R_trial < best_R:
                    c = c_trial
                    best_c = c.copy()
                    best_R = R_trial
                else:
                    # Try quarter step
                    s_quarter = 0.25 * s
                    c_trial = c + s_quarter
                    cn_t = np.linalg.norm(c_trial)
                    if cn_t > 1e-15:
                        c_trial = c_trial / cn_t
                        R_trial = _ratio_from_coeffs(c_trial, H_flat, w_flat)
                        if R_trial < best_R:
                            c = c_trial
                            best_c = c.copy()
                            best_R = R_trial
                        else:
                            break
                    else:
                        break
            else:
                break
                
    return best_c

def _annealing_search(c_init, H_flat, w_flat, deadline):
    """
    Perform annealing: minimize ||f||_p for p = 6, 5, 4.
    This helps escape local minima.
    """
    c = c_init.copy()
    cn = np.linalg.norm(c)
    if cn > 1e-15:
        c = c / cn
        
    for p in [6.0, 5.0, 4.0]:
        # We don't have a direct trust region for p-norm, so we use gradient descent
        # for the annealing steps to guide the solution towards flatter distributions.
        # Objective: E[|f|^p]
        # Grad: p H^T (|f|^(p-2) f w)
        
        for it in range(50):
            if (time.time() - deadline[0]) > deadline[1]:
                break
                
            f = H_flat @ c
            f_abs_p_minus_2 = np.abs(f) ** (p - 2.0)
            grad = p * (H_flat.T @ (f_abs_p_minus_2 * f * w_flat))
            
            c_dot_g = np.dot(c, grad)
            grad_tan = grad - c_dot_g * c
            g_norm = np.linalg.norm(grad_tan)
            
            if g_norm < 1e-10:
                break
            
            # Simple gradient step with adaptive size
            step = 0.1 / (g_norm + 1e-10)
            c_trial = c + step * grad_tan
            cn_t = np.linalg.norm(c_trial)
            if cn_t < 1e-15:
                break
            c_trial = c_trial / cn_t
            
            # Check improvement in p-norm
            f_trial = H_flat @ c_trial
            val_trial = np.dot(np.abs(f_trial) ** p, w_flat)
            val_curr = np.dot(np.abs(f) ** p, w_flat)
            
            if val_trial < val_curr:
                c = c_trial
            else:
                break
                
    # Final polish with Trust Region on L4/L2
    c_final = _trust_region_method(c, H_flat, w_flat, max_iters=100, deadline=deadline)
    return c_final

def _make_initializations(k, rng, n):
    """Generate diverse starting points on the unit sphere."""
    inits = []
    
    # Single modes
    for m in range(-k, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        inits.append(v.copy())
    
    # Pairs (cos + sin for same |m|)
    for m in range(1, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0 / np.sqrt(2)
        v[-m + k] = 1.0 / np.sqrt(2)
        inits.append(v.copy())
    
    # All ones normalized
    v = np.ones(n)
    v /= np.linalg.norm(v)
    inits.append(v.copy())
    
    # Even m only
    v = np.zeros(n)
    for m in range(-k, k + 1, 2):
        v[m + k] = 1.0
    v /= np.linalg.norm(v)
    inits.append(v.copy())
    
    # Odd m only
    v = np.zeros(n)
    for m in range(-k + 1, k + 1, 2):
        v[m + k] = 1.0
    v /= np.linalg.norm(v)
    inits.append(v.copy())
    
    # m=0 + high m combinations
    for m in [2, 4, 6, 8, 10]:
        for n_m in [2, 4, 6, 8, 10]:
            if m != n_m:
                v = np.zeros(n)
                v[0 + k] = 1.0
                v[m + k] = 1.0
                v[-m + k] = 1.0
                v[n_m + k] = 0.5
                v[-n_m + k] = 0.5
                v /= np.linalg.norm(v)
                inits.append(v.copy())
    
    # Structured: emphasize certain m values
    for weight_m in [2, 4, 6, 8, 10]:
        v = np.zeros(n)
        for m in range(-k, k + 1):
            v[m + k] = 1.0 if abs(m) == weight_m else (0.3 if abs(m) == 0 else 0.1)
        v /= np.linalg.norm(v)
        inits.append(v.copy())
        
    # Random
    for _ in range(20):
        v = rng.normal(size=n)
        v /= np.linalg.norm(v)
        inits.append(v.copy())
        
    # Sparse random
    for _ in range(10):
        v = np.zeros(n)
        idx = rng.choice(n, size=rng.integers(3, 8), replace=False)
        v[idx] = rng.normal(size=len(idx))
        v /= np.linalg.norm(v)
        inits.append(v.copy())
        
    return inits

def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m (m = -10..10) of a degree-10 real spherical
    harmonic f = sum_m c_m Y_m that MINIMIZES the norm ratio
        R = ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}
    
    Uses Riemannian Trust-Region (Damped Newton) with Annealing.
    """
    start_time = time.time()
    k = 10
    n = 2 * k + 1

    H, weights = _build_harmonic_matrix()
    H_flat = H.reshape(-1, n)
    w_flat = weights.reshape(-1)

    rng = np.random.default_rng(12345)

    def remaining():
        return _DEADLINE_SECONDS - (time.time() - start_time)

    inits = _make_initializations(k, rng, n)

    best_c = np.ones(n, dtype=np.float64) / np.sqrt(n)
    best_R = _ratio_from_coeffs(best_c, H_flat, w_flat)

    # Phase 1: Annealing + Trust Region from diverse starts
    for init_idx, c0 in enumerate(inits):
        if remaining() < 30.0:
            break

        c_result = _annealing_search(c0, H_flat, w_flat, deadline=(start_time, _DEADLINE_SECONDS - 20.0))
        
        R = _ratio_from_coeffs(c_result, H_flat, w_flat)
        if R < best_R:
            best_R = R
            best_c = c_result.copy()

    # Phase 2: Additional random restarts
    n_extra = 10
    for _ in range(n_extra):
        if remaining() < 30.0:
            break
        
        v = rng.normal(size=n)
        v /= np.linalg.norm(v)
        
        c_result = _annealing_search(v, H_flat, w_flat, deadline=(start_time, _DEADLINE_SECONDS - 20.0))
        
        R = _ratio_from_coeffs(c_result, H_flat, w_flat)
        if R < best_R:
            best_R = R
            best_c = c_result.copy()

    # Final deep polish
    if remaining() > 30.0:
        c_final = _trust_region_method(best_c, H_flat, w_flat, max_iters=100,
                                        deadline=(start_time, _DEADLINE_SECONDS - 10.0))
        R_final = _ratio_from_coeffs(c_final, H_flat, w_flat)
        if R_final < best_R:
            best_R = R_final
            best_c = c_final.copy()

    # Validate
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(n, dtype=np.float64)
    mx = np.max(np.abs(best_c))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)

    return best_c

# EVOLVE-BLOCK-END
