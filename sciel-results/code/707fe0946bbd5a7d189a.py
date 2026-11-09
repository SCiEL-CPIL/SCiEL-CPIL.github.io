# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import lpmv
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    """
    Rudin problem for polynomials on the 2-sphere. Find the real coefficients
    (c_m) for m = -k..k of a real degree-k spherical harmonic

        f = sum_{m=-k}^{k} c_m Y_k^m

    on the round unit sphere S^2 that MINIMIZE the norm ratio

        ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}

    (norms taken with the standard surface measure dOmega of total mass 4*pi).
    Here the degree is fixed to k = 10, so the returned vector has length
    2*k + 1 = 21.

    Returns:
        coeffs: np.ndarray of shape (21,) with the real coefficients
            c_m, ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    k = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Leave 20s margin before 600s limit

    # Set up quadrature points
    # 70-point Gauss-Legendre in cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    
    # 130 equally spaced phi points
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Precompute sin/cos of m*phi for m = 0..10
    cos_m_phi = np.zeros((11, n_phi))
    sin_m_phi = np.zeros((11, n_phi))
    for m in range(11):
        if m == 0:
            cos_m_phi[m] = 1.0
            sin_m_phi[m] = 0.0
        else:
            cos_m_phi[m] = np.cos(m * phi)
            sin_m_phi[m] = np.sin(m * phi)
    
    # Precompute normalization constants N_m
    N = np.zeros(11)
    for m in range(11):
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) * math.factorial(10 - m) / math.factorial(10 + m))
    
    # Precompute associated Legendre polynomials P_10^m(x) for x in x_gl
    P_legendre = np.zeros((11, 70))
    for m in range(11):
        P_legendre[m] = lpmv(m, 10, x_gl)
    
    # Build the full matrix Y_flat of shape (70*130, 21)
    # Indexing: row = i*n_phi + j, col = m+10
    N_q = 70 * n_phi
    Y_flat = np.zeros((N_q, 21), dtype=np.float64)
    
    # Weights for each quadrature point
    w_flat = np.zeros(N_q, dtype=np.float64)
    
    for i in range(70):
        row_start = i * n_phi
        row_end = (i + 1) * n_phi
        w_flat[row_start:row_end] = w_gl[i] * dphi
        
        # m = 0
        Y_flat[row_start:row_end, 10] = N[0] * P_legendre[0, i]
        
        # m > 0 and m < 0
        for m in range(1, 11):
            coeff_m_i = np.sqrt(2.0) * N[m] * P_legendre[m, i]
            # c_m (index 10+m) corresponds to cos(m*phi)
            Y_flat[row_start:row_end, 10 + m] = coeff_m_i * cos_m_phi[m]
            # c_-m (index 10-m) corresponds to sin(m*phi)
            Y_flat[row_start:row_end, 10 - m] = coeff_m_i * sin_m_phi[m]

    # Structural Change: Reduce dimensionality
    # The objective J(c) is invariant under rotation in phi for each m.
    # Thus, J depends only on c_0 and A_m = sqrt(c_m^2 + c_-m^2) for m=1..10.
    # We optimize over u = [c_0, A_1, ..., A_10] (11 variables).
    
    def u_to_c(u):
        """
        Convert amplitude vector u to coefficient vector c.
        u[0] = c_0
        u[m] = A_m for m=1..10
        We set c_m = A_m / sqrt(2) and c_-m = A_m / sqrt(2) to satisfy c_m^2 + c_-m^2 = A_m^2.
        """
        c = np.zeros(21)
        c[10] = u[0]
        for m in range(1, 11):
            A = u[m]
            c[10 + m] = A / np.sqrt(2.0)
            c[10 - m] = A / np.sqrt(2.0)
        return c

    def objective_u(u):
        """
        Compute J = S4 / S2^2 for coefficients derived from u.
        """
        c = u_to_c(u)
        f = Y_flat @ c
        f2 = f * f
        S2 = np.sum(w_flat * f2)
        S4 = np.sum(w_flat * f2 * f2)
        if S2 <= 1e-300:
            return 1e100
        return S4 / (S2 * S2)

    # Gradient of J with respect to u is not directly available from c-gradient easily without chain rule,
    # but since dimension is small (11), we can use numerical gradient or derive it.
    # However, SLSQP with analytical gradient in c-space was fine. 
    # Let's stick to analytical gradient in c-space but map the optimizer?
    # No, it's easier to just use L-BFGS-B with numerical gradient on u if dimension is 11.
    # Or, we can compute gradient in c-space and project?
    # Actually, let's just use `minimize` with `method='L-BFGS-B'` and `jac=None` (numerical) on u.
    # With 11 variables, numerical gradient is cheap.
    
    # Alternatively, we can define the objective and gradient in c-space and use SLSQP with constraint ||c||=1.
    # The previous approach did this. The structural change is the *initialization* and *search space reduction*.
    # Let's use the 11-variable space for a global search, then refine in 21-variable space.
    
    best_c = None
    best_obj = np.inf

    # 1. Global Search in 11-dim amplitude space
    # We will use random restarts in the 11-dim space.
    # We normalize u such that sum(u^2) = 1? No, J is scale invariant.
    # We can fix ||u||_2 = 1 to break scaling symmetry.
    # u_norm = u / ||u||.
    
    def objective_u_normalized(u):
        norm_u = np.linalg.norm(u)
        if norm_u < 1e-15:
            return 1e100
        u_norm = u / norm_u
        return objective_u(u_norm)

    starts_u = []
    
    # Deterministic starts in u-space
    # 1. m=0 only
    u0 = np.zeros(11)
    u0[0] = 1.0
    starts_u.append(u0)
    
    # 2. m=10 only
    u10 = np.zeros(11)
    u10[10] = 1.0
    starts_u.append(u10)
    
    # 3. Mix m=0 and m=10
    umix = np.zeros(11)
    umix[0] = 1.0 / np.sqrt(2)
    umix[10] = 1.0 / np.sqrt(2)
    starts_u.append(umix)
    
    # 4. Uniform amplitudes
    uuni = np.ones(11) / np.sqrt(11)
    starts_u.append(uuni)
    
    # 5. Random starts
    for seed in range(100):
        np.random.seed(seed)
        u_rand = np.random.randn(11)
        u_rand = np.abs(u_rand) # Amplitudes are non-negative
        u_rand = u_rand / np.linalg.norm(u_rand)
        starts_u.append(u_rand)
        
    # Optimize in 11-dim space
    for i, u_start in enumerate(starts_u):
        if time.time() > deadline - 30.0:
            break
        try:
            result = minimize(
                objective_u_normalized,
                u_start,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            u_best = result.x
            u_best = np.abs(u_best) # Ensure amplitudes are non-negative
            u_best = u_best / np.linalg.norm(u_best)
            
            c_candidate = u_to_c(u_best)
            obj_val = objective_u(c_candidate) # Use original objective for consistency
            
            if obj_val < best_obj:
                best_obj = obj_val
                best_c = c_candidate.copy()
        except Exception:
            pass

    if best_c is None:
        best_c = np.ones(21) / np.sqrt(21)

    # 2. Refine in 21-dim space with SLSQP (constrained to sphere)
    # This allows the phase to adjust if my assumption about phase invariance was slightly off 
    # (e.g. due to quadrature errors) or to polish the result.
    
    def objective_J(c_vec):
        f = Y_flat @ c_vec
        f2 = f * f
        S2 = np.sum(w_flat * f2)
        S4 = np.sum(w_flat * f2 * f2)
        if S2 <= 1e-300:
            return 1e100
        return S4 / (S2 * S2)

    def jac_J(c_vec):
        f = Y_flat @ c_vec
        f2 = f * f
        S2 = np.sum(w_flat * f2)
        S4 = np.sum(w_flat * f2 * f2)
        
        if S2 <= 1e-300:
            return np.zeros(21)
            
        wf = w_flat * f
        wf3 = w_flat * f * f2
        
        dS2 = 2.0 * (Y_flat.T @ wf)
        dS4 = 4.0 * (Y_flat.T @ wf3)
        
        grad = (S2 * dS4 - 2.0 * S4 * dS2) / (S2 * S2 * S2)
        return grad

    def norm_constraint(c_vec):
        return np.dot(c_vec, c_vec) - 1.0

    def norm_jac(c_vec):
        return 2.0 * c_vec

    cons = [{'type': 'eq', 'fun': norm_constraint, 'jac': norm_jac}]

    # Polish from the best c found in 11-dim space
    if time.time() < deadline - 5.0:
        try:
            result_polish = minimize(
                objective_J,
                best_c,
                method='SLSQP',
                jac=jac_J,
                constraints=cons,
                options={
                    'maxiter': 500,
                    'ftol': 1e-14
                }
            )
            if result_polish.fun < best_obj:
                best_obj = result_polish.fun
                best_c = result_polish.x.copy()
        except Exception:
            pass

    c = best_c
    
    # Ensure valid range
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3:
        c = c / max_abs * 1e-2
    elif max_abs > 1e5:
        c = c / max_abs * 1e4
        
    return c

# EVOLVE-BLOCK-END
