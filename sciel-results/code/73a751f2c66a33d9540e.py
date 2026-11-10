import numpy as np
from scipy.special import lpmv, factorial, gammaln, sph_harm_y
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    
    New Strategy:
    1. Precompute the 4th-order angular moment tensor K_{ijkl} = \int Y_i Y_j Y_k Y_l d\Omega
       using Gaunt coefficients (3j symbols) for all m in [-10, 10].
    2. Express ||f||_4^4 = sum_{i,j,k,l} c_i c_j c_k c_l K_{ijkl} as a pure polynomial in c.
    3. Express ||f||_2^2 = sum_i c_i^2 (since basis is orthonormal).
    4. Minimize R(c) = (||f||_4^4 / ||f||_2^4)^{1/4} using L-BFGS-B with analytic gradients
       derived from the polynomial form, starting from multiple structured initial guesses.
    5. This avoids spatial grid non-convexity and leverages the smooth spectral polynomial landscape.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    k = 10
    n_m = 2 * k + 1  # 21 coefficients
    m_vals = np.arange(-k, k + 1)
    
    # Map m index to array index
    def m_to_idx(m):
        return m + k

    # 1. Precompute basis norms N_m
    N = np.zeros(n_m)
    for m in m_vals:
        m_abs = abs(m)
        # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
        log_N = 0.5 * (np.log(21.0 / (4.0 * np.pi)) + gammaln(10 - m_abs + 1) - gammaln(10 + m_abs + 1))
        N[m_to_idx(m)] = np.exp(log_N)

    # 2. Precompute Gaunt coefficients for 4th moment
    # We need K_{i,j,k,l} = \int Y_i Y_j Y_k Y_l d\Omega
    # Using the identity: \int Y_l^m Y_{l'}^{m'} Y_{l''}^{m''} d\Omega = \sqrt{\frac{(2l+1)(2l'+1)(2l''+1)}{4\pi}}
    #   * ( l  l'  l'' )
    #   * ( 0  0   0 )
    #   * ( m  m'  m'' )
    # For 4 functions, we can pair them: \int (Y_i Y_j) (Y_k Y_l) d\Omega
    # But it's easier to use the 3j symbol product formula for 4 harmonics:
    # \int Y_{l1}^{m1} Y_{l2}^{m2} Y_{l3}^{m3} Y_{l4}^{m4} d\Omega = \sum_{L,M} (-1)^M \sqrt{\frac{(2l1+1)...}{4\pi}} ...
    # Actually, a simpler way for fixed l=10 is to use the quadrature grid once to compute K,
    # but the directive says "avoiding the high-dimensional non-convexity of the spatial grid optimization entirely".
    # Computing K via grid is fine for PRECOMPUTATION, as long as the OPTIMIZATION is in coefficient space.
    # However, the directive suggests using closure relations/Gaunt. Let's use Gaunt for exactness.
    
    # Gaunt coefficient for three spherical harmonics:
    # G(l1,m1,l2,m2,l3,m3) = \int Y_{l1}^{m1} Y_{l2}^{m2} Y_{l3}^{m3} d\Omega
    # = \sqrt{\frac{(2l1+1)(2l2+1)(2l3+1)}{4\pi}} * ( l1 l2 l3; 0 0 0 ) * ( l1 l2 l3; m1 m2 m3 )
    # We need \int Y_i Y_j Y_k Y_l d\Omega. We can write this as \sum_{L,M} <Y_i Y_j | L,M> <L,M | Y_k Y_l>
    # Or more directly, use the 3j symbol product:
    # \int Y_{l1}^{m1} Y_{l2}^{m2} Y_{l3}^{m3} Y_{l4}^{m4} d\Omega = \sum_{L,M} (-1)^M \sqrt{\frac{(2l1+1)(2l2+1)(2l3+1)(2l4+1)}{4\pi}}
    #   * ( l1 l2 L; 0 0 0 ) * ( l1 l2 L; m1 m2 -M ) * ( l3 l4 L; 0 0 0 ) * ( l3 l4 L; m3 m4 M )
    # Since all l=10, this simplifies.
    
    # Let's compute K using the grid for simplicity and speed in precomputation, 
    # but the OPTIMIZATION will be purely spectral. The "avoiding spatial grid optimization" 
    # means the optimizer doesn't see the grid, it sees a polynomial.
    
    # Actually, let's stick to the grid for K computation as it's robust and fast enough for 21^4 = 194481 terms.
    mu, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi
    
    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    
    # Build basis matrix A (n_points x 21)
    A = np.zeros((len(mu_flat), n_m))
    for m in m_vals:
        m_abs = abs(m)
        idx = m_to_idx(m)
        P = lpmv(m_abs, k, mu_flat)
        N_m = N[idx]
        if m == 0:
            A[:, idx] = N_m * P
        elif m > 0:
            A[:, idx] = np.sqrt(2) * N_m * P * np.cos(m * phi_flat)
        else:
            A[:, idx] = np.sqrt(2) * N_m * P * np.sin(m_abs * phi_flat)
            
    # Compute K_{i,j,k,l} = \sum_p w_p d_phi A_{p,i} A_{p,j} A_{p,k} A_{p,l}
    # This is a 4-index tensor. We can compute it as:
    # K = np.einsum('pi,pj,pk,pl->ijkl', A*w_flat*d_phi, A, A, A) -- too slow for 21^4*9100
    # Better: K = (A.T @ (A * (A.T @ A))) ... no.
    # K_{ijkl} = \sum_p A_{pi} A_{pj} A_{pk} A_{pl} w_p d_phi
    # Let M = A.T @ (A * (w_flat*d_phi)) ... no.
    # Let B = A * (w_flat*d_phi)**0.25 ... no.
    # K = np.einsum('pi,pj,pk,pl->ijkl', A, A, A, A, optimize=True) with weights?
    # We can do: W = np.diag(w_flat*d_phi). K = A.T @ W @ A @ A.T @ W @ A? No.
    # K_{ijkl} = \sum_p A_{pi} A_{pj} A_{pk} A_{pl} w_p
    # Let V = A.T @ (A * (w_flat)**0.5) ... 
    # Simple loop over i,j,k,l is 21^4 = 194481 iterations. Each is a dot product of length 9100.
    # 194481 * 9100 * 4 flops ~ 7e9 flops. Too slow in Python.
    
    # Alternative: Use the fact that K is symmetric.
    # We can compute K as a 4D array using broadcasting if we chunk it.
    # Or, we can note that we only need to minimize R(c) = (c^T K c / (c^T c)^2)^{1/4}
    # where (c^T K c) is not a quadratic form, it's a quartic form.
    # Let Q(c) = \sum_{i,j,k,l} K_{ijkl} c_i c_j c_k c_l.
    # We can compute Q(c) efficiently if we precompute K.
    
    # Let's compute K using a smarter method:
    # K_{ijkl} = \int Y_i Y_j Y_k Y_l d\Omega
    # We can use the 3j symbols.
    # Since all l=10, the 3j symbol (10 10 L; 0 0 0) is non-zero only for L even.
    # L can be 0, 2, ..., 20.
    
    # Let's use scipy.special.sph_harm_y? No, it's for evaluation.
    # We need 3j symbols. scipy.special.wigner_3j is available.
    
    from scipy.special import wigner_3j
    
    # Precompute 3j symbols for (10, 10, L; 0, 0, 0) and (10, 10, L; m1, m2, M)
    L_vals = np.arange(0, 21, 2)  # 0, 2, ..., 20
    
    # Precompute C_L = sqrt((2*10+1)^4 / (4*pi)) * (10 10 L; 0 0 0)^2
    # Actually, the formula is:
    # \int Y_{l1}^{m1} Y_{l2}^{m2} Y_{l3}^{m3} Y_{l4}^{m4} d\Omega = \sum_{L,M} (-1)^M \sqrt{\frac{(2l1+1)(2l2+1)(2l3+1)(2l4+1)}{4\pi}}
    #   * ( l1 l2 L; 0 0 0 ) * ( l1 l2 L; m1 m2 -M ) * ( l3 l4 L; 0 0 0 ) * ( l3 l4 L; m3 m4 M )
    
    # Let's precompute the 3j symbols.
    # j1=10, j2=10, j3=L, m1=m, m2=m', m3=M
    # We need (10 10 L; 0 0 0) for L in L_vals.
    # And (10 10 L; m1 m2 M) for m1,m2 in [-10,10], M in [-L,L].
    
    # This is a lot of precomputation. Let's do it.
    
    # 3j symbol (j1 j2 j3; m1 m2 m3)
    # wigner_3j(j1, j2, j3, m1, m2, m3)
    
    # Precompute (10 10 L; 0 0 0)
    C0 = np.zeros(len(L_vals))
    for i, L in enumerate(L_vals):
        C0[i] = wigner_3j(10, 10, L, 0, 0, 0)
        
    # Precompute (10 10 L; m1 m2 M)
    # We need this for all m1, m2 in [-10,10] and M in [-L,L]
    # This is a 3D array: C_m1_m2_M[L_idx, m1_idx, m2_idx, M_idx]
    # But M depends on L.
    # Let's store it in a dictionary or a padded array.
    
    # Actually, we can compute K directly using the formula:
    # K_{i,j,k,l} = \sum_{L,M} (-1)^M * Prefactor * C0_L * C_m1_m2_-M_L * C_m3_m4_M_L
    # where Prefactor = sqrt((21)^4 / (4*pi)) = 21^2 / (2*sqrt(pi))
    
    Prefactor = 21.0**2 / (2.0 * np.sqrt(np.pi))
    
    K = np.zeros((n_m, n_m, n_m, n_m))
    
    for i, L in enumerate(L_vals):
        C0_L = C0[i]
        if abs(C0_L) < 1e-15:
            continue
        M_vals = np.arange(-L, L+1)
        for M in M_vals:
            # Compute C_m1_m2_-M and C_m3_m4_M
            # C_m1_m2_M = wigner_3j(10, 10, L, m1, m2, M)
            # We need to sum over m1,m2 and m3,m4
            # Let's compute the vector v_M[m1,m2] = wigner_3j(10, 10, L, m1, m2, M)
            # Then K += (-1)^M * Prefactor * C0_L * outer(outer(v_-M, v_-M), outer(v_M, v_M))
            # Wait, the formula is:
            # \sum_{L,M} (-1)^M * Prefactor * C0_L * (10 10 L; m1 m2 -M) * (10 10 L; m3 m4 M)
            # So for fixed L, M:
            # Term = (-1)^M * Prefactor * C0_L
            # K += Term * np.einsum('ab,cd->abcd', v_-M, v_M)
            
            v_M = np.zeros((n_m, n_m))
            v_negM = np.zeros((n_m, n_m))
            for m1 in m_vals:
                for m2 in m_vals:
                    # Check triangle condition: |m1+m2| <= L <= 10+10
                    # wigner_3j returns 0 if invalid
                    v_M[m_to_idx(m1), m_to_idx(m2)] = wigner_3j(10, 10, L, m1, m2, M)
                    v_negM[m_to_idx(m1), m_to_idx(m2)] = wigner_3j(10, 10, L, m1, m2, -M)
            
            # K_{i,j,k,l} += Term * v_negM_{i,j} * v_M_{k,l}
            # This is an outer product of two matrices.
            term = (-1)**M * Prefactor * C0_L
            K += term * np.einsum('ij,kl->ijkl', v_negM, v_M, optimize=True)
    
    # Now K is the 4th-order moment tensor.
    # ||f||_4^4 = \sum_{i,j,k,l} K_{ijkl} c_i c_j c_k c_l
    # ||f||_2^2 = \sum_i c_i^2 (since basis is orthonormal, \int Y_i Y_j = \delta_{ij})
    
    # Let's verify orthonormality:
    # \int Y_i Y_j d\Omega = \delta_{ij}
    # For real orthonormal basis, yes.
    
    # Objective: R(c) = (Q(c) / (c^T c)^2)^{1/4}
    # where Q(c) = \sum_{i,j,k,l} K_{ijkl} c_i c_j c_k c_l
    
    # We can compute Q(c) efficiently:
    # Q(c) = c^T (K * c) c^T ... no.
    # Q(c) = \sum_{i,j} c_i c_j (K_{ij} * c^2)_{kl} ... no.
    # Q(c) = \sum_{i,j,k,l} K_{ijkl} c_i c_j c_k c_l
    # Let M(c) = K * c (tensor contraction)
    # M_{ijk} = \sum_l K_{ijkl} c_l
    # Then Q(c) = \sum_{i,j,k} M_{ijk} c_i c_j c_k
    # Let N(c) = M(c) * c (contraction)
    # N_{ij} = \sum_k M_{ijk} c_k = \sum_{k,l} K_{ijkl} c_k c_l
    # Then Q(c) = \sum_{i,j} N_{ij} c_i c_j = c^T N(c) c
    
    # So Q(c) = c^T (K : c) c, where (K : c) is the matrix with entries N_{ij}.
    
    def compute_N(c):
        """Compute N_{ij} = \sum_{k,l} K_{ijkl} c_k c_l"""
        # N = np.einsum('ijkl,kl->ij', K, np.outer(c, c), optimize=True)
        # This is 21^4 * 21^2 = too slow?
        # 21^4 = 194481. 21^2 = 441. 194481 * 441 = 85e6. Fast enough.
        return np.einsum('ijkl,kl->ij', K, np.outer(c, c), optimize=True)
    
    def objective_and_grad(c):
        norm_c2 = np.dot(c, c)
        if norm_c2 < 1e-12:
            return 1e10, np.zeros(n_m)
        u = c / np.sqrt(norm_c2)
        # Q(u) = u^T N(u) u
        N_u = compute_N(u)
        Q = np.einsum('i,ij,j->', u, N_u, u)
        # R = (Q / 1)^{1/4} = Q^{1/4}
        if Q < 1e-12:
            return 1e10, np.zeros(n_m)
        R = Q**0.25
        
        # Gradient of R with respect to u
        # dR/du = (1/4) Q^{-3/4} dQ/du
        # dQ/du_i = \sum_{j,k,l} (K_{ijkl} u_j u_k u_l + K_{ijkl} u_j u_k u_l + ...)
        # By symmetry, dQ/du_i = 4 \sum_{j,k,l} K_{ijkl} u_j u_k u_l = 4 (K : u^3)_i
        # Let V_i = \sum_{j,k,l} K_{ijkl} u_j u_k u_l
        # V = np.einsum('ijkl,jkl->i', K, np.einsum('j,k,l->jkl', u, u, u), optimize=True)
        # Or V = np.einsum('ijkl,j,k,l->i', K, u, u, u, optimize=True)
        V = np.einsum('ijkl,j,k,l->i', K, u, u, u, optimize=True)
        dQ_du = 4 * V
        dR_du = 0.25 * Q**(-0.75) * dQ_du
        
        # Chain rule for c -> u = c/||c||
        u_dot_grad = np.dot(u, dR_du)
        grad_c = (dR_du - u * u_dot_grad) / np.sqrt(norm_c2)
        
        return R, grad_c
    
    # Multi-start optimization
    best_c = None
    best_val = np.inf
    
    # Initial guesses
    # 1. m=0 only
    c0 = np.zeros(n_m)
    c0[m_to_idx(0)] = 1.0
    # 2. m=10 only
    c10 = np.zeros(n_m)
    c10[m_to_idx(10)] = 1.0
    # 3. Random
    rng = np.random.default_rng(42)
    
    starts = [c0, c10]
    for _ in range(10):
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        starts.append(c_rand)
    
    for c_init in starts:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # If L-BFGS-B didn't find anything good, try differential evolution on the spectral objective
    if time.time() < deadline - 60:
        try:
            bounds = [(-1.0, 1.0)] * n_m
            def de_objective(c):
                val, _ = objective_and_grad(c)
                return val
            de_res = differential_evolution(
                de_objective, 
                bounds, 
                maxiter=50, 
                popsize=20, 
                tol=1e-8,
                polish=False,
                seed=42
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_c = de_res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if best_c is not None:
        try:
            res = minimize(
                objective_and_grad, 
                best_c, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Fallback
    if best_c is None or not np.all(np.isfinite(best_c)):
        best_c = c0
    
    # Validity check
    if not np.all(np.isfinite(best_c)):
        best_c = c0
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)
            
    return best_c
