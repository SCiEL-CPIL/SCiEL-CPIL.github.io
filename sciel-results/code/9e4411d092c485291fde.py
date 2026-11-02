import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s limit

    # Constants for Hermite polynomials H_n(x)
    # We use the physicists' convention where H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
    # Orthogonality: int H_m H_n e^{-x^2} dx = sqrt(pi) 2^n n! delta_mn
    
    # Indices for H0, H4, H8, H12
    n_idx = [0, 4, 8, 12]
    
    # Precompute norms: N_n = sqrt(pi) * 2^n * n!
    # We work with 4 coefficients c0, c4, c8, c12
    # Constraint: c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0) = 1
    # H4(0) = 3
    # H8(0) = 105
    # H12(0) = 10395
    
    H_vals_at_0 = np.array([1.0, 3.0, 105.0, 10395.0])
    
    # Compute N_n for n in [0, 4, 8, 12]
    N_n = np.array([np.sqrt(np.pi) * (2.0 ** n) * np.math.factorial(n) for n in n_idx])
    
    # M0 is diagonal with entries N_n
    # Mx2 is diagonal with entries n * N_n (since int x^2 H_n^2 e^{-x^2} = n * int H_n^2 e^{-x^2})
    Mx2_diag = np.array([n * N_n[i] for i, n in enumerate(n_idx)])
    
    # Mderiv:
    # Q(x) = P'(x) - x P(x)
    # For term c_n H_n(x):
    # P' part: c_n 2n H_{n-1}
    # xP part: c_n (0.5 H_{n+1} + 0.5 n H_{n-1})
    # Q part: c_n (1.5 n H_{n-1} - 0.5 H_{n+1})
    # Integral of Q^2 e^{-x^2}:
    # Sum over terms. Since H_{n-1} and H_{n+1} are orthogonal to each other and to others from different n,
    # we can sum the squares of coefficients times their norms.
    
    # Norm of H_k is N_k = sqrt(pi) 2^k k!
    # For n=0: term is -0.5 c0 H_1. Norm contribution: (-0.5)^2 * N_1
    # For n=4: terms 6 c4 H_3 and -0.5 c4 H_5. Norm contribution: (6^2 * N_3 + (-0.5)^2 * N_5) * c4^2
    # For n=8: terms 12 c8 H_7 and -0.5 c8 H_9. Norm contribution: (12^2 * N_7 + (-0.5)^2 * N_9) * c8^2
    # For n=12: terms 18 c12 H_11 and -0.5 c12 H_13. Norm contribution: (18^2 * N_11 + (-0.5)^2 * N_13) * c12^2
    
    # We need N_k for k in [1, 3, 5, 7, 9, 11, 13]
    def get_N(k):
        return np.sqrt(np.pi) * (2.0 ** k) * np.math.factorial(k)
        
    Mderiv_diag = np.zeros(4)
    # n=0
    Mderiv_diag[0] = (0.5 ** 2) * get_N(1)
    # n=4
    Mderiv_diag[1] = (6.0 ** 2) * get_N(3) + (0.5 ** 2) * get_N(5)
    # n=8
    Mderiv_diag[2] = (12.0 ** 2) * get_N(7) + (0.5 ** 2) * get_N(9)
    # n=12
    Mderiv_diag[3] = (18.0 ** 2) * get_N(11) + (0.5 ** 2) * get_N(13)
    
    # Now we have diagonal matrices in the 4D space (c0, c4, c8, c12)
    # We need to reduce to 3D (c0, c4, c8) using c12 = -(c0 + 3c4 + 105c8) / 10395
    
    # Let c4d = [c0, c4, c8, c12]
    # c12 = A @ c3d, where c3d = [c0, c4, c8]
    # A = [-1/10395, -3/10395, -105/10395]
    
    A_c12 = -1.0 / 10395.0 * np.array([1.0, 3.0, 105.0])
    
    # Construct 3x3 matrices M0_3, Mx2_3, Mderiv_3
    # M_3 = M_4[:3,:3] + M_4[:3,3] A + A^T M_4[3,:3] + A^T M_4[3,3] A
    # Since M_4 are diagonal, this simplifies.
    
    def reduce_matrix(M_diag_4):
        # M_4 is diag(M_diag_4)
        # M_3[i,j] = M_4[i,j] if i,j < 3
        #           + M_4[i,3] A_j + A_i M_4[3,j] + A_i M_4[3,3] A_j
        # Since M_4 is diagonal, M_4[i,j] = 0 if i != j.
        M3 = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                val = 0.0
                if i == j:
                    val += M_diag_4[i]
                # Cross terms with index 3
                val += M_diag_4[i] * 0.0 # M_4[i,3] is 0 for i<3
                val += 0.0 # M_4[3,j] is 0 for j<3
                # Wait, M_4 is diagonal, so M_4[i,3] = 0 for i != 3.
                # So the only non-zero terms involving index 3 are M_4[3,3] A_i A_j
                # And the diagonal terms M_4[i,i].
                # Is that right?
                # Let's check the expansion:
                # c^T M c = sum_i M_ii c_i^2 + 2 sum_{i<j} M_ij c_i c_j
                # Here M_4 is diagonal.
                # c_4d = [c3d, c12].
                # I = c3d^T M_4[:3,:3] c3d + 2 c3d^T M_4[:3,3] c12 + c12^2 M_4[3,3]
                # Since M_4[:3,3] is a column vector with 0s except at index 3? No, it's a 3x1 vector.
                # M_4 is diagonal, so M_4[:3,3] is a column vector with all zeros.
                # So the cross term vanishes?
                # Yes, if M_4 is diagonal, the variables are decoupled in the 4D space.
                # But c12 is a linear combination of c0, c4, c8.
                # So I = sum_{i=0}^2 M_4[i,i] c_i^2 + M_4[3,3] (A @ c3d)^2
                # I = sum_{i=0}^2 M_4[i,i] c_i^2 + M_4[3,3] (sum_k A_k c_k)^2
                # I = sum_{i=0}^2 M_4[i,i] c_i^2 + M_4[3,3] sum_k sum_l A_k A_l c_k c_l
                # So M3[i,j] = M_4[i,i] delta_ij + M_4[3,3] A_i A_j
                
                if i == j:
                    val = M_diag_4[i] + M_diag_4[3] * A_c12[i] * A_c12[j]
                else:
                    val = M_diag_4[3] * A_c12[i] * A_c12[j]
                M3[i, j] = val
        return M3
        
    M0_3 = reduce_matrix(N_n)
    Mx2_3 = reduce_matrix(Mx2_diag)
    Mderiv_3 = reduce_matrix(Mderiv_diag)
    
    # Score function
    def compute_score_fast(coeffs):
        if time.time() > deadline - 2:
            return np.inf
        c = coeffs
        I0 = c @ M0_3 @ c
        if I0 <= 1e-30 or not np.isfinite(I0):
            return np.inf
        Ix2 = c @ Mx2_3 @ c
        if not np.isfinite(Ix2) or Ix2 < 0:
            return np.inf
        Ideriv = c @ Mderiv_3 @ c
        if not np.isfinite(Ideriv) or Ideriv < 0:
            return np.inf
        
        A_f_sq = Ix2 / I0
        A_gf_sq = (1.0 / (4.0 * np.pi**2)) * Ideriv / I0
        
        return np.sqrt(A_f_sq * A_gf_sq)

    def compute_grad(coeffs):
        c = coeffs
        I0 = c @ M0_3 @ c
        Ix2 = c @ Mx2_3 @ c
        Ideriv = c @ Mderiv_3 @ c
        
        if I0 <= 1e-30 or not np.isfinite(I0):
            return np.array([np.nan, np.nan, np.nan])
        
        # d(I0)/dc = 2 M0_3 c
        dI0 = 2 * M0_3 @ c
        dIx2 = 2 * Mx2_3 @ c
        dIderiv = 2 * Mderiv_3 @ c
        
        A = Ix2 / I0
        B = Ideriv / (4.0 * np.pi**2 * I0)
        score = np.sqrt(A * B)
        
        if score < 1e-30:
            return np.array([np.nan, np.nan, np.nan])
        
        dA = (dIx2 * I0 - Ix2 * dI0) / (I0**2)
        dB = (dIderiv * I0 - Ideriv * dI0) / (4.0 * np.pi**2 * I0**2)
        
        grad = (B * dA + A * dB) / (2.0 * score)
        
        return grad

    # Initial guess
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = compute_score_fast(x0)
    best_coeffs = x0.copy()
    
    # Optimization
    if time.time() < deadline - 30:
        try:
            res_bfgs = opt.minimize(
                compute_score_fast, 
                best_coeffs, 
                jac=compute_grad,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50}
            )
            if res_bfgs.fun < best_score and np.isfinite(res_bfgs.fun):
                best_score = res_bfgs.fun
                best_coeffs = res_bfgs.x.copy()
        except Exception:
            pass

    # Global search
    bounds = [(-5, 5), (-0.5, 0.5), (-0.05, 0.05)]
    if time.time() < deadline - 20:
        try:
            res_de = opt.differential_evolution(
                compute_score_fast, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=10,
                tol=1e-10,
                init='sobol',
                polish=False
            )
            if res_de.fun < best_score and np.isfinite(res_de.fun):
                best_score = res_de.fun
                best_coeffs = res_de.x.copy()
                
                if time.time() < deadline - 10:
                    try:
                        res_polish = opt.minimize(
                            compute_score_fast, 
                            best_coeffs, 
                            jac=compute_grad,
                            method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14}
                        )
                        if res_polish.fun < best_score and np.isfinite(res_polish.fun):
                            best_score = res_polish.fun
                            best_coeffs = res_polish.x.copy()
                    except Exception:
                        pass
        except Exception:
            pass

    return np.array(best_coeffs)
