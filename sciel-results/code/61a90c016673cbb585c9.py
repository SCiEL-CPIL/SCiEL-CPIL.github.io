"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = 170  # seconds, keep margin for evaluator kill at 210s

    # Degrees of Hermite polynomials involved
    degrees = [0, 4, 8, 12]
    
    # Coefficients matrix A
    # Rows correspond to basis functions psi_0, psi_4, psi_8
    # Columns correspond to H_0, H_4, H_8, H_12
    # Constraint: c12 = -(c0 + 12c4 + 1680c8) / 40320
    # psi_0 = (H0 - H12/40320) * exp(-x^2/2)
    # psi_4 = (H4 - 12H12/40320) * exp(-x^2/2)
    # psi_8 = (H8 - 1680H12/40320) * exp(-x^2/2)
    
    A = np.zeros((3, 4))
    A[0, 0] = 1.0
    A[0, 3] = -1.0 / 40320.0
    
    A[1, 1] = 1.0
    A[1, 3] = -12.0 / 40320.0
    
    A[2, 2] = 1.0
    A[2, 3] = -1680.0 / 40320.0

    # Helper to compute factorial
    def fact(n):
        return math.factorial(n)
        
    # Helper to compute Gaussian moment integral int x^(2m) exp(-x^2) dx
    def gauss_moment(m):
        # int_{-inf}^{inf} x^(2m) exp(-x^2) dx = (2m-1)!! / 2^m * sqrt(pi)
        if m == 0:
            return np.sqrt(np.pi)
        res = 1.0
        for i in range(1, m+1):
            res *= (2*i - 1)
        return res / (2**m) * np.sqrt(np.pi)

    # Precompute Hermite polynomials as coefficient vectors in basis x^k
    # H_n(x) = sum_{k=0}^n c_k x^k
    # We need to multiply polynomials and integrate x^k exp(-x^2).
    # Only even powers contribute to the integral.
    
    def hermite_coeffs(n):
        """Returns list of coefficients c_k such that H_n(x) = sum c_k x^k"""
        if n == 0:
            return [1.0]
        if n == 1:
            return [0.0, 2.0]
        
        c_prev = hermite_coeffs(n-1)
        c_prev2 = hermite_coeffs(n-2)
        
        # H_n = 2x H_{n-1} - 2(n-1) H_{n-2}
        # 2x H_{n-1}: shift coefficients by 1 and multiply by 2
        term1 = [0.0] + [2.0 * c for c in c_prev]
        term2 = [2.0 * (n-1) * c for c in c_prev2]
        
        # Ensure lengths match
        max_len = max(len(term1), len(term2))
        term1 += [0.0] * (max_len - len(term1))
        term2 += [0.0] * (max_len - len(term2))
        
        res = [term1[i] - term2[i] for i in range(max_len)]
        return res

    # Get coefficients for H_0, H_4, H_8, H_12
    H_coeffs = [hermite_coeffs(d) for d in degrees]
    
    # Function to compute integral of P(x) Q(x) exp(-x^2) dx
    # where P, Q are polynomials given by coefficient lists
    def integrate_prod(P, Q):
        # Product polynomial R(x) = P(x)Q(x)
        # R_k = sum_{i+j=k} P_i Q_j
        len_P = len(P)
        len_Q = len(Q)
        R = [0.0] * (len_P + len_Q - 1)
        for i in range(len_P):
            if P[i] == 0: continue
            for j in range(len_Q):
                if Q[j] == 0: continue
                R[i+j] += P[i] * Q[j]
                
        integral = 0.0
        for k in range(len(R)):
            if k % 2 == 0:
                m = k // 2
                integral += R[k] * gauss_moment(m)
        return integral

    # Compute M, Mx, Mf
    # M_ij = int psi_i psi_j dx = int (sum A_ik H_k) (sum A_jl H_l) exp(-x^2)
    # Mx_ij = int x^2 psi_i psi_j dx
    # Mf_ij = int xi^2 hat_psi_i hat_psi_j dxi = - int psi_i'' psi_j dx
    
    M = np.zeros((3, 3))
    Mx = np.zeros((3, 3))
    Mf = np.zeros((3, 3))
    
    for i in range(3):
        for j in range(i, 3):
            # Construct polynomial for psi_i * psi_j
            # psi_i = sum_k A_ik H_k(x)
            # prod = (sum_k A_ik H_k) (sum_l A_jl H_l)
            # We can just sum the integrals of H_k H_l
            
            val_M = 0.0
            val_Mx = 0.0
            val_Mf = 0.0
            
            for k_idx, k_deg in enumerate(degrees):
                a_k = A[i, k_idx]
                if a_k == 0: continue
                H_k = H_coeffs[k_idx]
                
                for l_idx, l_deg in enumerate(degrees):
                    a_l = A[j, l_idx]
                    if a_l == 0: continue
                    H_l = H_coeffs[l_idx]
                    
                    # M part
                    val_M += a_k * a_l * integrate_prod(H_k, H_l)
                    
                    # Mx part: int x^2 H_k H_l exp(-x^2)
                    # x^2 H_k H_l. Let P = H_k H_l. We need int x^2 P exp(-x^2).
                    # This is equivalent to integrating the polynomial x^2 * P(x).
                    # Multiply P by x^2: shift coefficients by 2.
                    P_coeffs = [0.0] * (len(H_k) + len(H_l) - 1)
                    for idx_k in range(len(H_k)):
                        if H_k[idx_k] == 0: continue
                        for idx_l in range(len(H_l)):
                            if H_l[idx_l] == 0: continue
                            P_coeffs[idx_k + idx_l] += H_k[idx_k] * H_l[idx_l]
                    
                    # Multiply by x^2
                    P_x2 = [0.0, 0.0] + P_coeffs
                    val_Mx += a_k * a_l * integrate_prod(P_x2, [1.0]) # Wait, integrate_prod expects two polys.
                    # My integrate_prod(P, Q) computes int P Q exp(-x^2).
                    # So I should call integrate_prod(P_x2, [1.0])? No, [1.0] is 1.
                    # So int P_x2 * 1 * exp(-x^2). Correct.
                    
                    # Mf part: - int psi_i'' psi_j
                    # psi_i'' = sum_k a_k H_k''(x)
                    # H_k''(x) is a polynomial.
                    # We need int H_k''(x) H_l(x) exp(-x^2).
                    # Let's compute H_k'' coefficients.
                    # H_k(x) = sum c_m x^m
                    # H_k'(x) = sum m c_m x^{m-1}
                    # H_k''(x) = sum m(m-1) c_m x^{m-2}
                    
                    H_k_coeffs = H_k
                    H_k_dd = [0.0] * (len(H_k_coeffs) - 2)
                    for m in range(2, len(H_k_coeffs)):
                        H_k_dd[m-2] = m * (m-1) * H_k_coeffs[m]
                    
                    val_Mf += a_k * a_l * integrate_prod(H_k_dd, H_l)
            
            M[i, j] = val_M
            M[j, i] = val_M
            Mx[i, j] = val_Mx
            Mx[j, i] = val_Mx
            Mf[i, j] = val_Mf
            Mf[j, i] = val_Mf

    # Note: Mf calculated as -int psi_i'' psi_j.
    # Let's verify sign.
    # int xi^2 hat_f hat_g = int f'' g (with sign?).
    # F[f''] = (i xi)^2 F[f] = - xi^2 F[f].
    # So int (i xi)^2 F[f] conj(F[g]) = - int xi^2 F[f] conj(F[g]).
    # Also int f'' conj(g) = int (i xi)^2 F[f] conj(F[g]) (Parseval).
    # So int f'' conj(g) = - int xi^2 F[f] conj(F[g]).
    # Thus int xi^2 F[f] conj(F[g]) = - int f'' conj(g).
    # My code computes val_Mf = sum a_k a_l int H_k'' H_l exp(-x^2).
    # This is int psi_i'' psi_j.
    # So Mf_ij = - val_Mf.
    
    Mf = -Mf

    def compute_bound(c):
        c = np.array(c)
        norm2 = c @ M @ c
        if norm2 <= 1e-12 or not np.isfinite(norm2):
            return 1e10
        x2_norm = c @ Mx @ c
        f2_norm = c @ Mf @ c
        
        if x2_norm < 1e-12 or f2_norm < 1e-12 or not np.isfinite(x2_norm) or not np.isfinite(f2_norm):
            return 1e10
            
        delta_x = np.sqrt(x2_norm / norm2)
        delta_xi = np.sqrt(f2_norm / norm2)
        return delta_x * delta_xi

    # Optimization
    # Initial guess from seed
    c_seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    c_seed_norm = c_seed / np.linalg.norm(c_seed)
    
    best_bound = compute_bound(c_seed_norm)
    best_c = c_seed_norm.copy()
    
    # Multi-start optimization
    rng = np.random.RandomState(42)
    
    starts = [c_seed_norm]
    for _ in range(50):
        c_rand = rng.randn(3)
        c_rand /= np.linalg.norm(c_rand)
        starts.append(c_rand)
        
    # Add some structured points
    for t in np.linspace(0.1, np.pi-0.1, 10):
        for p in np.linspace(0.1, 2*np.pi-0.1, 10):
            c_struct = np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
            starts.append(c_struct)

    for c0 in starts:
        if time.time() - start_time >= deadline - 10:
            break
        try:
            res = minimize(
                compute_bound, 
                x0=c0, 
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_c = res.x
        except Exception:
            pass
            
    # Final polish
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                compute_bound, 
                x0=best_c, 
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 5000, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_c = res.x
        except Exception:
            pass

    return np.array(best_c)

# EVOLVE-BLOCK-END
