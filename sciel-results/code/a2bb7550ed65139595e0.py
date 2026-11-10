# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize
import time
import warnings
from math import lgamma, pi, sqrt

# Precompute quadrature and harmonic basis at module load
def _build_quadrature():
    n_theta = 70
    n_phi = 130
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    return x, w, phi

def _build_basis():
    k = 10
    x, w, phi = _QUAD
    n_theta = len(x)
    n_phi = len(phi)
    
    P = np.zeros((11, n_theta), dtype=np.float64)
    for m in range(11):
        P[m, :] = lpmv(m, k, x)
    
    N = np.zeros(11, dtype=np.float64)
    for m in range(11):
        log_N = 0.5 * (np.log(21.0 / (4.0 * pi)) + lgamma(k - m + 1) - lgamma(k + m + 1))
        N[m] = np.exp(log_N)
    
    Y = np.zeros((21, n_theta, n_phi), dtype=np.float64)
    Y[10, :, :] = N[0] * P[0, :][:, None]
    
    for m in range(1, 11):
        coeff = np.sqrt(2.0) * N[m] * P[m, :]
        Y[10 + m, :, :] = coeff[:, None] * np.cos(m * phi)[None, :]
        Y[10 - m, :, :] = coeff[:, None] * np.sin(m * phi)[None, :]
    
    return Y, w, phi

_QUAD = _build_quadrature()
_Y_BASIS, _W, _PHI = _build_basis()
_N_PHI = len(_PHI)
_DPHI = 2.0 * np.pi / _N_PHI
_N_THETA = len(_W)

def _ratio_val_and_grad(c):
    """Compute R = ||f||_4 / ||f||_2 and its gradient."""
    f = np.tensordot(c, _Y_BASIS, axes=(0, 0))
    f2 = f * f
    f4 = f2 * f2
    
    integral_f2 = _DPHI * np.sum(_W[:, None] * f2, axis=(0, 1))
    integral_f4 = _DPHI * np.sum(_W[:, None] * f4, axis=(0, 1))
    
    if integral_f2 < 1e-30:
        return np.inf, np.zeros(21)
    
    norm2 = np.sqrt(integral_f2)
    norm4 = np.power(integral_f4, 0.25)
    R = norm4 / norm2
    
    d_int_f2 = _DPHI * np.sum(_W[:, None] * 2.0 * f * _Y_BASIS, axis=(1, 2))
    d_int_f4 = _DPHI * np.sum(_W[:, None] * 4.0 * f2 * f * _Y_BASIS, axis=(1, 2))
    
    grad = (0.25 * np.power(integral_f4, -0.75) * d_int_f4 / norm2
            - norm4 * 0.5 * np.power(integral_f2, -1.5) * d_int_f2)
    
    return R, grad

def _ratio_val(c):
    R, _ = _ratio_val_and_grad(c)
    return R

def _hessian(c):
    """Compute the Hessian of R = ||f||_4 / ||f||_2."""
    f = np.tensordot(c, _Y_BASIS, axes=(0, 0))
    f2 = f * f
    f4 = f2 * f2
    
    integral_f2 = _DPHI * np.sum(_W[:, None] * f2, axis=(0, 1))
    integral_f4 = _DPHI * np.sum(_W[:, None] * f4, axis=(0, 1))
    
    if integral_f2 < 1e-30:
        return np.zeros((21, 21))
    
    norm2_sq = integral_f2
    norm4_4th = integral_f4
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.power(norm4_4th, 0.25)
    
    # R = (integral_f4)^(1/4) / (integral_f2)^(1/2)
    # Let A = integral_f2, B = integral_f4
    # R = B^(1/4) * A^(-1/2)
    # dR/dA = -0.5 * B^(1/4) * A^(-3/2)
    # dR/dB = 0.25 * B^(-3/4) * A^(-1/2)
    # d2R/dA2 = 0.75 * B^(1/4) * A^(-5/2)
    # d2R/dB2 = -0.1875 * B^(-7/4) * A^(-1/2)
    # d2R/dA dB = -0.125 * B^(-3/4) * A^(-3/2)
    
    # dA/dc_j = 2 * <f, Y_j>
    # dB/dc_j = 4 * <f^3, Y_j>
    # d2A/dc_j dc_k = 2 * <Y_j, Y_k> = 2 * delta_jk (orthonormal basis)
    # d2B/dc_j dc_k = 4 * <3 f^2 Y_j, Y_k> + 4 * <f^2 Y_j, Y_k> * ... 
    # Actually dB = 4 * integral f^3 Y_j dOmega
    # d2B/dc_j dc_k = 4 * integral 3 f^2 Y_j Y_k dOmega
    
    # Let's compute using matrix form
    # A = c^T G c where G = <Y_i, Y_j> = delta_ij (orthonormal)
    # Actually A = integral f^2 = integral (sum c_i Y_i)^2 = sum c_i c_j <Y_i, Y_j> = c^T c (since orthonormal)
    # Wait, the basis is orthonormal on the sphere, so <Y_i, Y_j> = delta_ij
    # So A = ||c||^2 = c^T c
    # B = integral f^4 = integral (sum c_i Y_i)^4
    
    # dA/dc = 2c
    # d2A/dc dc = 2I
    
    # dB/dc_j = 4 integral f^3 Y_j
    # d2B/dc_j dc_k = 12 integral f^2 Y_j Y_k
    
    # Hessian of R:
    # H_jk = d2R/dc_j dc_k
    # = d(dR/dB)/dB * dB2_jk + d(dR/dB)/dA * dA2_jk + d(dR/dA)/dA * dA2_jk + d(dR/dA)/dB * dB2_jk + dR/dB * d2B_jk + dR/dA * d2A_jk
    # Wait, let me be more careful.
    # R = B^(1/4) A^(-1/2)
    # dR/dc_j = (1/4) B^(-3/4) A^(-1/2) dB_j - (1/2) B^(1/4) A^(-3/2) dA_j
    # where dB_j = dB/dc_j, dA_j = dA/dc_j
    
    # d2R/dc_j dc_k = d/d c_k [ (1/4) B^(-3/4) A^(-1/2) dB_j - (1/2) B^(1/4) A^(-3/2) dA_j ]
    # = (1/4) [ (-3/4) B^(-7/4) dB_k A^(-1/2) dB_j + B^(-3/4) (-1/2) A^(-3/2) dA_k dB_j + B^(-3/4) A^(-1/2) d2B_jk ]
    #   - (1/2) [ (1/4) B^(-3/4) dB_k A^(-3/2) dA_j + B^(1/4) (-3/2) A^(-5/2) dA_k dA_j + B^(1/4) A^(-3/2) d2A_jk ]
    
    # Since A = c^T c, dA_j = 2 c_j, d2A_jk = 2 delta_jk
    # dB_j = 4 integral f^3 Y_j
    # d2B_jk = 12 integral f^2 Y_j Y_k
    
    # Let's compute the integrals
    Yf2 = np.tensordot(f2, _Y_BASIS, axes=(0, 0))  # shape (21,) = integral f^2 Y_j * dphi * w
    Yf3 = np.tensordot(f2 * f, _Y_BASIS, axes=(0, 0))  # shape (21,) = integral f^3 Y_j
    
    # Actually we need weighted sums
    Yf2_int = _DPHI * np.sum(_W[:, None] * f2 * _Y_BASIS, axis=(1, 2))  # shape (21,)
    Yf3_int = _DPHI * np.sum(_W[:, None] * f2 * f * _Y_BASIS, axis=(1, 2))  # shape (21,)
    
    # d2B_jk = 12 * integral f^2 Y_j Y_k
    # We need a 21x21 matrix: H2B[j,k] = 12 * integral f^2 Y_j Y_k
    # This is 12 * (Y^T * W * f^2 * Y) in some sense
    # Let M[j,k] = integral f^2 Y_j Y_k dOmega
    # M = (DPHI) * sum_i W_i * (Y_j(x_i) Y_k(x_i) f^2(x_i)) summed over phi
    
    # More efficiently: M[j,k] = DPHI * sum_i W_i * sum_l Y_j(x_i,phi_l) Y_k(x_i,phi_l) f^2(x_i,phi_l)
    # = DPHI * sum_i W_i * (Y_i^T diag(f^2_i) Y_i) where Y_i is the 21-vector of basis values at theta_i
    
    # Let's compute this
    M = np.zeros((21, 21), dtype=np.float64)
    for i in range(_N_THETA):
        Yi = _Y_BASIS[:, i, :]  # shape (21, n_phi)
        fi2 = f2[i, :]  # shape (n_phi,)
        # M_i = Yi^T diag(fi2) Yi
        # = (Yi * fi2[None, :])^T Yi
        M += _W[i] * (Yi * fi2[None, :]).T @ Yi
    M *= _DPHI
    d2B = 12.0 * M
    
    dB = 4.0 * Yf3_int
    dA = 2.0 * c
    d2A = 2.0 * np.eye(21)
    
    B = norm4_4th
    A = norm2_sq
    
    coeff1 = 0.25 * np.power(B, -0.75) * np.power(A, -0.5)
    coeff2 = -0.5 * np.power(B, 0.25) * np.power(A, -1.5)
    
    # dR/dc_j = coeff1 * dB_j + coeff2 * dA_j
    
    # d2R/dc_j dc_k:
    # Term 1: d(coeff1)/dB * dB_k * dB_j + coeff1 * d2B_jk
    # d(coeff1)/dB = 0.25 * (-0.75) * B^(-1.75) * A^(-0.5) = -0.1875 * B^(-1.75) * A^(-0.5)
    # d(coeff1)/dA = 0.25 * B^(-0.75) * (-0.5) * A^(-1.5) = -0.125 * B^(-0.75) * A^(-1.5)
    
    # Term 2: d(coeff2)/dB * dB_k * dA_j + coeff2 * d2B_jk
    # Wait, no. Let me redo.
    
    # R = B^(1/4) A^(-1/2)
    # dR/dc_j = (1/4) B^(-3/4) A^(-1/2) * dB_j + B^(1/4) * (-1/2) A^(-3/2) * dA_j
    
    # d2R/dc_j dc_k = 
    #   (1/4) [ (-3/4) B^(-7/4) * dB_k * A^(-1/2) * dB_j + B^(-3/4) * (-1/2) A^(-3/2) * dA_k * dB_j + B^(-3/4) A^(-1/2) * d2B_jk ]
    #   + (-1/2) [ (1/4) B^(-3/4) * dB_k * A^(-3/2) * dA_j + B^(1/4) * (-3/2) A^(-5/2) * dA_k * dA_j + B^(1/4) A^(-3/2) * d2A_jk ]
    
    # = (-3/16) B^(-7/4) A^(-1/2) * outer(dB, dB)
    #   - (1/8) B^(-3/4) A^(-3/2) * outer(dA, dB)
    #   - (1/8) B^(-3/4) A^(-3/2) * outer(dB, dA)
    #   + (1/4) B^(-3/4) A^(-1/2) * d2B
    #   - (1/8) B^(-3/4) A^(-3/2) * outer(dB, dA)
    #   + (3/4) B^(1/4) A^(-5/2) * outer(dA, dA)
    #   - (1/2) B^(1/4) A^(-3/2) * d2A
    
    # Note: outer(dA, dB) + outer(dB, dA) = 2 * outer(dA, dB) if symmetric, but they're different
    # Actually outer(dA, dB)[j,k] = dA_j * dB_k and outer(dB, dA)[j,k] = dB_j * dA_k
    # These are transposes of each other.
    
    H = np.zeros((21, 21), dtype=np.float64)
    
    c1 = (-3.0/16.0) * np.power(B, -1.75) * np.power(A, -0.5)
    c2 = (-1.0/8.0) * np.power(B, -0.75) * np.power(A, -1.5)
    c3 = (1.0/4.0) * np.power(B, -0.75) * np.power(A, -0.5)
    c4 = (3.0/4.0) * np.power(B, 0.25) * np.power(A, -2.5)
    c5 = (-1.0/2.0) * np.power(B, 0.25) * np.power(A, -1.5)
    
    H += c1 * np.outer(dB, dB)
    H += c2 * (np.outer(dA, dB) + np.outer(dB, dA))
    H += c3 * d2B
    H += c4 * np.outer(dA, dA)
    H += c5 * d2A
    
    return H

def _newton_polish(c, max_iter=20, tol=1e-14):
    """Newton's method for minimizing R."""
    for it in range(max_iter):
        R, g = _ratio_val_and_grad(c)
        if not np.isfinite(R):
            break
        if np.linalg.norm(g) < tol:
            break
        
        H = _hessian(c)
        
        # Solve H * delta = -g
        try:
            # Add small regularization for numerical stability
            H_reg = H + 1e-10 * np.eye(21)
            delta = np.linalg.solve(H_reg, -g)
        except np.linalg.LinAlgError:
            delta = -g * 0.01
        
        # Line search
        alpha = 1.0
        for _ in range(20):
            c_new = c + alpha * delta
            R_new, _ = _ratio_val_and_grad(c_new)
            if np.isfinite(R_new) and R_new < R:
                break
            alpha *= 0.5
        
        c = c_new
    
    return c

def rudin_problem10() -> np.ndarray:
    k = 10
    n = 21
    
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    best_c = np.ones(n, dtype=np.float64)
    best_R = _ratio_val(best_c)
    
    np.random.seed(42)
    
    # Generate diverse initializations
    inits = []
    
    # Random unit vectors
    for _ in range(30):
        c0 = np.random.randn(n)
        c0 /= np.linalg.norm(c0)
        inits.append(c0)
    
    # Single basis vectors
    for i in range(n):
        c0 = np.zeros(n)
        c0[i] = 1.0
        inits.append(c0)
    
    # Pairs of basis vectors
    for i in range(n):
        for j in range(i+1, n):
            c0 = np.zeros(n)
            c0[i] = 1.0 / np.sqrt(2)
            c0[j] = 1.0 / np.sqrt(2)
            inits.append(c0)
    
    # Symmetric patterns
    # Only m=0
    c0 = np.zeros(n); c0[10] = 1.0; inits.append(c0)
    # Only m=±10
    c0 = np.zeros(n); c0[0] = 1.0; inits.append(c0)
    c0 = np.zeros(n); c0[20] = 1.0; inits.append(c0)
    # m=0 and m=±10
    c0 = np.zeros(n); c0[10] = 1.0/np.sqrt(3); c0[0] = 1.0/np.sqrt(3); c0[20] = 1.0/np.sqrt(3); inits.append(c0)
    
    # Optimize each initialization
    for s, c0 in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        try:
            result = minimize(
                _ratio_val_and_grad,
                c0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500}
            )
            
            c_opt = result.x
            R_opt = _ratio_val(c_opt)
            
            if np.isfinite(R_opt) and R_opt < best_R:
                best_R = R_opt
                best_c = c_opt.copy()
        except Exception:
            continue
    
    # Newton polish on best result
    if time.time() < deadline - 60.0:
        try:
            c_newton = _newton_polish(best_c, max_iter=30, tol=1e-15)
            R_newton = _ratio_val(c_newton)
            if np.isfinite(R_newton) and R_newton < best_R:
                best_R = R_newton
                best_c = c_newton
        except Exception:
            pass
    
    # Additional Newton polish from a few other good candidates
    if time.time() < deadline - 30.0:
        for c0 in inits[:10]:
            if time.time() > deadline - 30.0:
                break
            try:
                result = minimize(
                    _ratio_val_and_grad,
                    c0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500}
                )
                c_opt = result.x
                R_opt = _ratio_val(c_opt)
                
                if np.isfinite(R_opt) and R_opt < best_R + 1e-6:
                    c_newton = _newton_polish(c_opt, max_iter=30, tol=1e-15)
                    R_newton = _ratio_val(c_newton)
                    if np.isfinite(R_newton) and R_newton < best_R:
                        best_R = R_newton
                        best_c = c_newton
            except Exception:
                continue
    
    # Validate and rescale
    if best_c.size != n:
        best_c = np.ones(n, dtype=np.float64)
    
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(n, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 0:
            best_c = best_c / max_abs * 1.0
        else:
            best_c = np.ones(n, dtype=np.float64)
    
    return best_c

# EVOLVE-BLOCK-END
