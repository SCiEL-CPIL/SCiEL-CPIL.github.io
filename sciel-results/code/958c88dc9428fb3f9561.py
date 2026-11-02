import numpy as np
import scipy.optimize as opt
import time
import math
from math import gamma, sqrt, pi

def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s limit

    # Exact integrals using Gamma function
    # For n even: ∫ x^{2n} e^{-x²} dx = Γ(n+1/2) = (2n)! / (4^n n!) * sqrt(pi)
    # Precompute needed gamma values
    # P is even polynomial in x (degrees 0,4,8,12 -> after constraint, still even)
    # P(x) = sum_{k=0}^{6} a_k x^{2k} where k corresponds to degree 2k
    # We need integrals of x^{2m} * P^2 * e^{-x^2} for various m
    # And Q(x) = P'(x) - x*P(x), which is also even (since P is even)
    # Q has terms x^{2k-1} * (2k*a_k) + x^{2k+1}*(-a_k) = x^{2k+1}*(2k-1)*a_k ... 
    # Wait, let me be more careful.
    
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1260x^8 - 9240x^6 + 23100x^4 - 27720x^2 + 10395
    # c12 = -(c0 + 3*c4 + 105*c8)/10395
    
    # P(x) = sum_{j=0}^{6} p_j * x^{2j}
    # where p_j = c0*h0_j + c4*h4_j + c8*h8_j + c12*h12_j
    # h0: [1, 0, 0, 0, 0, 0, 0]
    # h4: [3, -6, 1, 0, 0, 0, 0]  (coeff of x^0, x^2, x^4, ...)
    # h8: [105, -420, 210, -28, 1, 0, 0]
    # h12: [10395, -27720, 23100, -9240, 1260, -66, 1]
    
    h0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=float)
    h4 = np.array([3, -6, 1, 0, 0, 0, 0], dtype=float)
    h8 = np.array([105, -420, 210, -28, 1, 0, 0], dtype=float)
    h12 = np.array([10395, -27720, 23100, -9240, 1260, -66, 1], dtype=float)
    
    # c12 = -(c0 + 3*c4 + 105*c8) / 10395
    # So P = c0*(h0 - h12/10395) + c4*(h4 - 3*h12/10395) + c8*(h8 - 105*h12/10395)
    
    basis0 = h0 - h12 / 10395.0
    basis4 = h4 - 3.0 * h12 / 10395.0
    basis8 = h8 - 105.0 * h12 / 10395.0
    
    # P(x) = c0*basis0 + c4*basis4 + c8*basis8
    # p_j = c0*basis0[j] + c4*basis4[j] + c8*basis8[j]
    
    # Q(x) = P'(x) - x*P(x)
    # P'(x) = sum 2j*p_j * x^{2j-1}
    # x*P(x) = sum p_j * x^{2j+1}
    # Q(x) = sum_{j=0}^{5} (2j*p_j - p_{j-1}) * x^{2j-1} ... let me index differently
    # Q(x) = sum_{k=0}^{5} q_k * x^{2k+1} where q_k = (2(k+1))*p_{k+1} - p_k
    # Actually: P'(x) = sum_{j=1}^{6} 2j*p_j*x^{2j-1}
    # x*P(x) = sum_{j=0}^{5} p_j*x^{2j+1}
    # Q(x) = sum_{j=1}^{6} 2j*p_j*x^{2j-1} - sum_{j=0}^{5} p_j*x^{2j+1}
    # Let m = 2j-1 for first sum (j=1..6 -> m=1,3,5,7,9,11)
    # Let m = 2j+1 for second sum (j=0..5 -> m=1,3,5,7,9,11)
    # Q(x) = sum_{j=0}^{5} [(2(j+1))*p_{j+1} - p_j] * x^{2j+1}
    
    # So Q is odd polynomial: Q(x) = sum_{j=0}^{5} q_j * x^{2j+1}
    # q_j = 2(j+1)*p_{j+1} - p_j
    
    # Now for integrals:
    # I0 = ∫ P^2 e^{-x^2} dx = 2 * sum_{j,k} p_j*p_k * ∫_0^∞ x^{2j+2k} e^{-x^2} dx
    #    = 2 * sum_{j,k} p_j*p_k * Gamma(j+k+1/2)/2
    #    = sum_{j,k} p_j*p_k * Gamma(j+k+1/2)
    # Wait: ∫_{-∞}^{∞} x^{2n} e^{-x^2} dx = Gamma(n+1/2)
    # So I0 = sum_{j,k} p_j*p_k * Gamma(j+k+1/2)
    
    # Ix2 = ∫ x^2 P^2 e^{-x^2} dx = sum_{j,k} p_j*p_k * Gamma(j+k+3/2)
    
    # Ideriv = ∫ Q^2 e^{-x^2} dx = sum_{j,k} q_j*q_k * Gamma((2j+1+2k+1)/2 + 1/2)
    #        = sum_{j,k} q_j*q_k * Gamma(j+k+2)
    # Since ∫ x^{2(j+k+1)} e^{-x^2} dx = Gamma(j+k+1+1/2) = Gamma(j+k+3/2)
    # Wait: Q^2 has terms x^{2j+1+2k+1} = x^{2(j+k+1)}
    # ∫ x^{2(j+k+1)} e^{-x^2} dx = Gamma(j+k+1+1/2) = Gamma(j+k+3/2)
    
    # Let me precompute the gamma matrices
    # G0[j,k] = Gamma(j+k+1/2) for j,k = 0..6
    # Gx2[j,k] = Gamma(j+k+3/2) for j,k = 0..6
    # GQ[j,k] = Gamma(j+k+3/2) for j,k = 0..5 (Q has 6 terms, j,k=0..5)
    
    # Actually let me use a helper
    def gamma_half(n):
        """Gamma(n + 1/2) for integer n >= 0"""
        return gamma(n + 0.5)
    
    # Precompute 7x7 and 6x6 matrices
    G0 = np.zeros((7, 7))
    Gx2 = np.zeros((7, 7))
    for j in range(7):
        for k in range(7):
            G0[j, k] = gamma(j + k + 0.5)
            Gx2[j, k] = gamma(j + k + 1.5)
    
    GQ = np.zeros((6, 6))
    for j in range(6):
        for k in range(6):
            GQ[j, k] = gamma(j + k + 1.5)  # Gamma((2j+1+2k+1)/2 + 1/2) = Gamma(j+k+1+1/2)
    
    # Basis matrices for P coefficients
    # p = B @ c where B is 7x3, c = [c0, c4, c8]
    B = np.column_stack([basis0, basis4, basis8])  # 7x3
    
    # Q coefficients: q_j = 2(j+1)*p_{j+1} - p_j for j=0..5
    # q = D @ p where D is 6x7
    D = np.zeros((6, 7))
    for j in range(6):
        D[j, j] = -1.0
        D[j, j+1] = 2.0 * (j + 1)
    
    # So q = D @ B @ c = DB @ c
    DB = D @ B  # 6x3
    
    def compute_bound(c):
        """Compute the uncertainty bound for coefficients c = [c0, c4, c8]"""
        p = B @ c  # 7-vector
        q = DB @ c  # 6-vector
        
        I0 = p @ G0 @ p
        if I0 <= 1e-30 or not np.isfinite(I0):
            return np.inf
        
        Ix2 = p @ Gx2 @ p
        Ideriv = q @ GQ @ q
        
        if not np.isfinite(Ix2) or not np.isfinite(Ideriv) or Ix2 < 0 or Ideriv < 0:
            return np.inf
        
        A = Ix2 / I0
        Bval = Ideriv / I0
        bound = sqrt(A * Bval / (4.0 * pi**2))
        
        if not np.isfinite(bound) or bound <= 0:
            return np.inf
        return bound
    
    def compute_bound_grad(c):
        """Compute bound and gradient."""
        p = B @ c
        q = DB @ c
        
        I0 = p @ G0 @ p
        if I0 <= 1e-30 or not np.isfinite(I0):
            return np.inf, np.zeros(3)
        
        Ix2 = p @ Gx2 @ p
        Ideriv = q @ GQ @ q
        
        if not np.isfinite(Ix2) or not np.isfinite(Ideriv) or Ix2 < 0 or Ideriv < 0:
            return np.inf, np.zeros(3)
        
        A = Ix2 / I0
        Bval = Ideriv / I0
        bound = sqrt(A * Bval / (4.0 * pi**2))
        
        if not np.isfinite(bound) or bound <= 0:
            return np.inf, np.zeros(3)
        
        # Gradients
        # dI0/dc_i = 2 * (G0 @ p)[i mapped back] ... 
        # Actually: I0 = p^T G0 p, dp/dc_i = B[:, i]
        # dI0/dc_i = 2 * B[:,i]^T G0 p
        dI0 = 2.0 * (B.T @ G0 @ p)  # 3-vector
        dIx2 = 2.0 * (B.T @ Gx2 @ p)
        dIderiv = 2.0 * (DB.T @ GQ @ q)
        
        dA = (dIx2 * I0 - Ix2 * dI0) / (I0 * I0)
        dB = (dIderiv * I0 - Ideriv * dI0) / (I0 * I0)
        
        const = 1.0 / (2.0 * bound * 4.0 * pi**2)
        grad = const * (Bval * dA + A * dB)
        
        if not np.all(np.isfinite(grad)):
            return bound, np.zeros(3)
        
        return bound, grad
    
    # Initial guess (from seed)
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_score, _ = compute_bound_grad(x0)
    best_coeffs = x0.copy()
    
    if not np.isfinite(best_score):
        best_score = 1.0
        best_coeffs = x0.copy()
    
    # Strategy: Multi-start L-BFGS-B + Basin Hopping
    # Since evaluation is very fast (exact analytic), we can do many starts
    
    rng = np.random.RandomState(12345)
    
    # Phase 1: Multi-start L-BFGS-B from diverse initial points
    # Generate initial points: around seed, random in reasonable range
    n_starts = 50
    for i in range(n_starts):
        if time.time() > deadline - 5:
            break
        
        if i == 0:
            c_start = x0.copy()
        elif i < 10:
            # Near best
            c_start = best_coeffs + rng.normal(0, 0.1, 3)
        elif i < 25:
            # Moderate exploration
            c_start = best_coeffs + rng.normal(0, 0.5, 3)
        else:
            # Wide exploration
            c0 = rng.uniform(-3, 1)
            c4 = rng.uniform(-0.5, 0.5)
            c8 = rng.uniform(-0.1, 0.3)
            c_start = np.array([c0, c4, c8])
        
        def obj(x):
            return compute_bound(x)
        
        def obj_grad(x):
            f, g = compute_bound_grad(x)
            return f, g
        
        try:
            res = opt.minimize(obj, c_start, method='L-BFGS-B', 
                              jac=lambda x: compute_bound_grad(x)[1],
                              options={'gtol': 1e-14, 'maxiter': 200, 'ftol': 1e-15})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Basin Hopping for global optimization
    if time.time() < deadline - 10:
        def bh_obj(x):
            return compute_bound(x)
        
        def bh_grad(x):
            return compute_bound_grad(x)[1]
        
        try:
            bh_res = opt.basinhopping(
                bh_obj, 
                best_coeffs.copy(), 
                minimizer_kwargs={'jac': bh_grad, 'method': 'L-BFGS-B', 'options': {'maxiter': 100}},
                niter=30,
                T=0.1,
                stepsize=0.3,
                seed=99,
                minimizer='L-BFGS-B'
            )
            if np.isfinite(bh_res.fun) and bh_res.fun < best_score:
                best_score = bh_res.fun
                best_coeffs = bh_res.xf.copy()
        except Exception:
            pass
    
    # Phase 3: Fine polish around best
    if time.time() < deadline - 2:
        for i in range(20):
            if time.time() > deadline - 1:
                break
            c_start = best_coeffs + rng.normal(0, 0.01, 3)
            try:
                res = opt.minimize(compute_bound, c_start, method='L-BFGS-B',
                                  jac=lambda x: compute_bound_grad(x)[1],
                                  options={'gtol': 1e-15, 'maxiter': 300, 'ftol': 1e-16})
                if np.isfinite(res.fun) and res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass
    
    return np.array(best_coeffs)
