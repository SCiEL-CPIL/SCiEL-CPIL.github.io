"""Fourier-invariant Hermite-Gaussian test function: minimize the C4 uncertainty
upper bound via direct numerical optimization with multi-start and high-precision polish."""
import numpy as np
import time
from scipy.optimize import minimize


def _c4_bound(a):
    """Compute the C4 upper bound for coefficient vector a = [a0, a4, a8].
    
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 = -(a0+a4+a8) so g(0) = 0.
    
    C4 = (1/8) * (1 - 4*<x^2>) * (1 - 2*<xi^2>)^2 / ||g||^2
    """
    a0, a4, a8 = a
    a12 = -(a0 + a4 + a8)
    
    # Coefficients for H_k * exp(-x^2/2)
    # ||g||^2 = sum c_k^2 * (2^k * k!) / (2*pi)^(1/2) ... 
    # Actually for orthonormal Hermite functions: phi_k = (2^k k! sqrt(pi))^{-1/2} H_k(x) e^{-x^2/2}
    # But we're using H_k(x) e^{-x^2/2} directly, so:
    # <H_k e^{-x^2/2}, H_m e^{-x^2/2}> = sqrt(pi) * 2^k * k! * delta_{km}
    
    # Let's use the standard normalization. The Hermite-Gaussian functions are:
    # psi_k(x) = (1/sqrt(2^k k! sqrt(pi))) H_k(x) e^{-x^2/2}
    # So H_k(x) e^{-x^2/2} = sqrt(2^k k! sqrt(pi)) * psi_k(x)
    
    # For g = sum c_k H_k e^{-x^2/2} = sum c_k sqrt(2^k k! sqrt(pi)) psi_k
    # ||g||^2 = sum c_k^2 * 2^k * k! * sqrt(pi)
    
    # <x^2> = <g, x^2 g> / ||g||^2
    # x^2 psi_k = (1/sqrt(2(k+1))) psi_{k+1} + sqrt(k/2) psi_{k-1} + sqrt((k+1)/2) psi_k  ... 
    # Actually: x psi_k = (1/sqrt(2)) (psi_{k+1} + k psi_{k-1})... let me be more careful.
    
    # Standard: x psi_k = sqrt((k+1)/2) psi_{k+1} + sqrt(k/2) psi_{k-1}
    # So x^2 psi_k = sqrt((k+1)/2) x psi_{k+1} + sqrt(k/2) x psi_{k-1}
    #              = sqrt((k+1)/2)[sqrt((k+2)/2) psi_{k+2} + sqrt((k+1)/2) psi_k] 
    #                + sqrt(k/2)[sqrt(k/2) psi_{k-2} + sqrt((k-1)/2) psi_k]
    #              = sqrt((k+1)(k+2)/4) psi_{k+2} + [(k+1)/2 + (k-1)/2] psi_k + sqrt(k(k-1)/4) psi_{k-2}
    #              = sqrt((k+1)(k+2)/4) psi_{k+2} + k psi_k + sqrt(k(k-1)/4) psi_{k-2}
    
    # So <psi_k, x^2 psi_k> = k (the middle term, since psi are orthonormal)
    # Wait, that gives <x^2> for psi_k alone = k. But for a superposition:
    # <g, x^2 g> = sum_{k,m} c_k c_m sqrt(2^k k! sqrt(pi)) sqrt(2^m m! sqrt(pi)) <psi_k, x^2 psi_m>
    # <psi_k, x^2 psi_m> is nonzero only when m = k, k±2.
    # <psi_k, x^2 psi_k> = k
    # <psi_k, x^2 psi_{k+2}> = sqrt((k+1)(k+2)/4)
    # <psi_k, x^2 psi_{k-2}> = sqrt(k(k-1)/4)
    
    # Similarly for xi (Fourier conjugate, which is -i d/dx for these):
    # xi psi_k = -i d/dx psi_k = i sqrt((k+1)/2) psi_{k+1} - i sqrt(k/2) psi_{k-1}... 
    # Actually the Fourier transform of psi_k is (-i)^k psi_k, so xi = -i d/dx acts as:
    # xi psi_k = i sqrt((k+1)/2) psi_{k+1} - i sqrt(k/2) psi_{k-1}
    # <psi_k, xi^2 psi_k> = (k + 1/2)
    # <psi_k, xi^2 psi_{k+2}> = ... similar structure
    
    # Let me just compute this numerically using the explicit formula.
    
    # Norm squared:
    # ||g||^2 = sum_k c_k^2 * 2^k * k! * sqrt(pi)
    def fact(k):
        return int(np.prod(np.arange(1, k+1))) if k > 0 else 1
    
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    norm2 = 0.0
    x2_num = 0.0
    xi2_num = 0.0
    
    sqrt_pi = np.sqrt(np.pi)
    
    for k in range(4):
        ck = coeffs[k]
        dk = degrees[k]
        wk = (2.0 ** dk) * fact(dk) * sqrt_pi  # weight for ||g||^2
        
        norm2 += ck * ck * wk
        
        # <psi_dk, x^2 psi_dk> = dk
        x2_num += ck * ck * wk * dk
        
        # Cross terms for x^2: <psi_k, x^2 psi_m> nonzero when |k-m|=2
        # For k=0: <psi_0, x^2 psi_2> = sqrt(1*2/4) = sqrt(0.5)
        # For k=4: <psi_4, x^2 psi_2> = sqrt(4*3/4) = sqrt(3), <psi_4, x^2 psi_6> = sqrt(5*6/4) = sqrt(7.5)
        # For k=8: <psi_8, x^2 psi_6> = sqrt(8*7/4) = sqrt(14), <psi_8, x^2 psi_10> = sqrt(9*10/4) = sqrt(22.5)
        # For k=12: <psi_12, x^2 psi_10> = sqrt(12*11/4) = sqrt(33)
        
        # <xi^2> diagonal: k + 1/2
        xi2_num += ck * ck * wk * (dk + 0.5)
    
    # Cross terms for x^2 (symmetric, so we add both directions):
    # <psi_0, x^2 psi_2>: but psi_2 is not in our basis, so no cross term
    # <psi_4, x^2 psi_2>: psi_2 not in basis
    # <psi_4, x^2 psi_6>: psi_6 not in basis
    # <psi_8, x^2 psi_6>: psi_6 not in basis
    # <psi_8, x^2 psi_10>: psi_10 not in basis
    # <psi_12, x^2 psi_10>: psi_10 not in basis
    # So actually NO cross terms since our degrees are 0,4,8,12 (all differ by 4, not 2)!
    
    # Similarly for xi^2 cross terms: same structure, only differ by 2, so no cross terms.
    
    # Therefore:
    # <x^2> = x2_num / norm2
    # <xi^2> = xi2_num / norm2
    
    if norm2 <= 0:
        return np.inf
    
    x2 = x2_num / norm2
    xi2 = xi2_num / norm2
    
    # C4 = (1/8) * (1 - 4*x2) * (1 - 2*xi2)^2
    # Wait, is it divided by norm2 or not? Let me reconsider.
    # The uncertainty bound is C4 = (1/8)(1 - 4<x^2>)(1-2<xi^2>)^2 where <x^2> and <xi^2> 
    # are expectation values (already normalized by norm2).
    
    c4 = (1.0 / 8.0) * (1.0 - 4.0 * x2) * (1.0 - 2.0 * xi2) ** 2
    return c4


def _c4_bound_grad(a):
    """Analytic gradient of C4 bound."""
    a0, a4, a8 = a
    a12 = -(a0 + a4 + a8)
    
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    def fact(k):
        return int(np.prod(np.arange(1, k+1))) if k > 0 else 1
    
    sqrt_pi = np.sqrt(np.pi)
    
    norm2 = 0.0
    x2_num = 0.0
    xi2_num = 0.0
    
    weights = []
    for k in range(4):
        ck = coeffs[k]
        dk = degrees[k]
        wk = (2.0 ** dk) * fact(dk) * sqrt_pi
        weights.append(wk)
        norm2 += ck * ck * wk
        x2_num += ck * ck * wk * dk
        xi2_num += ck * ck * wk * (dk + 0.5)
    
    if norm2 <= 0:
        return np.zeros(3)
    
    x2 = x2_num / norm2
    xi2 = xi2_num / norm2
    
    c4 = (1.0 / 8.0) * (1.0 - 4.0 * x2) * (1.0 - 2.0 * xi2) ** 2
    
    # dC4/da_j for j in {0,4,8}
    # c4 = (1/8)(1-4x2)(1-2xi2)^2
    # dc4/dx2 = (1/8)(-4)(1-2xi2)^2 = -(1/2)(1-2xi2)^2
    # dc4/dxi2 = (1/8)(1-4x2)*2*(1-2xi2)*(-2) = -(1/2)(1-4x2)(1-2xi2)
    
    dc4_dx2 = -(1.0 / 2.0) * (1.0 - 2.0 * xi2) ** 2
    dc4_dxi2 = -(1.0 / 2.0) * (1.0 - 4.0 * x2) * (1.0 - 2.0 * xi2)
    
    # dx2_num/da_j = 2 * a_j * w_j * d_j  (for the diagonal term)
    # dnorm2/da_j = 2 * a_j * w_j
    # dx2/da_j = (dx2_num/da_j * norm2 - x2_num * dnorm2/da_j) / norm2^2
    #           = (2*a_j*w_j*d_j*norm2 - x2_num*2*a_j*w_j) / norm2^2
    #           = 2*a_j*w_j*(d_j - x2) / norm2
    
    # Similarly dxi2/da_j = 2*a_j*w_j*(d_j+0.5-xi2) / norm2
    
    # But a12 depends on a0,a4,a8: da12/da0=da12/da4=da12/da8=-1
    # So we need to account for the a12 contribution.
    
    grad = np.zeros(3)
    for j in range(3):
        aj = coeffs[j]
        wj = weights[j]
        dj = degrees[j]
        
        # Contribution from coefficient j (direct)
        dx2_da = 2.0 * aj * wj * (dj - x2) / norm2
        dxi2_da = 2.0 * aj * wj * (dj + 0.5 - xi2) / norm2
        
        # Contribution from a12 (indirect, da12/da_j = -1)
        a12_val = a12
        w12 = weights[3]
        d12 = 12
        dx2_da12 = 2.0 * a12_val * w12 * (d12 - x2) / norm2
        dxi2_da12 = 2.0 * a12_val * w12 * (d12 + 0.5 - xi2) / norm2
        
        # Total: da12/da_j = -1
        total_dx2 = dx2_da - dx2_da12
        total_dxi2 = dxi2_da - dxi2_da12
        
        grad[j] = dc4_dx2 * total_dx2 + dc4_dxi2 * total_dxi2
    
    return grad


def construct():
    t_start = time.time()
    DEADLINE = 195.0  # safety margin under the 210 s kill
    
    try:
        # The seed program returns [-113/100, 1/25, 1/3240] = [-1.13, 0.04, 0.000308642]
        # Let's verify this and then try to improve.
        
        seed_a = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        seed_c4 = _c4_bound(seed_a)
        
        best_a = seed_a.copy()
        best_c4 = seed_c4
        
        # Multi-start optimization
        np.random.seed(42)
        
        starts = [seed_a.copy()]
        
        # Add random starts
        for _ in range(200):
            if time.time() - t_start > DEADLINE - 10:
                break
            start = np.random.randn(3) * 2.0
            starts.append(start)
        
        # Add some structured starts
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                start = np.array([scale * np.cos(angle), scale * np.sin(angle), scale * 0.5])
                starts.append(start)
        
        for start in starts:
            if time.time() - t_start > DEADLINE - 5:
                break
            
            # Try L-BFGS-B
            try:
                res = minimize(_c4_bound, start, method='L-BFGS-B', 
                              jac=_c4_bound_grad,
                              options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12})
                if res.success and res.fun < best_c4:
                    best_c4 = res.fun
                    best_a = res.x.copy()
            except Exception:
                pass
            
            # Try Nelder-Mead as backup
            if time.time() - t_start > DEADLINE - 5:
                break
            try:
                res2 = minimize(_c4_bound, start, method='Nelder-Mead',
                               options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14})
                if res2.fun < best_c4:
                    best_c4 = res2.fun
                    best_a = res2.x.copy()
            except Exception:
                pass
        
        # High-precision polish on the best result
        if time.time() - t_start < DEADLINE - 2:
            try:
                res_final = minimize(_c4_bound, best_a, method='L-BFGS-B',
                                   jac=_c4_bound_grad,
                                   options={'maxiter': 50000, 'ftol': 1e-18, 'gtol': 1e-15})
                if res_final.success and res_final.fun < best_c4:
                    best_c4 = res_final.fun
                    best_a = res_final.x.copy()
            except Exception:
                pass
        
        # Final verification
        final_c4 = _c4_bound(best_a)
        if final_c4 < best_c4:
            best_c4 = final_c4
        
        return np.array(best_a)
        
    except Exception:
        # Fallback to the known-good prior
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
