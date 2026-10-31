"""Analytic uncertainty bound with global search on normalized coefficient sphere."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
import math

_DEADLINE = time.time() + 195

# Precompute I_n = integral x^{2n} e^{-x^2} dx
def _precompute_I():
    """Precompute I_n for n=0..13"""
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty bound for coefficients a0, a4, a8.
    a12 is fixed such that g(0)=0.
    """
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    # p_k is the coefficient of x^{2k} in P(x)
    # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    
    p = np.zeros(7)
    # k=0 (x^0)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    # k=1 (x^2)
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    # k=2 (x^4)
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    # k=3 (x^6)
    p[3] = -28.0*a8 - 6930.0*a12
    # k=4 (x^8)
    p[4] = a8 + 1155.0*a12
    # k=5 (x^10)
    p[5] = -66.0*a12
    # k=6 (x^12)
    p[6] = a12
    
    I = _I
    
    # Calculate moments m_k = integral x^{2k} P(x)^2 e^{-x^2} dx
    # P(x)^2 = sum_{i,j} p_i p_j x^{2(i+j)}
    # m_k = sum_{i,j} p_i p_j I[k+i+j]
    
    m0 = 0.0
    m2 = 0.0
    # m4 is not needed for variance, but let's stick to m0, m2
    # Actually, variance is <x^2> - <x>^2. Since P(x) is even, <x> = 0.
    # So v_f = m2/m0.
    
    for i in range(7):
        pi = p[i]
        if pi == 0: continue
        for j in range(7):
            pp = pi * p[j]
            if pp == 0: continue
            idx0 = i + j
            if idx0 < 14:
                m0 += pp * I[idx0]
            idx2 = 2 + i + j
            if idx2 < 14:
                m2 += pp * I[idx2]
            
    if m0 <= 0 or not np.isfinite(m0):
        return 1e18
        
    vf = m2 / m0
    
    # Fourier transform part
    # The Fourier transform of x^{2k} e^{-x^2/2} involves Hermite polynomials again.
    # Specifically, if g(x) = P(x) e^{-x^2/2}, then \hat{g}(\xi) = (2\pi)^{1/2} P(-i\xi) e^{-\xi^2/2} ? 
    # Let's check the property.
    # Fourier transform of e^{-x^2/2} is sqrt(2\pi) e^{-\xi^2/2}.
    # Multiplication by x^2 corresponds to -d^2/d\xi^2 in Fourier space.
    # However, it is easier to use the relation with Hermite polynomials.
    # The eigenfunctions of the Fourier transform are Hermite functions.
    # H_n(x) e^{-x^2/2} transforms to (-i)^n H_n(\xi) e^{-\xi^2/2}.
    # Since P(x) is a combination of H_{2k}(x), let's express P in terms of H_{2k}.
    # Wait, the basis is H_0, H_4, H_8, H_12. These are even Hermite polynomials.
    # Let P(x) = c_0 H_0(x) + c_4 H_4(x) + c_8 H_8(x) + c_{12} H_{12}(x).
    # Then \hat{g}(\xi) = \sqrt{2\pi} [ c_0 H_0(\xi) + c_4 (-i)^4 H_4(\xi) + c_8 (-i)^8 H_8(\xi) + c_{12} (-i)^{12} H_{12}(\xi) ] e^{-\xi^2/2}
    # (-i)^4 = 1, (-i)^8 = 1, (-i)^{12} = 1.
    # So \hat{g}(\xi) = \sqrt{2\pi} P(\xi) e^{-\xi^2/2}.
    # This implies the Fourier transform just scales the function by sqrt(2\pi) and keeps the polynomial part P unchanged (for even degrees)?
    # Let's verify.
    # FT of H_n(x) e^{-x^2/2} is (-i)^n H_n(\xi) e^{-\xi^2/2}.
    # For n=4, (-i)^4 = 1.
    # For n=8, (-i)^8 = 1.
    # For n=12, (-i)^{12} = 1.
    # So yes, for even n, the sign is +1.
    # Thus, \hat{g}(\xi) = \sqrt{2\pi} P(\xi) e^{-\xi^2/2}.
    
    # Therefore, the integrals for the Fourier side are:
    # \int \hat{g}^2 d\xi = 2\pi \int P(\xi)^2 e^{-\xi^2/2} d\xi = 2\pi m0
    # \int \xi^2 \hat{g}^2 d\xi = 2\pi \int \xi^2 P(\xi)^2 e^{-\xi^2/2} d\xi = 2\pi m2
    
    # Wait, the definition of I_n used e^{-x^2}, not e^{-x^2/2}.
    # Let's re-evaluate the integrals with the correct Gaussian weight.
    # g(x) = P(x) e^{-x^2/2}.
    # g(x)^2 = P(x)^2 e^{-x^2}.
    # So m_k = \int x^{2k} P(x)^2 e^{-x^2} dx.
    # This matches the I_n definition: I_n = \int x^{2n} e^{-x^2} dx.
    # So m0 and m2 calculated above are correct for the position space.
    
    # For the Fourier space:
    # \hat{g}(\xi) = \sqrt{2\pi} P(\xi) e^{-\xi^2/2}.
    # \hat{g}(\xi)^2 = 2\pi P(\xi)^2 e^{-\xi^2}.
    # \int \hat{g}^2 d\xi = 2\pi \int P(\xi)^2 e^{-\xi^2} d\xi = 2\pi m0.
    # \int \xi^2 \hat{g}^2 d\xi = 2\pi \int \xi^2 P(\xi)^2 e^{-\xi^2} d\xi = 2\pi m2.
    
    # So n0 = 2\pi m0 and n2 = 2\pi m2.
    # vF = n2 / n0 = m2 / m0 = vf.
    # This implies vf = vF always?
    # If vf = vF, then the bound is vf.
    # And minimizing vf*vF is minimizing vf^2.
    # This seems too simple. Let's double check the Fourier transform property.
    
    # Standard Fourier Transform: \hat{f}(\xi) = \int f(x) e^{-i x \xi} dx.
    # For Hermite functions \psi_n(x) = H_n(x) e^{-x^2/2}, \hat{\psi}_n(\xi) = (-i)^n \psi_n(\xi).
    # This is correct.
    # So \hat{g}(\xi) = \sqrt{2\pi} \sum c_k (-i)^{2k} H_{2k}(\xi) e^{-\xi^2/2} = \sqrt{2\pi} \sum c_k H_{2k}(\xi) e^{-\xi^2/2} = \sqrt{2\pi} P(\xi) e^{-\xi^2/2}.
    # Yes, the polynomial part is the same.
    # So the variances are identical.
    # Then the problem reduces to minimizing $v_f = \frac{\int x^2 P(x)^2 e^{-x^2} dx}{\int P(x)^2 e^{-x^2} dx}$.
    
    # Wait, is the seed program's bound different?
    # The seed program calculates n_k with a 2*pi factor and alternating signs?
    # "For Fourier transform: Q(xi) = sum q_k xi^{2k} where q_k = p_k * (-1)^k"
    # Why (-1)^k?
    # Maybe the definition of Hermite polynomials or Fourier transform convention differs?
    # Or maybe I am missing a detail about the "Fourier-invariant" part.
    # Let's trust the seed program's logic for the Fourier side, as it's the "current" implementation.
    # The seed program uses:
    # q_k = p_k * (-1)^k
    # n_k = 2*pi * sum q_i q_j I[k+i+j]
    
    # Let's re-implement the seed's logic exactly to ensure we are minimizing the same quantity.
    
    q = np.array([p[k] * ((-1.0)**k) for k in range(7)])
    
    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0: continue
        for j in range(7):
            qq = qi * q[j]
            if qq == 0: continue
            idx0 = i + j
            if idx0 < 14:
                n0 += 2.0 * np.pi * qq * I[idx0]
            idx2 = 2 + i + j
            if idx2 < 14:
                n2 += 2.0 * np.pi * qq * I[idx2]
                
    if n0 <= 0 or not np.isfinite(n0):
        return 1e18
        
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
        
    return np.sqrt(vf * vF)

def _objective_angles(theta, phi):
    """
    Objective function on the unit sphere.
    theta in [0, pi], phi in [0, 2*pi)
    """
    a0 = np.sin(theta) * np.cos(phi)
    a4 = np.sin(theta) * np.sin(phi)
    a8 = np.cos(theta)
    return _compute_bound(a0, a4, a8)

def construct():
    best_val = 1e18
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # 1. Grid search on the sphere
    # We use a coarse grid to find promising regions
    n_theta = 50
    n_phi = 100
    
    # We can restrict the search space. 
    # Due to symmetry, we can likely restrict phi to [0, pi] and theta to [0, pi/2] if we allow sign flips?
    # a0, a4, a8 can be negative.
    # The function is even in each coefficient?
    # P(x) depends on a0, a4, a8.
    # If we flip the sign of all coefficients, P(x) -> -P(x).
    # P(x)^2 is unchanged. So the bound is unchanged.
    # So we can fix the sign of one coefficient, say a0 >= 0.
    # This reduces the sphere to a hemisphere.
    # theta in [0, pi/2] (since a8 can be negative, we need full theta range? 
    # If a0 >= 0, then sin(theta)cos(phi) >= 0.
    # Let's just do a full grid for safety, it's fast enough.
    
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    candidates = []
    
    for t in thetas:
        for p in phis:
            val = _objective_angles(t, p)
            if val < best_val:
                best_val = val
                a0 = np.sin(t) * np.cos(p)
                a4 = np.sin(t) * np.sin(p)
                a8 = np.cos(t)
                best_params = np.array([a0, a4, a8])
                candidates.append((t, p, val))
            if time.time() > _DEADLINE - 20:
                return best_params

    # 2. Refine local minima
    # Sort candidates by value and take the top few
    candidates.sort(key=lambda x: x[2])
    top_candidates = candidates[:20]
    
    for t, p, _ in top_candidates:
        if time.time() > _DEADLINE - 10:
            break
            
        # Use L-BFGS-B or Nelder-Mead to refine
        # Nelder-Mead is derivative-free and robust for 2D
        res = minimize(_objective_angles, np.array([t, p]), method='Nelder-Mead', 
                       options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 2000})
        
        if res.fun < best_val and np.isfinite(res.fun):
            best_val = res.fun
            t_opt, p_opt = res.x
            a0 = np.sin(t_opt) * np.cos(p_opt)
            a4 = np.sin(t_opt) * np.sin(p_opt)
            a8 = np.cos(t_opt)
            best_params = np.array([a0, a4, a8])
            
    # 3. Final polish with high precision
    if best_val < 1e17:
        # Convert back to angles for polish? Or just use the 3D params?
        # The 3D params are on the sphere.
        # Let's do a final Nelder-Mead on the angles again with tighter tol
        # We need to recover angles from best_params
        norm = np.linalg.norm(best_params)
        if norm > 1e-10:
            a0, a4, a8 = best_params / norm
            # Recover angles
            # a0 = sin(t) cos(p), a4 = sin(t) sin(p), a8 = cos(t)
            # sin(t) = sqrt(a0^2 + a4^2)
            sin_t = np.sqrt(a0**2 + a4**2)
            if sin_t > 1e-10:
                t_init = np.arccos(a8)
                p_init = np.arctan2(a4, a0)
            else:
                t_init = 0.0
                p_init = 0.0
                
            res = minimize(_objective_angles, np.array([t_init, p_init]), method='Nelder-Mead', 
                           options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 5000})
            
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                t_opt, p_opt = res.x
                a0 = np.sin(t_opt) * np.cos(p_opt)
                a4 = np.sin(t_opt) * np.sin(p_opt)
                a8 = np.cos(t_opt)
                best_params = np.array([a0, a4, a8])

    return best_params
# EVOLVE-BLOCK-END
