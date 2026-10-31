"""
Structural change: Use exact analytic Hermite-Gaussian integrals and solve the
stationary point equations (gradient = 0) using a robust hybrid root-finder
(LM/Hybr) instead of gradient descent. This avoids local minima traps inherent
to descent methods and directly targets critical points.
"""
import time
import numpy as np
from scipy.optimize import root

T_START = time.time()
DEADLINE = 205.0

def _time_left():
    return DEADLINE - (time.time() - T_START)

def _precompute_exact():
    """
    Build 4x4 matrices M, P, F analytically for basis {H0, H4, H8, H12} * phi.
    phi = exp(-x^2/2).
    M_mn = <H_m phi, H_n phi> = sqrt(pi) 2^n n! delta_mn (if m=n else 0)
    P_mn = <x^2 H_m phi, x^2 H_n phi>
    F_mn = <(H_m phi)'', (H_n phi)''>
    
    We use the recurrence: x H_n = 0.5 H_{n+1} + n H_{n-1}.
    x^2 H_n = 0.25 H_{n+2} + (n + 0.5) H_n + 0.25 n(n-1) H_{n-2}.
    
    Let A_n be the vector of coefficients for x^2 H_n in the Hermite basis.
    A_n[k] is non-zero only for k = n+2, n, n-2.
    A_n[n+2] = 0.25
    A_n[n] = n + 0.5
    A_n[n-2] = 0.25 n(n-1)
    
    P_mn = sum_k A_m[k] A_n[k] <H_k phi, H_k phi>
         = sum_k A_m[k] A_n[k] sqrt(pi) 2^k k!
    
    For F: (H_n phi)'' = (x^2 - 2n - 1) H_n phi.
    Let B_n be coefficients for (x^2 - 2n - 1) H_n.
    B_n = A_n - (2n + 1) * e_n (where e_n is unit vector at n).
    B_n[n+2] = 0.25
    B_n[n] = (n + 0.5) - (2n + 1) = -n - 0.5
    B_n[n-2] = 0.25 n(n-1)
    
    F_mn = sum_k B_m[k] B_n[k] sqrt(pi) 2^k k!
    """
    indices = [0, 4, 8, 12]
    N = len(indices)
    M = np.zeros((N, N))
    P = np.zeros((N, N))
    F = np.zeros((N, N))
    
    # Precompute factorials and powers of 2 up to 14
    max_n = 14
    fact = [1.0] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i
    pow2 = [1.0] * (max_n + 1)
    for i in range(1, max_n + 1):
        pow2[i] = pow2[i-1] * 2
    
    sqrt_pi = np.sqrt(np.pi)
    
    # Coefficients for x^2 H_n and (x^2 - 2n - 1) H_n
    # We only need n in {0, 4, 8, 12}, but the expansion goes to n+2.
    # So we need A_n for n up to 12.
    
    def get_coeffs(n):
        """Return dict {k: coeff} for x^2 H_n and B_n."""
        A = {}
        B = {}
        if n >= 0:
            A[n] = n + 0.5
            B[n] = (n + 0.5) - (2*n + 1) # -n - 0.5
        if n + 2 <= max_n:
            A[n+2] = 0.25
            B[n+2] = 0.25
        if n - 2 >= 0:
            A[n-2] = 0.25 * n * (n-1)
            B[n-2] = 0.25 * n * (n-1)
        return A, B
    
    coeff_cache = {}
    for n in indices:
        coeff_cache[n] = get_coeffs(n)
    
    for i, ni in enumerate(indices):
        # M
        M[i, i] = sqrt_pi * pow2[ni] * fact[ni]
        
        Ai, Bi = coeff_cache[ni]
        
        for j, nj in enumerate(indices):
            if i == j:
                continue
            Aj, Bj = coeff_cache[nj]
            
            # P: sum over k where Ai and Aj overlap
            p_val = 0.0
            f_val = 0.0
            # Keys can be n-2, n, n+2
            # Intersection of keys
            keys = set(Ai.keys()).intersection(Aj.keys())
            for k in keys:
                w = sqrt_pi * pow2[k] * fact[k]
                p_val += Ai[k] * Aj[k] * w
                f_val += Bi[k] * Bj[k] * w
            
            P[i, j] = p_val
            F[i, j] = f_val
            
            # Symmetry
            P[j, i] = p_val
            F[j, i] = f_val
            
    # Values at x=0
    # H0(0)=1, H4(0)=12, H8(0)=5040, H12(0)=4790016
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 5040.0
    h12_0 = 4790016.0
    
    return {
        'M': M, 'P': P, 'F': F,
        'h0_0': h0_0, 'h4_0': h4_0, 'h8_0': h8_0, 'h12_0': h12_0
    }

def _objective_and_grad(params, pre):
    """Compute objective and gradient for (c0, c4, c8)."""
    if _time_left() < 2.0:
        return 1e10, np.zeros(3)
    
    c0, c4, c8 = params
    h0_0 = pre['h0_0']
    h4_0 = pre['h4_0']
    h8_0 = pre['h8_0']
    h12_0 = pre['h12_0']
    
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    u = np.array([c0, c4, c8, c12])
    
    M = pre['M']
    P = pre['P']
    F = pre['F']
    
    n_sq = u @ M @ u
    if n_sq < 1e-15:
        return 1e10, np.zeros(3)
    
    pos_sq = (u @ P @ u) / n_sq
    freq_sq = (u @ F @ u) / n_sq
    
    if pos_sq < 0 or freq_sq < 0:
        return 1e10, np.zeros(3)
    
    pos = np.sqrt(pos_sq)
    freq = np.sqrt(freq_sq)
    obj = pos * freq
    
    # Gradient
    Pu = P @ u
    Fu = F @ u
    Mu = M @ u
    
    dpos_sq = (2.0 / n_sq) * (Pu - (u @ Pu / n_sq) * Mu)
    dfreq_sq = (2.0 / n_sq) * (Fu - (u @ Fu / n_sq) * Mu)
    
    if pos > 1e-15 and freq > 1e-15:
        dpos = dpos_sq / (2.0 * pos)
        dfreq = dfreq_sq / (2.0 * freq)
        grad_full = freq * dpos + pos * dfreq
    else:
        grad_full = np.zeros(4)
    
    dc12 = np.array([-h0_0 / h12_0, -h4_0 / h12_0, -h8_0 / h12_0])
    grad = grad_full[:3] + grad_full[3] * dc12
    
    return obj, grad

def _stationary_equations(params, pre):
    """Return gradient of the objective. We solve grad = 0."""
    _, grad = _objective_and_grad(params, pre)
    return grad

def construct():
    """Find optimal coefficients (c0, c4, c8) minimizing the uncertainty bound."""
    if _time_left() < 10.0:
        return np.array([-1.13, 0.04, 1.0 / 3240.0])
    
    pre = _precompute_exact()
    
    best_val = np.inf
    best_x = np.array([-1.13, 0.04, 1.0 / 3240.0])
    
    # Multi-start root finding for stationary points
    starts = [
        np.array([-1.13, 0.04, 1.0 / 3240.0]),
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.5, 0.03, 0.0005]),
        np.array([-0.8, 0.06, 0.0002]),
        np.array([-1.2, 0.035, 0.00035]),
        np.array([-2.0, 0.1, 0.001]),
        np.array([-0.5, 0.1, 0.001]),
        np.array([-1.1, 0.045, 0.00031]),
        np.array([-1.15, 0.038, 0.00029]),
        np.array([-1.3, 0.02, 0.0006]),
        np.array([-0.9, 0.07, 0.00015]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-1.13, 0.04, 0.0]),
        np.array([-1.0, 0.04, 1.0 / 3240.0]),
        np.array([-1.13, 0.0, 1.0 / 3240.0]),
        np.array([-1.0, 0.04, 0.0]),
        np.array([-1.2, 0.05, 0.0]),
        np.array([-0.9, 0.05, 0.0]),
    ]
    
    for start in starts:
        if _time_left() < 5.0:
            break
        try:
            # Use 'lm' (Levenberg-Marquardt) which is robust for small systems
            # and doesn't require bounds. 'hybr' is also good.
            res = root(
                lambda p: _stationary_equations(p, pre),
                start,
                method='lm',
                tol=1e-12
            )
            if res.success:
                val, _ = _objective_and_grad(res.x, pre)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Polish best solution with L-BFGS-B to ensure local minimum
    if _time_left() > 5.0 and np.isfinite(best_val):
        try:
            from scipy.optimize import minimize
            res = minimize(
                lambda p: _objective_and_grad(p, pre),
                best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
            
    return best_x

if __name__ == "__main__":
    result = construct()
    print("Coefficients (c0, c4, c8):", result)
