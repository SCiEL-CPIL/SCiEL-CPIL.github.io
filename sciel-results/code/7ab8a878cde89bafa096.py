"""Bootstrap spawn: FAST SEARCH approach – use numerical approximation to explore search space rapidly."""
import time
import math
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START

def _compute_score_fast(z_list):
    """
    Fast numerical approximation of the score B/(2*pi).
    z_list: list of 12 floats, assumed sorted and distinct.
    Returns score or a large penalty if invalid.
    """
    try:
        z = np.array(z_list, dtype=float)
        if np.any(z <= 0) or np.any(np.diff(z) < 1e-8):
            return 1e100
        
        # Pre-compute Laguerre polynomial coefficients L_{2k}^{(-1/2)}(x)
        # L_n^{(alpha)}(x) = sum_{k=0}^n binom(n+alpha, n-k) (-1)^k x^k / k!
        # We need degrees 0, 2, ..., 50.
        # Coeff matrix C[j, k] for x^k in L_{2j}
        C = np.zeros((26, 51))
        alpha = -0.5
        for j in range(26):
            n = 2 * j
            for k in range(n + 1):
                # binom(n+alpha, n-k)
                # Using log-gamma for stability? Or just math.comb? No, alpha is not integer.
                # Use scipy.special.binom
                from scipy.special import binom as sp_binom
                coeff = sp_binom(n + alpha, n - k) * ((-1.0)**k) / math.factorial(k)
                C[j, k] = coeff

        # Build system A c = b
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Row 0: g(0) = 0
        # L_n(0) = binom(n+alpha, n)
        for j in range(26):
            n = 2 * j
            A[0, j] = sp_binom(n + alpha, n)
        b[0] = 0.0
        
        # Row 1: g'(0) = 1
        # L_n'(0) = coeff of x^1 in L_n = binom(n+alpha, n-1) * (-1)^1 / 1!
        for j in range(26):
            n = 2 * j
            if n >= 1:
                A[1, j] = -sp_binom(n + alpha, n - 1)
            else:
                A[1, j] = 0.0
        b[1] = 1.0
        
        # Rows 2..25: g(z_i)=0, g'(z_i)=0
        for i in range(12):
            zi = z[i]
            row_g = 2 + 2 * i
            row_dp = 3 + 2 * i
            
            # Precompute powers of zi
            powers = np.zeros(51)
            powers[0] = 1.0
            for k in range(1, 51):
                powers[k] = powers[k-1] * zi
            
            for j in range(26):
                n = 2 * j
                # g(zi) = sum c_j L_{2j}(zi) = sum c_j sum_k C[j,k] zi^k
                A[row_g, j] = np.dot(C[j, :n+1], powers[:n+1])
                
                # g'(zi) = sum c_j L'_{2j}(zi)
                # L'_{2j}(zi) = sum_{k=1}^n k C[j,k] zi^{k-1}
                # Let m = k-1. sum_{m=0}^{n-1} (m+1) C[j, m+1] zi^m
                if n > 0:
                    coeffs_deriv = np.arange(1, n+1) * C[j, 1:n+1]
                    A[row_dp, j] = np.dot(coeffs_deriv, powers[:n])
                else:
                    A[row_dp, j] = 0.0
            
            b[row_g] = 0.0
            b[row_dp] = 0.0
        
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1e100
            
        # Construct g(x) coefficients
        # g(x) = sum_j c_j L_{2j}(x) = sum_j c_j sum_k C[j,k] x^k
        # g_coeffs[k] = sum_j c_j C[j,k]
        g_coeffs = np.dot(c, C) # shape (51,)
        
        # Construct D(x) = x * prod_i (x - z_i)^2
        # P(x) = prod_i (x - z_i)
        p_coeffs = np.ones(13) # p_12=1, others 0 initially? No, start with [1]
        p_coeffs = np.array([1.0])
        for zi in z:
            # multiply by (x - zi)
            new_p = np.zeros(len(p_coeffs) + 1)
            new_p[1:] += p_coeffs
            new_p[:-1] -= p_coeffs * zi
            p_coeffs = new_p
        
        # P(x)^2
        p2_coeffs = np.convolve(p_coeffs, p_coeffs) # degree 24
        
        # D(x) = x * P(x)^2 => shift by 1
        d_coeffs = np.zeros(26) # degree 25
        d_coeffs[1:] = p2_coeffs
        
        # Polynomial division g / D
        # g has degree 50, D has degree 25. q has degree 25.
        # We want q such that g = q * D + R
        # Since g is constructed to have double roots at z_i, R should be 0.
        # But numerically, it might not be exactly 0. We just take the quotient.
        
        g_work = g_coeffs.copy()
        q_coeffs = np.zeros(25) # degree 24? No, g is deg 50, D is deg 25. q is deg 25.
        # Wait. deg(g)=50, deg(D)=25. deg(q)=25.
        # q_coeffs should have length 26.
        q_coeffs = np.zeros(26)
        
        # Long division
        # Leading coeff of D is 1 (from x * x^24).
        # Leading coeff of g is g_coeffs[50].
        for k in range(25, -1, -1):
            if g_work[50] != 0:
                factor = g_work[50] / d_coeffs[25]
                q_coeffs[k] = factor
                # Subtract factor * D(x) * x^(50-25-k) = factor * D(x) * x^(25-k)
                # D is deg 25. D * x^(25-k) has deg 25 + 25 - k = 50 - k.
                # Shift D coeffs by 25-k
                shift = 25 - k
                g_work[shift:shift+26] -= factor * d_coeffs
        
        # Find real roots of q
        try:
            roots = np.roots(q_coeffs)
        except Exception:
            return 1e100
            
        # Filter real roots
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        if len(real_roots) == 0:
            return 0.0
            
        # Check sign change for each real root
        eps = 1e-12
        max_r = 0
        for r in real_roots:
            q_minus = np.polyval(q_coeffs, r - eps)
            q_plus = np.polyval(q_coeffs, r + eps)
            if q_minus * q_plus < 0:
                if r > max_r:
                    max_r = r
                    
        B = max(0, max_r)
        score = B / (2 * math.pi)
        return score
        
    except Exception:
        return 1e100

def _unconstrained_to_z(u):
    """Map unconstrained positive u to sorted z."""
    z = np.cumsum(u)
    return z

def _objective(u):
    """Objective function for optimization. u are positive increments."""
    z = _unconstrained_to_z(u)
    return _compute_score_fast(z)

def construct():
    """Return 12 double-root locations using FAST SEARCH."""
    deadline = time.time() + 580.0  # 630s kill, keep 50s margin

    # Initial best: Bessel zeros squared
    def _bessel_zero_J0(k):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            return 2.4048255576957728
        return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

    z_baseline = np.array([_bessel_zero_J0(i+1)**2 for i in range(12)])
    best_z = z_baseline.copy()
    best_score = _compute_score_fast(best_z)

    # Differential Evolution
    # Bounds for u_i: [1e-3, 100]
    bounds = [(1e-3, 100.0)] * 12
    
    # We need to run DE for a limited time
    # popsize=10, maxiter=50 -> ~6000 evals. Should take < 10s.
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=50, 
            popsize=10,
            tol=1e-6,
            seed=42,
            polish=False # We will polish manually
        )
        
        if res.success or res.fun < best_score:
            u_best = res.x
            z_candidate = _unconstrained_to_z(u_best)
            score_candidate = _compute_score_fast(z_candidate)
            if score_candidate < best_score:
                best_z = z_candidate
                best_score = score_candidate
    except Exception:
        pass

    # Local Polish using Nelder-Mead on the fast score
    if time.time() < deadline - 60:
        try:
            # Start from best_u
            u_start = np.diff(np.concatenate(([0], best_z)))
            # Ensure u_start is within bounds
            u_start = np.clip(u_start, 1e-3, 100.0)
            
            res_polish = minimize(
                _objective, 
                u_start, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res_polish.fun < best_score:
                u_polished = res_polish.x
                z_polished = _unconstrained_to_z(u_polished)
                score_polished = _compute_score_fast(z_polished)
                if score_polished < best_score:
                    best_z = z_polished
                    best_score = score_polished
        except Exception:
            pass

    # Final check: ensure z is valid
    if np.any(best_z <= 0) or np.any(np.diff(best_z) < 1e-8):
        return z_baseline.tolist()
        
    return best_z.tolist()

# EVOLVE-BLOCK-END
