"""
Optimized uncertainty bound for Hermite-Gaussian test function.
Uses exact Hermite-Gaussian Fourier transform properties to express
frequency moments in terms of position moments, reducing to a
generalized eigenvalue problem on small moment matrices.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

T_START = time.time()
DEADLINE = 205.0  # safety margin from 210s hard limit


def _time_left():
    return DEADLINE - (time.time() - T_START)


def _hermite_coeffs(n):
    """Return coefficients of the (physicists') Hermite polynomial H_n(x) = sum c_k x^k.
    H_0 = 1, H_1 = 2x, H_n = 2x H_{n-1} - 2(n-1) H_{n-2}
    Returns list [c_0, c_1, ..., c_n] where H_n(x) = c_0 + c_1*x + ... + c_n*x^n
    """
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    # Build up
    c_prev2 = [1.0]  # H_0
    c_prev1 = [0.0, 2.0]  # H_1
    for k in range(2, n + 1):
        # H_k = 2x * H_{k-1} - 2(k-1) * H_{k-2}
        c_new = [0.0] * (k + 1)
        # 2x * H_{k-1}: shift c_prev1 up by 1, multiply by 2
        for i in range(len(c_prev1)):
            c_new[i + 1] += 2.0 * c_prev1[i]
        # -2(k-1) * H_{k-2}
        for i in range(len(c_prev2)):
            c_new[i] -= 2.0 * (k - 1) * c_prev2[i]
        c_prev2 = c_prev1
        c_prev1 = c_new
    return c_prev1


def _poly_eval(coeffs, x):
    """Evaluate polynomial with given coefficients at x (array)."""
    result = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        result += c * x**i
    return result


def _hermite_gaussian_moments():
    """
    Compute exact moments for Hermite-Gaussian functions.
    
    For g(x) = sum_j c_j H_{n_j}(x) * exp(-x^2/2), with n_j in {0,4,8,12}:
    
    We need:
    - ||g||^2 = sum_{j,k} c_j c_k <H_{n_j} phi, H_{n_k} phi>
    - <x^2 g, x^2 g> (position 4th moment related)
    - <g'', g''> (frequency 4th moment related)
    
    Key identity: For Hermite-Gaussian phi_n = H_n(x) e^{-x^2/2} / sqrt(2^n n! sqrt(pi)),
    the Fourier transform is i^n phi_n (up to convention).
    
    But we're using unnormalized H_n * phi where phi = e^{-x^2/2}.
    
    <H_m phi, H_n phi> = 2^n n! sqrt(pi) * delta_{mn} (orthogonality)
    
    For x^2: x^2 H_n = (1/2) H_{n+2} + n H_n + (n/2) H_{n-2} (for n>=2)
    Actually: x^2 H_n = (1/2) H_{n+2} + n H_n + (n(n-1)/2) * ... let me use recurrence.
    
    H_{n+1} = 2x H_n - 2n H_{n-1}
    => x H_n = (1/2) H_{n+1} + n H_{n-1}
    => x^2 H_n = x * [(1/2) H_{n+1} + n H_{n-1}]
    = (1/2) x H_{n+1} + n x H_{n-1}
    = (1/2)[(1/2) H_{n+2} + (n+1) H_n] + n[(1/2) H_n + (n-1) H_{n-2}]
    = (1/4) H_{n+2} + [(n+1)/2 + n/2] H_n + n(n-1)/2 H_{n-2}
    = (1/4) H_{n+2} + (2n+1)/2 H_n + n(n-1)/2 H_{n-2}
    
    For the second derivative:
    H_n'' = 4n(n-1) H_{n-2} (from H_n' = 2n H_{n-1}, H_n'' = 2n * 2(n-1) H_{n-2})
    
    So (H_n phi)'' = H_n'' phi + 2 H_n' phi' + H_n phi''
    phi = e^{-x^2/2}, phi' = -x phi, phi'' = (x^2 - 1) phi
    H_n' = 2n H_{n-1}
    H_n'' = 4n(n-1) H_{n-2}
    
    (H_n phi)'' = [4n(n-1) H_{n-2} - 4n x H_{n-1} + (x^2 - 1) H_n] phi
    
    Now x H_{n-1} = (1/2) H_n + (n-1) H_{n-2}
    x^2 H_n = (1/4) H_{n+2} + (2n+1)/2 H_n + n(n-1)/2 H_{n-2}
    
    So (H_n phi)'' = [4n(n-1) H_{n-2} - 4n((1/2) H_n + (n-1) H_{n-2}) + (x^2-1) H_n] phi
    = [4n(n-1) H_{n-2} - 2n H_n - 4n(n-1) H_{n-2} + (x^2-1) H_n] phi
    = [-2n H_n + (x^2-1) H_n] phi
    = [(x^2 - 2n - 1) H_n] phi
    
    Wait, let me recheck:
    (H_n phi)'' = H_n'' phi + 2 H_n' phi' + H_n phi''
    = 4n(n-1) H_{n-2} phi + 2(2n H_{n-1})(-x phi) + H_n (x^2 - 1) phi
    = [4n(n-1) H_{n-2} - 4nx H_{n-1} + (x^2-1) H_n] phi
    
    Now -4nx H_{n-1} = -4n * [(1/2) H_n + (n-1) H_{n-2}] = -2n H_n - 4n(n-1) H_{n-2}
    
    So: 4n(n-1) H_{n-2} - 2n H_n - 4n(n-1) H_{n-2} + (x^2-1) H_n
    = -2n H_n + (x^2-1) H_n
    = (x^2 - 2n - 1) H_n
    
    Great! So (H_n phi)'' = (x^2 - 2n - 1) H_n phi
    
    This means <(H_n phi)'', (H_m phi)''> = <(x^2 - 2n - 1) H_n phi, (x^2 - 2m - 1) H_m phi>
    
    For n=m: <(x^2 - 2n - 1)^2 H_n^2 phi^2, 1> ... this is getting complex.
    
    Actually, since (H_n phi)'' = (x^2 - 2n - 1) H_n phi, and H_n phi is orthogonal to H_m phi for n≠m,
    we can write:
    
    Let psi_n = H_n phi. Then psi_n'' = (x^2 - 2n - 1) psi_n.
    
    <psi_n'', psi_m''> = <(x^2-2n-1) psi_n, (x^2-2m-1) psi_m>
    
    For n=m: <(x^2-2n-1)^2 psi_n^2>
    For n≠m: (2n+1)(2m+1) <psi_n, psi_m> + <x^2 psi_n, x^2 psi_m> + ... 
    
    Actually this is still complex. Let me use a different approach.
    
    Since we only have 4 basis functions (n = 0, 4, 8, 12), and the constraint fixes c12 in terms of c0,c4,c8,
    we have 3 free parameters. Let me just compute the integrals numerically with high precision
    but use the analytic structure to get exact values.
    
    Actually, let me compute everything analytically using the orthogonality and recurrence relations.
    """
    degrees = [0, 4, 8, 12]
    N = len(degrees)
    
    # Norms: ||H_n phi||^2 = 2^n n! sqrt(pi)
    import math
    norms = {}
    for n in degrees:
        norms[n] = 2.0**n * math.factorial(n) * math.sqrt(math.pi)
    
    # x^2 H_n expansion: x^2 H_n = (1/4) H_{n+2} + (2n+1)/2 H_n + n(n-1)/2 H_{n-2}
    # (for n >= 2; for n=0: x^2 H_0 = x^2 = (1/4) H_2 + (1/2) H_0)
    # (for n=1: x^2 H_1 = 2x^3 = 2[(1/4)H_3 + (3/2)H_1] = (1/2)H_3 + 3H_1)
    
    def x2_hermite_coeffs(n):
        """Return dict mapping degree -> coefficient for x^2 H_n expressed in Hermite basis."""
        result = {}
        if n >= 2:
            result[n+2] = 0.25
            result[n] = (2*n+1)/2.0
            result[n-2] = n*(n-1)/2.0
        elif n == 1:
            result[3] = 0.5
            result[1] = 3.0
        elif n == 0:
            result[2] = 0.25
            result[0] = 0.5
        return result
    
    # Second derivative: (H_n phi)'' = (x^2 - 2n - 1) H_n phi
    # So (H_n phi)'' = x^2 H_n phi - (2n+1) H_n phi
    # = [sum_k a_k H_k] phi - (2n+1) H_n phi
    # where sum_k a_k H_k = x^2 H_n
    
    def d2_hermite_coeffs(n):
        """Return dict mapping degree -> coefficient for (H_n phi)'' expressed as sum c_k H_k phi."""
        x2_coeffs = x2_hermite_coeffs(n)
        result = dict(x2_coeffs)
        result[n] = result.get(n, 0.0) - (2*n + 1)
        # Remove zero entries
        return {k: v for k, v in result.items() if abs(v) > 1e-15}
    
    # Now compute Gram matrices
    # M[i,j] = <psi_{n_i}, psi_{n_j}> = norms[n_i] * delta_{ij}
    M = np.zeros((N, N))
    for i, ni in enumerate(degrees):
        M[i, i] = norms[ni]
    
    # P[i,j] = <x^2 psi_{n_i}, x^2 psi_{n_j}>
    # x^2 psi_n = sum_k a_{n,k} psi_k where a from x2_hermite_coeffs
    # So <x^2 psi_i, x^2 psi_j> = sum_k sum_l a_{i,k} a_{j,l} <psi_k, psi_l>
    # = sum_k a_{i,k} a_{j,k} norms[k] (by orthogonality)
    
    P = np.zeros((N, N))
    for i, ni in enumerate(degrees):
        ci = x2_hermite_coeffs(ni)
        for j, nj in enumerate(degrees):
            cj = x2_hermite_coeffs(nj)
            val = 0.0
            for k in set(list(ci.keys()) + list(cj.keys())):
                ai = ci.get(k, 0.0)
                aj = cj.get(k, 0.0)
                if abs(ai) > 1e-15 and abs(aj) > 1e-15:
                    # Need norm of H_k phi
                    val += ai * aj * 2.0**k * math.factorial(k) * math.sqrt(math.pi)
            P[i, j] = val
    
    # F[i,j] = <psi_i'', psi_j''>
    # psi_i'' = sum_k b_{i,k} psi_k where b from d2_hermite_coeffs
    F = np.zeros((N, N))
    for i, ni in enumerate(degrees):
        bi = d2_hermite_coeffs(ni)
        for j, nj in enumerate(degrees):
            bj = d2_hermite_coeffs(nj)
            val = 0.0
            for k in set(list(bi.keys()) + list(bj.keys())):
                ai = bi.get(k, 0.0)
                aj = bj.get(k, 0.0)
                if abs(ai) > 1e-15 and abs(aj) > 1e-15:
                    val += ai * aj * 2.0**k * math.factorial(k) * math.sqrt(math.pi)
            F[i, j] = val
    
    # Values at x=0: H_n(0)
    # H_0(0) = 1, H_4(0) = 12, H_8(0) = -1344, H_12(0) = 408408? Let me compute.
    # H_n(0) for even n = 2n: (-1)^n * (2n)! / n!
    # H_0(0) = 1
    # H_4(0) = (-1)^2 * 4! / 2! = 24/2 = 12
    # H_8(0) = (-1)^4 * 8! / 4! = 40320/24 = 1680... wait let me recalculate
    # H_8(0): using recurrence H_n(0) = -2(n-1) H_{n-2}(0)
    # H_0(0)=1, H_2(0)=-2*1*1=-2, H_4(0)=-2*3*(-2)=12, H_6(0)=-2*5*12=-120
    # H_8(0)=-2*7*(-120)=1680, H_10(0)=-2*9*1680=-30240, H_12(0)=-2*11*(-30240)=665280
    
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 665280.0
    
    return {
        'M': M,
        'P': P,
        'F': F,
        'h0_0': h0_0,
        'h4_0': h4_0,
        'h8_0': h8_0,
        'h12_0': h12_0,
        'degrees': degrees
    }


def _objective_and_grad(params, pre):
    """Compute objective value and gradient for given (c0, c4, c8)."""
    if _time_left() < 3.0:
        return 1e10, np.zeros(3)
    
    c0, c4, c8 = params
    h0_0 = pre['h0_0']
    h4_0 = pre['h4_0']
    h8_0 = pre['h8_0']
    h12_0 = pre['h12_0']
    
    # c12 from constraint: c0*h0_0 + c4*h4_0 + c8*h8_0 + c12*h12_0 = 0
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    
    coeffs = np.array([c0, c4, c8, c12])
    M = pre['M']
    P = pre['P']
    F = pre['F']
    
    # Normalization: ||g||^2 = coeffs^T M coeffs
    norm_sq = coeffs @ M @ coeffs
    if norm_sq < 1e-15:
        return 1e10, np.zeros(3)
    
    # Position 4th moment: (coeffs^T P coeffs) / (coeffs^T M coeffs)
    pos_sq = (coeffs @ P @ coeffs) / norm_sq
    # Frequency 4th moment: (coeffs^T F coeffs) / (coeffs^T M coeffs)
    freq_sq = (coeffs @ F @ coeffs) / norm_sq
    
    pos_sq = max(pos_sq, 0.0)
    freq_sq = max(freq_sq, 0.0)
    
    pos = np.sqrt(pos_sq)
    freq = np.sqrt(freq_sq)
    
    obj = pos * freq
    
    # Gradient computation using chain rule
    # obj = sqrt(P_c) * sqrt(F_c) where P_c = c^T P c / c^T M c, F_c = c^T F c / c^T M c
    # d(obj)/dc = freq * d(sqrt(P_c))/dc + pos * d(sqrt(F_c))/dc
    # d(sqrt(P_c))/dc = (1/(2*sqrt(P_c))) * d(P_c)/dc
    # d(P_c)/dc = 2(Pc)/(c^T M c) - 2(c^T P c)(M c)/(c^T M c)^2 = 2/(c^T M c) * (P c - P_c M c)
    
    u = coeffs
    n_sq = u @ M @ u
    Pu = P @ u
    Mu = M @ u
    Fu = F @ u
    
    P_c = u @ Pu / n_sq
    F_c = u @ Fu / n_sq
    
    dP_dc = (2.0 / n_sq) * (Pu - P_c * Mu)
    dF_dc = (2.0 / n_sq) * (Fu - F_c * Mu)
    
    if pos > 1e-15 and freq > 1e-15:
        dpos_dc = dP_dc / (2.0 * pos)
        dfreq_dc = dF_dc / (2.0 * freq)
        grad_full = freq * dpos_dc + pos * dfreq_dc
    else:
        grad_full = np.zeros(4)
    
    # Project gradient to (c0, c4, c8)
    dc12_dc0 = -h0_0 / h12_0
    dc12_dc4 = -h4_0 / h12_0
    dc12_dc8 = -h8_0 / h12_0
    
    grad0 = grad_full[0] + grad_full[3] * dc12_dc0
    grad4 = grad_full[1] + grad_full[3] * dc12_dc4
    grad8 = grad_full[2] + grad_full[3] * dc12_dc8
    
    grad = np.array([grad0, grad4, grad8])
    
    return obj, grad


def _objective(params, pre):
    val, _ = _objective_and_grad(params, pre)
    return val


def _objective_grad(params, pre):
    val, grad = _objective_and_grad(params, pre)
    return val, grad


def construct():
    """Find optimal coefficients (c0, c4, c8) minimizing the uncertainty bound."""
    if _time_left() < 15.0:
        return np.array([-1.13, 0.04, 1.0/3240.0])
    
    # Precompute exact moment matrices
    pre = _hermite_gaussian_moments()
    
    # Seed value
    x0_seed = np.array([-1.13, 0.04, 1.0/3240.0])
    seed_val, _ = _objective_and_grad(x0_seed, pre)
    
    best_val = seed_val
    best_x = x0_seed.copy()
    
    # Multi-start with L-BFGS-B
    starts = [
        x0_seed,
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.5, 0.03, 0.0005]),
        np.array([-0.8, 0.06, 0.0002]),
        np.array([-1.2, 0.035, 0.00035]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-2.0, 0.1, 0.001]),
        np.array([-0.5, 0.1, 0.001]),
        np.array([-1.1, 0.045, 0.00031]),
        np.array([-1.15, 0.038, 0.00029]),
        np.array([-1.05, 0.042, 0.00032]),
        np.array([-0.9, 0.03, 0.0004]),
        np.array([-1.3, 0.05, 0.00025]),
        np.array([-0.7, 0.08, 0.00015]),
        np.array([-1.25, 0.038, 0.00033]),
    ]
    
    # Add random starts
    rng = np.random.RandomState(42)
    for _ in range(20):
        s = x0_seed + rng.randn(3) * np.array([0.3, 0.02, 0.0001])
        starts.append(s)
    
    for start in starts:
        if _time_left() < 10.0:
            break
        try:
            res = minimize(
                lambda p: _objective_grad(p, pre),
                start,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 2000}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            try:
                res = minimize(
                    lambda p: _objective(p, pre),
                    start,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
    
    # Differential evolution for global search
    if _time_left() > 30.0:
        try:
            bounds = [(-3.0, 0.0), (-0.2, 0.2), (-0.001, 0.002)]
            de_result = differential_evolution(
                lambda p: _objective(p, pre),
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-10,
                popsize=15,
                polish=True,
                init='sobol'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x.copy()
        except Exception:
            pass
    
    # Final polish
    if _time_left() > 10.0 and np.isfinite(best_val):
        try:
            res = minimize(
                lambda p: _objective_grad(p, pre),
                best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 5000}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x


if __name__ == "__main__":
    result = construct()
    print("Coefficients (c0, c4, c8):", result)
