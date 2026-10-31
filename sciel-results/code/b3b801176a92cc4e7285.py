"""Rewritten program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import factorial, sqrt, pi
from scipy.optimize import minimize, differential_evolution


def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # safety margin

    # Hermite-Gaussian test function: g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    # Constraint: g(0) = 0 => c0 + H4(0)*c4 + H8(0)*c8 + H12(0)*c12 = 0
    # H_n(0) for even n: H_0(0)=1, H_4(0)=-12, H_8(0)=1680, H_12(0)=-665280
    # So: c0 - 12*c4 + 1680*c8 - 665280*c12 = 0
    # => c12 = (c0 - 12*c4 + 1680*c8) / 665280

    degrees = [0, 4, 8, 12]
    n_vals = len(degrees)

    # Precompute factorials and powers
    fact = [factorial(d) for d in degrees]
    pow2 = [2 ** d for d in degrees]
    sqrt_pi = sqrt(pi)

    # Mass matrix M: M[i,j] = ∫ H_i H_j e^{-x^2} dx = sqrt(pi) * 2^d * d! * delta_ij
    M = np.zeros((n_vals, n_vals))
    for i, d in enumerate(degrees):
        M[i, i] = sqrt_pi * pow2[i] * fact[i]

    # Position matrix A: A[i,j] = ∫ x^2 H_i H_j e^{-x^2} dx
    # x^2 H_n = (1/4)H_{n+2} + n*H_n + (n/2)H_{n-2}
    A = np.zeros((n_vals, n_vals))
    for i, di in enumerate(degrees):
        for j, dj in enumerate(degrees):
            val = 0.0
            if di + 2 == dj:
                val += 0.25 * sqrt_pi * pow2[j] * fact[j]
            if di == dj:
                val += di * sqrt_pi * pow2[i] * fact[i]
            if di >= 2 and di - 2 == dj:
                val += (di / 2.0) * sqrt_pi * pow2[j] * fact[j]
            A[i, j] = val

    # Momentum matrix B: B[i,j] = ∫ (p'_i - x*p_i)(p'_j - x*p_j) e^{-x^2} dx
    # p'_n - x*p_n = n*H_{n-1} - (1/2)H_{n+1}
    # B is diagonal: B_ii = sqrt(pi) * [i^2 * 2^{i-1}*(i-1)! + (1/4)*2^{i+1}*(i+1)!] for i>=1
    # For i=0: B_00 = sqrt(pi) * (1/4) * 2^1 * 1! = sqrt(pi)/2
    B = np.zeros((n_vals, n_vals))
    for i, d in enumerate(degrees):
        if d == 0:
            B[i, i] = sqrt_pi / 2.0
        else:
            B[i, i] = sqrt_pi * (2 ** (d - 1)) * factorial(d - 1) * d * (2 * d + 1)

    # Constraint: c12 = (c0 - 12*c4 + 1680*c8) / 665280
    # Parameterize: c = [c0, c4, c8, c12] where c12 is determined by c0, c4, c8.
    # Scale invariance: fix c0 = 1, optimize c4 = a, c8 = b.
    # c = [1, a, b, (1 - 12a + 1680b)/665280]

    c_base = np.array([1.0, 0.0, 0.0, 1.0 / 665280.0])
    v_a = np.array([0.0, 1.0, 0.0, -12.0 / 665280.0])
    v_b = np.array([0.0, 0.0, 1.0, 1680.0 / 665280.0])

    # Precompute quadratic form coefficients
    M0 = c_base @ M @ c_base
    A0 = c_base @ A @ c_base
    B0 = c_base @ B @ c_base

    Ma = c_base @ M @ v_a
    Aa = c_base @ A @ v_a
    Ba = c_base @ B @ v_a

    Mb = c_base @ M @ v_b
    Ab = c_base @ A @ v_b
    Bb = c_base @ B @ v_b

    Maa = v_a @ M @ v_a
    Aaa = v_a @ A @ v_a
    Baa = v_a @ B @ v_a

    Mbb = v_b @ M @ v_b
    Abb = v_b @ A @ v_b
    Bbb = v_b @ B @ v_b

    Mab = v_a @ M @ v_b
    Aab = v_a @ A @ v_b
    Bab = v_a @ B @ v_b

    def objective(params):
        a, b = params
        M_c = M0 + 2 * a * Ma + 2 * b * Mb + a * a * Maa + 2 * a * b * Mab + b * b * Mbb
        A_c = A0 + 2 * a * Aa + 2 * b * Ab + a * a * Aaa + 2 * a * b * Aab + b * b * Abb
        B_c = B0 + 2 * a * Ba + 2 * b * Bb + a * a * Baa + 2 * a * b * Bab + b * b * Bbb
        if M_c <= 0:
            return 1e10
        return (A_c * B_c) / (M_c * M_c)

    def objective_log(params):
        """Log-magnitude parameterization: a = sign_a * exp(m_a), b = sign_b * exp(m_b)"""
        m_a, m_b, s_a, s_b = params
        a = s_a * np.exp(m_a)
        b = s_b * np.exp(m_b)
        return objective([a, b])

    best_score = float('inf')
    best_result = None

    # Strategy 1: Differential evolution for global search
    if time.time() < deadline - 60:
        try:
            bounds = [(-3, 3), (-3, 3)]
            result_de = differential_evolution(
                objective, bounds, seed=42, maxiter=200, popsize=15,
                tol=1e-12, polish=True, init='sobol'
            )
            if result_de.fun < best_score:
                best_score = result_de.fun
                best_result = result_de.x
        except Exception:
            pass

    # Strategy 2: Multiple local optimizations from diverse starting points
    if time.time() < deadline - 30:
        start_points = []
        # Grid
        for a in np.linspace(-0.5, 0.5, 11):
            for b in np.linspace(-0.05, 0.05, 11):
                start_points.append((a, b))
        # Focused around previous good solutions
        for a0 in [0.04, -0.04, 0.02, -0.02]:
            for b0 in [0.0003, -0.0003, 0.001, -0.001]:
                for da in np.linspace(-0.02, 0.02, 5):
                    for db in np.linspace(-0.001, 0.001, 5):
                        start_points.append((a0 + da, b0 + db))

        for sp in start_points:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(objective, sp, method='Nelder-Mead',
                                  options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-17})
                if result.fun < best_score:
                    best_score = result.fun
                    best_result = result.x
            except Exception:
                pass

    # Strategy 3: Log-magnitude parameterization for wider exploration
    if time.time() < deadline - 15:
        try:
            # Search over log-magnitudes and signs
            for s_a in [1.0, -1.0]:
                for s_b in [1.0, -1.0]:
                    for m_a0 in np.linspace(-2, 2, 5):
                        for m_b0 in np.linspace(-4, 0, 5):
                            if time.time() > deadline - 5:
                                break
                            try:
                                init = [m_a0, m_b0, s_a, s_b]
                                result = minimize(objective_log, init, method='Nelder-Mead',
                                                  options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
                                if result.fun < best_score:
                                    best_score = result.fun
                                    a_val = s_a * np.exp(result.x[0])
                                    b_val = s_b * np.exp(result.x[1])
                                    best_result = np.array([a_val, b_val])
                            except Exception:
                                pass
                    else:
                        continue
                    break
        except Exception:
            pass

    if best_result is None:
        return np.array([-113 / 100, 1 / 25, 1 / 3240])

    a_opt, b_opt = best_result
    return np.array([1.0, a_opt, b_opt])


# EVOLVE-BLOCK-END
