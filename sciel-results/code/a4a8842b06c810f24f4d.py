"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def construct():
    start_time = time.time()
    deadline = start_time + 180.0

    try:
        # Hermite-Gaussian test function: g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
        # Constraint: g(0) = 0 => c0 + 12*c4 + 1680*c8 + 665280*c12 = 0
        # So c12 = -(c0 + 12*c4 + 1680*c8) / 665280
        #
        # The uncertainty product bound is:
        # (∫ x^2 |g|^2 dx) * (∫ |g'|^2 dx) / (∫ |g|^2 dx)^2
        #
        # Using physicists' Hermite polynomials with weight e^{-x^2}:
        # ∫ H_m H_n e^{-x^2} dx = sqrt(pi) * 2^n * n! * delta_mn
        #
        # But our test function is (polynomial) * exp(-x^2/2), so |g|^2 = (poly)^2 * exp(-x^2)
        # This is exactly the Hermite weight!
        #
        # Let p(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        # Then:
        # ∫ |g|^2 dx = ∫ p(x)^2 e^{-x^2} dx = c^T M c
        # ∫ x^2 |g|^2 dx = ∫ x^2 p(x)^2 e^{-x^2} dx = c^T A c
        # ∫ |g'|^2 dx = ∫ (p'(x) exp(-x^2/2) - x p(x) exp(-x^2/2))^2 dx
        #             = ∫ (p'(x) - x p(x))^2 e^{-x^2} dx
        #
        # Using d/dx H_n = 2n H_{n-1} and x H_n = (1/2)H_{n+1} + n H_{n-1}:
        # p'(x) - x p(x) = sum c_k (2k H_{k-1} - x H_k)
        # = sum c_k (2k H_{k-1} - (1/2)H_{k+1} - k H_{k-1})
        # = sum c_k (k H_{k-1} - (1/2)H_{k+1})
        #
        # So ∫ |g'|^2 = ∫ (sum c_k (k H_{k-1} - (1/2)H_{k+1}))^2 e^{-x^2} dx
        # = c^T B c where B_{ij} = ∫ (i H_{i-1} - (1/2)H_{i+1})(j H_{j-1} - (1/2)H_{j+1}) e^{-x^2} dx
        #
        # By orthogonality, this is diagonal:
        # B_{ii} = i^2 * (2^{i-1} (i-1)!) + (1/4) * (2^{i+1} (i+1)!)  [for i >= 1]
        # For i = 0: B_{00} = (1/4) * 2^1 * 1! = 1/2
        #
        # Wait, let me be more careful. The cross terms vanish by orthogonality.
        # ∫ (i H_{i-1})(j H_{j-1}) e^{-x^2} dx = i*j * sqrt(pi) * 2^{i-1} * (i-1)! * delta_{i,j} (for i,j >= 1)
        # ∫ (i H_{i-1})(-1/2 H_{j+1}) e^{-x^2} dx = 0 (different degrees)
        # ∫ (-1/2 H_{i+1})(j H_{j-1}) e^{-x^2} dx = 0
        # ∫ (-1/2 H_{i+1})(-1/2 H_{j+1}) e^{-x^2} dx = (1/4) * sqrt(pi) * 2^{i+1} * (i+1)! * delta_{i,j}
        #
        # So B is diagonal with:
        # B_{00} = (1/4) * sqrt(pi) * 2^1 * 1! = sqrt(pi)/2
        # B_{ii} = i^2 * sqrt(pi) * 2^{i-1} * (i-1)! + (1/4) * sqrt(pi) * 2^{i+1} * (i+1)!  for i >= 1
        #        = sqrt(pi) * [i^2 * 2^{i-1} * (i-1)! + 2^{i-1} * (i+1)!]
        #        = sqrt(pi) * 2^{i-1} * [i^2 * (i-1)! + (i+1)!]
        #        = sqrt(pi) * 2^{i-1} * (i-1)! * [i^2 + i(i+1)]
        #        = sqrt(pi) * 2^{i-1} * (i-1)! * [i^2 + i^2 + i]
        #        = sqrt(pi) * 2^{i-1} * (i-1)! * [2i^2 + i]
        #        = sqrt(pi) * 2^{i-1} * (i-1)! * i * (2i + 1)

        degrees = [0, 4, 8, 12]
        n_vals = len(degrees)

        # Moment matrix M (diagonal)
        M = np.zeros((n_vals, n_vals))
        for i, deg in enumerate(degrees):
            M[i, i] = np.sqrt(np.pi) * (2 ** deg) * _factorial(deg)

        # Position matrix A
        # x^2 H_n = (1/4) H_{n+2} + n H_n + (n/2) H_{n-2}
        A = np.zeros((n_vals, n_vals))
        for i, deg_i in enumerate(degrees):
            for j, deg_j in enumerate(degrees):
                val = 0.0
                if deg_i + 2 == deg_j:
                    val += 0.25 * np.sqrt(np.pi) * (2 ** deg_j) * _factorial(deg_j)
                if deg_i == deg_j:
                    val += deg_i * np.sqrt(np.pi) * (2 ** deg_i) * _factorial(deg_i)
                if deg_i >= 2 and deg_i - 2 == deg_j:
                    val += (deg_i / 2.0) * np.sqrt(np.pi) * (2 ** deg_j) * _factorial(deg_j)
                A[i, j] = val

        # Momentum matrix B (diagonal)
        B = np.zeros((n_vals, n_vals))
        for i, deg in enumerate(degrees):
            if deg == 0:
                B[i, i] = np.sqrt(np.pi) / 2.0
            else:
                B[i, i] = np.sqrt(np.pi) * (2 ** (deg - 1)) * _factorial(deg - 1) * deg * (2 * deg + 1)

        # Constraint: c12 = -(c0 + 12*c4 + 1680*c8) / 665280
        # Parameterize with c = [c0, c4, c8, c12] where c12 is determined.
        # Let's use c0 = 1 (scale invariance of the Rayleigh quotient).
        # c = [1, a, b, -(1 + 12a + 1680b)/665280]

        c_base = np.array([1.0, 0.0, 0.0, -1.0 / 665280.0])
        v_a = np.array([0.0, 1.0, 0.0, -12.0 / 665280.0])
        v_b = np.array([0.0, 0.0, 1.0, -1680.0 / 665280.0])

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

        # Use scipy optimize with multiple starting points
        from scipy.optimize import minimize

        best_score = float('inf')
        best_result = None

        # Try a grid of starting points
        start_points = []
        for a in np.linspace(-0.1, 0.1, 11):
            for b in np.linspace(-0.01, 0.01, 11):
                start_points.append((a, b))

        # Add some focused points around the seed solution
        seed_a = 1.0 / 25.0
        seed_b = 1.0 / 3240.0
        for da in np.linspace(-0.02, 0.02, 5):
            for db in np.linspace(-0.001, 0.001, 5):
                start_points.append((seed_a + da, seed_b + db))

        for sp in start_points:
            if time.time() > deadline - 30:
                break
            try:
                result = minimize(objective, sp, method='Nelder-Mead',
                                  options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16})
                if result.success and result.fun < best_score:
                    best_score = result.fun
                    best_result = result.x
            except Exception:
                pass

        if best_result is None:
            return np.array([-113 / 100, 1 / 25, 1 / 3240])

        a_opt, b_opt = best_result
        return np.array([1.0, a_opt, b_opt])

    except Exception:
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
# EVOLVE-BLOCK-END
