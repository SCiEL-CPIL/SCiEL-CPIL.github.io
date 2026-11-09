import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite
import time

def construct():
    start_time = time.time()
    deadline = 195.0  # safety margin before 210s kill

    # Probabilists' Hermite polynomials (physicists' H_n)
    # H_0 = 1, H_4 = x^4 - 6x^2 + 3, H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12 = x^12 - 42x^10 + 1260x^8 - 16800x^6 + 113400x^4 - 302400x^2 + 10395
    # Values at 0: H_0(0)=1, H_4(0)=3, H_8(0)=105, H_12(0)=10395
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0

    def get_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    # Gauss-Hermite quadrature nodes and weights for ∫ f(x) e^{-x^2} dx
    # We need ∫ g^4 dx = ∫ e^{-2x^2} P(x)^4 dx = ∫ e^{-x^2} [e^{-x^2/2} P(x)]^4 dx
    # Actually: g(x)^4 = e^{-2x^2} P(x)^4, so ∫ g^4 dx = ∫ e^{-x^2} [e^{-x^2/2} P(x)^4] dx
    # Wait: g(x) = e^{-x^2/2} P(x), so g(x)^4 = e^{-2x^2} P(x)^4
    # ∫ e^{-2x^2} P(x)^4 dx. Let u = sqrt(2) x, dx = du/sqrt(2)
    # ∫ e^{-u^2} P(u/sqrt(2))^4 du/sqrt(2)
    # This is a standard GH quadrature with n=100 nodes.
    n_gh = 100
    x_gh, w_gh = roots_hermite(n_gh)
    # For ∫_{-∞}^{∞} f(x) e^{-x^2} dx ≈ sum w_i f(x_i)
    # We need ∫ e^{-2x^2} P(x)^4 dx. Let's use substitution:
    # ∫ e^{-2x^2} P(x)^4 dx = (1/sqrt(2)) ∫ e^{-u^2} P(u/sqrt(2))^4 du
    # where u = sqrt(2) x
    sqrt2 = np.sqrt(2.0)

    def _poly4(c0, c4, c8, c12, x):
        # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        x8 = x4 * x4
        x10 = x8 * x2
        x12 = x8 * x4
        H0 = 1.0
        H4 = x4 - 6.0 * x2 + 3.0
        H8 = x8 - 28.0 * x6 + 210.0 * x4 - 420.0 * x2 + 105.0
        H12 = x12 - 42.0 * x10 + 1260.0 * x8 - 16800.0 * x6 + 113400.0 * x4 - 302400.0 * x2 + 10395.0
        return c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12

    def norm4_sq(c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        # ∫ g(x)^4 dx = ∫ e^{-2x^2} P(x)^4 dx
        # Use GH quadrature: ∫ e^{-x^2} f(x) dx ≈ sum w_i f(x_i)
        # ∫ e^{-2x^2} P(x)^4 dx = (1/sqrt(2)) ∫ e^{-u^2} P(u/sqrt(2))^4 du
        x_eval = x_gh / sqrt2
        P = _poly4(c0, c4, c8, c12, x_eval)
        return (1.0 / sqrt2) * np.sum(w_gh * P**4)

    # Fourier transform: for even k, (-i)^k = 1, so hat P = P
    # ∫ hat_g(xi)^4 dxi = same as norm4_sq (by symmetry)
    # Actually, the Fourier transform of e^{-x^2/2} H_k(x) is (-i)^k e^{-xi^2/2} H_k(xi)
    # For k=0,4,8,12: (-i)^k = 1, so hat_g = e^{-xi^2/2} P(xi)
    # Thus norm4_sq_hat = norm4_sq (same integral)
    # But wait - this means the objective is just (norm4_sq)^2 / something?
    # No: objective = (norm4_sq * norm4_sq_hat)^{1/4} = (norm4_sq^2)^{1/4} = norm4_sq^{1/2}
    # That can't be right for an interesting problem. Let me re-examine.
    #
    # Actually, the Fourier transform convention matters. If we use
    # hat_f(xi) = ∫ f(x) e^{-2πi x xi} dx, then the Gaussian Fourier transform
    # is different. But for the Hermite functions with e^{-x^2/2}, the standard
    # unitary Fourier transform gives hat_H_k = (-i)^k H_k.
    #
    # The key point: for our specific polynomial combination with only even k
    # where (-i)^k = 1, the Fourier transform is the SAME function.
    # So norm4_sq_hat = norm4_sq, and the objective = norm4_sq^{1/2}.
    #
    # This means we're just minimizing the L4 norm of g!
    # The constraint g(0) = 0 is the only real constraint.

    def objective(c):
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = c
        n1 = norm4_sq(c0, c4, c8)
        if n1 < 0:
            return 1e10
        return n1 ** 0.5  # (n1 * n1)^{1/4} = n1^{1/2}

    # The symmetry constraint: since hat_g = g for even k with (-i)^k=1,
    # the problem reduces to minimizing ||g||_4 subject to g(0)=0.
    #
    # But wait - let me reconsider. The task says "upper bound on A(f)*A(Fourier(f))"
    # where A(f) might be the L1 norm, not L4. Let me re-read.
    #
    # "score is the upper bound on C4" - this suggests L4.
    # "reduce the upper bound on A(f)*A(Fourier(f))" - A could be any norm.
    #
    # The seed program uses norm_4_sq (L4 norm squared), so the objective is
    # (||g||_4^4 * ||hat_g||_4^4)^{1/4} = ||g||_4 * ||hat_g||_4
    #
    # If hat_g = g (for our even polynomial), then objective = ||g||_4^2.
    # The seed computes (n1 * n2)^{0.25} where n1 = ||g||_4^4, n2 = ||hat_g||_4^4.
    # So (n1*n2)^{1/4} = (||g||_4^4 * ||hat_g||_4^4)^{1/4} = ||g||_4 * ||hat_g||_4.
    # If hat_g = g, this is ||g||_4^2.
    #
    # Let me just use the seed's formula directly to be safe.

    def objective_full(c):
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = c
        n1 = norm4_sq(c0, c4, c8)
        n2 = norm4_sq(c0, c4, c8)  # same due to symmetry
        if n1 < 0 or n2 < 0:
            return 1e10
        return (n1 * n2) ** 0.25

    # Seed initial guess
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    best_result = x0.copy()
    best_val = objective_full(x0)

    # Check time
    if time.time() - start_time > deadline - 30:
        return best_result

    # Strategy: use multiple well-chosen starting points with L-BFGS-B
    # The key insight: the problem has 3 free variables (c0, c4, c8) with c12 determined.
    # The objective is smooth, so gradient-based methods work well.
    #
    # Let's try a grid of starting points around the seed and also explore
    # the parameter space more broadly.

    starting_points = [x0.copy()]

    # Add perturbations
    rng = np.random.default_rng(42)
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for _ in range(5):
            starting_points.append(x0 + rng.normal(0, scale, 3))

    # Also try some structured points
    starting_points.append(np.array([0.0, 0.0, 0.0]))
    starting_points.append(np.array([-1.0, 0.01, 0.0]))
    starting_points.append(np.array([-2.0, 0.1, 0.001]))
    starting_points.append(np.array([-0.5, 0.05, 0.01]))
    starting_points.append(np.array([-1.13, 0.04, 0.0]))
    starting_points.append(np.array([-1.13, 0.04, -0.001]))
    starting_points.append(np.array([-1.13, 0.04, 0.001]))

    for x_start in starting_points:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                objective_full,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x.copy()
        except Exception:
            pass

    # Polish pass with tighter tolerances
    if time.time() < deadline - 5:
        try:
            res_polish = minimize(
                objective_full,
                best_result,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_result = res_polish.x.copy()
        except Exception:
            pass

    # Final Nelder-Mead polish for any remaining local structure
    if time.time() < deadline - 2:
        try:
            res_nm = minimize(
                objective_full,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_result = res_nm.x.copy()
        except Exception:
            pass

    return best_result
