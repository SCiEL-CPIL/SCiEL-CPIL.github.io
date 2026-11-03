import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite
import time


def construct():
    t_start = time.time()
    deadline = 195.0  # Safety margin from 210s limit

    # Hermite polynomial values at 0 (physicists' convention)
    # H_0(0) = 1
    # H_4(0) = -12
    # H_8(0) = 1680
    # H_12(0) = -3991680
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 1680.0
    H12_0 = -3991680.0

    def compute_a12(a0, a4, a8):
        """Compute a12 from constraint P(0) = 0"""
        return -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0

    # Precompute grids and Hermite values for speed
    # Use a moderate grid size for the main objective
    x_main = np.linspace(-10, 10, 2001)
    dx_main = x_main[1] - x_main[0]
    H0_main = eval_hermite(0, x_main)
    H4_main = eval_hermite(4, x_main)
    H8_main = eval_hermite(8, x_main)
    H12_main = eval_hermite(12, x_main)
    
    # For the Fourier side, we need the "minus" polynomial P_minus
    # where the a12 term is subtracted (due to Fourier transform properties of Hermite-Gaussians)
    # g(x) = P(x) e^{-pi x^2}
    # F[g](xi) is proportional to P_minus(xi) e^{-pi xi^2} where P_minus has -a12 for H12 term?
    # Actually, for physicists' Hermite polynomials H_n(x) e^{-x^2/2}, the FT is (i)^n H_n(k) e^{-k^2/2}.
    # Here we have e^{-pi x^2}. Let's stick to the numerical FFT approach or the specific
    # analytic property if known. 
    # The inspiration 2 suggests: P_minus(x) = a0 H0 + a4 H4 + a8 H8 - a12 H12.
    # This implies the FT of the H12 term flips sign? Or is it related to i^12 = 1?
    # i^12 = (i^4)^3 = 1. So it shouldn't flip sign in a simple way unless there's a phase.
    # However, the L-infinity norm is of the magnitude, so phase doesn't matter for the sup norm.
    # But the *shape* of the function in the frequency domain is what matters for the sup norm.
    # If the FT of P(x)e^{-pi x^2} is Q(xi)e^{-pi xi^2}, what is Q?
    # Using the property that FT of H_n(x) e^{-pi x^2} is (i)^n H_n(xi) e^{-pi xi^2} (with appropriate scaling).
    # For n=12, i^12 = 1. So the coefficient a12 stays a12.
    # Wait, inspiration 2 uses P_minus. Let's check the sign.
    # If the FT preserves the polynomial structure with coefficients c_n -> (i)^n c_n, then for n=12, c12 -> c12.
    # So P_hat(xi) = P(xi).
    # Why did inspiration 2 use P_minus? Maybe a different convention or a specific trick?
    # Let's use the direct numerical FFT approach to be safe, as it doesn't rely on analytic assumptions
    # about the exact form of the FT of the polynomial-Gaussian product, other than that it decays.
    # However, FFT is slow.
    # Alternative: The "upper bound on C4" likely refers to the constant in the uncertainty principle
    # ||f||_inf ||f_hat||_inf >= C ||f||_2^2 ? Or similar.
    # The task says "reduce the upper bound on A(f)*A(Fourier(f))".
    # Usually A(f) is the spread or amplitude.
    # Given the context of "C4" and "Hermite", this is likely related to the optimal constant in
    # ||f||_inf ||f_hat||_inf <= C ||f||_2^2 or similar, but for specific functions?
    # No, "Tune ... to reduce the upper bound". This implies we are finding the best constant C
    # such that the inequality holds for all f? Or we are finding a specific f that minimizes the ratio?
    # "score is the upper bound on C4".
    # If we minimize the ratio (||g||_inf ||g_hat||_inf) / ||g||_2^2, we are finding the best possible
    # constant for this class of functions?
    # Let's assume the objective is to minimize the product of the sup-norms, normalized by the L2 norm squared.
    # Objective: (max|g| * max|g_hat|) / (||g||_2^2)
    
    def objective(params):
        a0, a4, a8 = params
        a12 = compute_a12(a0, a4, a8)
        
        P = a0 * H0_main + a4 * H4_main + a8 * H8_main + a12 * H12_main
        g = P * np.exp(-np.pi * x_main**2)
        
        norm_g = np.max(np.abs(g))
        norm_g_l2_sq = np.sum(g**2) * dx_main
        
        if norm_g_l2_sq < 1e-10:
            return 1e10
            
        # Compute Fourier Transform
        # Use FFT
        N = len(x_main)
        G = np.fft.fftshift(np.fft.fft(g)) * dx_main
        norm_g_hat = np.max(np.abs(G))
        
        # The quantity to minimize
        # Is it the product? Or the ratio?
        # "reduce the upper bound on A(f)*A(Fourier(f))"
        # If A(f) is sup norm, then we minimize sup(g)*sup(g_hat).
        # But this is scale invariant? No, if we scale g, both scale.
        # We must fix the scale. The constraint P(0)=0 fixes the relative coefficients, but not the overall scale.
        # The overall scale cancels out in a ratio like sup(g)*sup(g_hat)/||g||_2^2.
        # Let's use this ratio as the objective.
        
        return (norm_g * norm_g_hat) / norm_g_l2_sq

    # Initial guess from the seed program
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    best_result = x0
    best_score = objective(x0)

    # Try multiple starting points with local optimization
    # Use L-BFGS-B for efficiency
    start_points = [
        x0,
        x0 + np.array([0.1, 0.01, 0.001]),
        x0 - np.array([0.1, 0.01, 0.001]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-0.5, 0.05, 0.0005]),
        np.array([0.0, 0.0, 0.0]),
    ]

    for sp in start_points:
        if time.time() > deadline - 30:
            break
        try:
            result = minimize(
                objective,
                sp,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            continue

    # If we have time, try a global optimization (Differential Evolution)
    if time.time() < deadline - 60:
        try:
            bounds = [(-5, 5), (-1, 1), (-0.1, 0.1)]
            de_result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-8,
                polish=False
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x.copy()
                
                # Polish the result
                if time.time() < deadline - 10:
                    polish_result = minimize(
                        objective,
                        best_result,
                        method='L-BFGS-B',
                        options={'maxiter': 1000, 'ftol': 1e-14}
                    )
                    if polish_result.fun < best_score:
                        best_score = polish_result.fun
                        best_result = polish_result.x.copy()
        except Exception:
            pass

    return best_result
