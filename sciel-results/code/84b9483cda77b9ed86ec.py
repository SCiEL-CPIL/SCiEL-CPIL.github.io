# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import hermite, eval_hermite
from scipy.optimize import minimize
import time

def _compute_uncertainty_product(c0, c4, c8):
    """Compute A(g)*A(g_hat) for g = (c0*H0 + c4*H4 + c8*H8 + c12*H12)*exp(-x^2/2)"""
    # H12 coefficient from g(0)=0
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    h0_0 = eval_hermite(0, 0.0)
    h4_0 = eval_hermite(4, 0.0)
    h8_0 = eval_hermite(8, 0.0)
    h12_0 = eval_hermite(12, 0.0)
    
    if abs(h12_0) < 1e-15:
        return 1e10
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    
    # Build polynomial P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # Use quadrature to compute integrals
    x = np.linspace(-20, 20, 20001)
    dx = x[1] - x[0]
    
    # Hermite polynomials
    H0 = np.ones_like(x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = P * np.exp(-x**2 / 2)
    
    g2 = g**2
    norm2 = np.sum(g2) * dx
    if norm2 < 1e-15:
        return 1e10
    
    # A(g)^2 = integral x^2 |g|^2 / integral |g|^2
    A2_g = np.sum(x**2 * g2) * dx / norm2
    
    # Fourier transform: for g(x) = P(x) e^{-x^2/2}, 
    # with physicists' Hermite, F[H_n(x) e^{-x^2/2}] = i^n H_n(ξ) e^{-ξ^2/2} (up to normalization)
    # Actually, the Fourier transform convention matters. Let's use the standard:
    # F[f](ξ) = ∫ f(x) e^{-2πiξx} dx
    # For Hermite-Gaussians with this convention, it's more complex.
    # 
    # Alternative: use the fact that for g(x) = P(x) e^{-x^2/2},
    # we can compute the Fourier transform numerically.
    
    # Use FFT or direct computation
    # Actually, let's use the property that the Fourier transform of H_n(x) e^{-x^2/2}
    # with the convention F[f](ξ) = (1/sqrt(2π)) ∫ f(x) e^{-ixξ} dx is i^n H_n(ξ) e^{-ξ^2/2}
    # But with F[f](ξ) = ∫ f(x) e^{-2πiξx} dx, it's different.
    
    # Let me use the convention where the uncertainty principle gives A(f)*A(f_hat) >= 1/4
    # with F[f](ξ) = (1/sqrt(2π)) ∫ f(x) e^{-ixξ} dx
    # Then for g(x) = P(x) e^{-x^2/2}, F[g](ξ) = (1/sqrt(2π)) ∫ P(x) e^{-x^2/2} e^{-ixξ} dx
    # 
    # This is the characteristic function of the measure P(x)e^{-x^2/2} dx.
    # 
    # For computational purposes, let me just compute it numerically.
    
    # Use a wider grid for Fourier transform
    N = 65536
    L = 40.0
    xi = np.linspace(-L, L, N)
    dxi = xi[1] - xi[0]
    
    # Compute F[g](ξ) = (1/sqrt(2π)) ∫ g(x) e^{-ixξ} dx
    # Use trapezoidal rule
    # g(x) = P(x) e^{-x^2/2}
    # We need P(x) on the x-grid
    # Let's use the same x-grid
    # F[g](ξ) ≈ (1/sqrt(2π)) Σ g(x_j) e^{-ix_j ξ} dx
    
    # This is expensive. Let's use FFT.
    # Resample g onto a uniform grid suitable for FFT
    
    # Actually, for a 210s budget, let me be smarter.
    # The Fourier transform of a Hermite-Gaussian is another Hermite-Gaussian.
    # With the convention F[f](ξ) = (1/√(2π)) ∫ f(x) e^{-ixξ} dx:
    # F[H_n(x) e^{-x²/2}](ξ) = i^n H_n(ξ) e^{-ξ²/2}
    
    # So F[g](ξ) = (c0·H0(ξ) + c4·i⁴H4(ξ) + c8·i⁸H8(ξ) + c12·i¹²H12(ξ)) e^{-ξ²/2}
    # i⁴ = 1, i⁸ = 1, i¹² = -1
    # So F[g](ξ) = (c0·H0(ξ) + c4·H4(ξ) + c8·H8(ξ) - c12·H12(ξ)) e^{-ξ²/2}
    
    P_hat = c0 * H0 + c4 * H4 + c8 * H8 - c12 * H12
    g_hat = P_hat * np.exp(-x**2 / 2)
    
    ghat2 = g_hat**2
    norm2_hat = np.sum(ghat2) * dx
    if norm2_hat < 1e-15:
        return 1e10
    
    A2_ghat = np.sum(x**2 * ghat2) * dx / norm2_hat
    
    A_g = np.sqrt(A2_g)
    A_ghat = np.sqrt(A2_ghat)
    
    return A_g * A_ghat


def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    # Initial guess from seed
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0
    
    best_result = (c0_init, c4_init, c8_init)
    best_val = _compute_uncertainty_product(c0_init, c4_init, c8_init)
    
    # Multi-start optimization
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = [(c0_init, c4_init, c8_init)]
    
    # Random perturbations
    for _ in range(20):
        scale = 1.0
        c0 = c0_init + rng.normal(0, scale)
        c4 = c4_init + rng.normal(0, scale * 0.1)
        c8 = c8_init + rng.normal(0, scale * 0.01)
        starts.append((c0, c4, c8))
    
    # Grid-like starts
    for c0 in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
        for c4 in [-0.5, -0.2, 0.0, 0.2, 0.5]:
            for c8 in [-0.05, -0.01, 0.0, 0.01, 0.05]:
                starts.append((c0, c4, c8))
    
    for c0, c4, c8 in starts:
        if time.time() > deadline - 30.0:
            break
        val = _compute_uncertainty_product(c0, c4, c8)
        if val < best_val:
            best_val = val
            best_result = (c0, c4, c8)
    
    # Local optimization from best point
    if time.time() < deadline - 30.0:
        def objective(params):
            return _compute_uncertainty_product(params[0], params[1], params[2])
        
        try:
            res = minimize(objective, best_result, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if time.time() < deadline - 10.0 and res.fun < best_val:
                best_val = res.fun
                best_result = (res.x[0], res.x[1], res.x[2])
        except Exception:
            pass
    
    # Second round of local optimization from multiple points
    if time.time() < deadline - 30.0:
        for c0, c4, c8 in starts[:10]:
            if time.time() > deadline - 20.0:
                break
            def objective(params):
                return _compute_uncertainty_product(params[0], params[1], params[2])
            try:
                res = minimize(objective, (c0, c4, c8), method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_result = (res.x[0], res.x[1], res.x[2])
            except Exception:
                pass
    
    return np.array(best_result)
# EVOLVE-BLOCK-END
