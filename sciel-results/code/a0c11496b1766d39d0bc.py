"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite(n, x):
    """Compute the physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2.0 * x
    h_nm2 = np.ones_like(x, dtype=float)
    h_nm1 = 2.0 * x
    for k in range(2, n + 1):
        h_n = 2.0 * x * h_nm1 - 2.0 * (k - 1) * h_nm2
        h_nm2 = h_nm1
        h_nm1 = h_n
    return h_nm1


def _compute_bound(a0, a4, a8):
    """Compute the upper bound C4 on A(f)*A(Fourier(f)).
    
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 chosen so g(0) = 0.
    """
    # H_n(0) for even n: H_{2k}(0) = (-1)^k * (2k)! / k!
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    
    # Compute moments numerically
    x = np.linspace(-15, 15, 10001)
    dx = x[1] - x[0]
    
    H0 = np.ones_like(x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    g2 = P**2 * np.exp(-x**2)
    
    norm = np.sum(g2) * dx
    if norm < 1e-15:
        return 1e10
    
    mean_x = np.sum(x * g2) * dx / norm
    var_x = np.sum((x - mean_x)**2 * g2) * dx / norm
    
    # Fourier side: F[H_n(x) e^{-x^2/2}] = i^n H_n(ξ) e^{-ξ^2/2} (unitary convention)
    # i^0=1, i^4=1, i^8=1, i^12=-1
    P_hat = a0 * H0 + a4 * H4 + a8 * H8 - a12 * H12
    g_hat2 = P_hat**2 * np.exp(-x**2)
    
    norm_hat = np.sum(g_hat2) * dx
    if norm_hat < 1e-15:
        return 1e10
    
    mean_xi = np.sum(x * g_hat2) * dx / norm_hat
    var_xi = np.sum((x - mean_xi)**2 * g_hat2) * dx / norm_hat
    
    # Bound: A(f)*A(F) ≈ 2*sqrt(var_x * var_xi)
    bound = 2.0 * np.sqrt(max(var_x, 0.0) * max(var_xi, 0.0))
    
    return bound


def construct():
    """Optimize the coefficients to minimize the upper bound C4."""
    start_time = time.time()
    deadline = start_time + 165  # Safety margin
    
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_x = x0.copy()
    best_bound = _compute_bound(*x0)
    
    # Primary optimization
    try:
        result = minimize(
            lambda x: _compute_bound(*x),
            x0,
            method='Nelder-Mead',
            options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result.fun < best_bound:
            best_bound = result.fun
            best_x = result.x
    except Exception:
        pass
    
    # Additional trials from different starting points
    if time.time() < deadline - 20:
        rng = np.random.RandomState(12345)
        for trial in range(30):
            if time.time() >= deadline - 15:
                break
            scale = 0.05 * (1 + trial * 0.5)
            x_trial = x0 + rng.normal(0, scale, size=3)
            try:
                b = _compute_bound(*x_trial)
                if b < best_bound:
                    best_bound = b
                    best_x = x_trial.copy()
                    result = minimize(
                        lambda x: _compute_bound(*x),
                        x_trial,
                        method='Nelder-Mead',
                        options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    if result.fun < best_bound:
                        best_bound = result.fun
                        best_x = result.x
            except Exception:
                pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
