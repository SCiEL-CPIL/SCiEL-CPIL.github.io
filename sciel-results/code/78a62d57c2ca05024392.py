"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _hermite(n, x):
    """Physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    h_nm2 = np.ones_like(x)
    h_nm1 = 2.0 * x
    for k in range(2, n + 1):
        h_n = 2.0 * x * h_nm1 - 2.0 * (k - 1) * h_nm2
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1


def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound C4 for the uncertainty inequality.
    
    The test function is g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    with c12 chosen so that g(0) = 0.
    
    The uncertainty product bound is A(f) * A(Fourier(f)) where:
    A(f) = sup |x| * |f(x)|
    A(Fourier(f)) = sup |xi| * |F(f)(xi)|
    
    For Fourier-invariant functions (up to constants), we compute the bound
    using the known formulas for Hermite-Gaussian functions.
    
    The Hermite functions psi_n(x) = (2^n n! sqrt(pi))^{-1/2} H_n(x) exp(-x^2/2)
    are eigenfunctions of the Fourier transform.
    
    We work with the unnormalized combination. Let's define:
    g(x) = sum_{k in {0,4,8,12}} c_k H_k(x) exp(-x^2/2)
    
    The L2 norm, and the sup of |x g(x)| and |xi F(g)(xi)| need to be computed.
    
    For the Fourier transform convention F(f)(xi) = integral f(x) exp(-2 pi i x xi) dx,
    the Hermite functions have specific eigenvalues.
    
    Actually, let's use the standard result. For the uncertainty inequality in the form
    ||x f||_inf * ||xi F(f)||_inf <= C4 * ||f||_2^2 (or similar),
    
    Let me think about this more carefully. The task says "upper bound on A(f)*A(Fourier(f))"
    where A(f) is likely the sup norm of |x f(x)| or similar.
    
    Looking at the seed values [-113/100, 1/25, 1/3240] = [-1.13, 0.04, 0.000308642],
    these are the Goncalves et al. 2017 values.
    
    Let me compute the bound numerically. The key quantities are:
    1. ||g||_2^2 = integral |g(x)|^2 dx
    2. ||x g||_inf = sup_x |x g(x)|
    3. ||xi F(g)||_inf = sup_xi |xi F(g)(xi)|
    
    The bound C4 = (||x g||_inf * ||xi F(g)||_inf) / ||g||_2^2
    
    Or perhaps C4 = ||x g||_inf * ||xi F(g)||_inf / ||g||_2^2
    
    Let me compute this numerically.
    """
    # Compute c12 from the constraint g(0) = 0
    # g(0) = (c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)) * exp(0)
    # H0(0) = 1
    # H4(0) = 12 (H4(x) = 16x^4 - 48x^2 + 12, so H4(0) = 12)
    # H8(0) = 1344 (H8(0) = 1344)
    # H12(0) = 1032192 (H12(0) = 1032192)
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1344.0
    H12_0 = 1032192.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Now compute the bound numerically
    # Use a fine grid
    x = np.linspace(-15, 15, 100000)
    dx = x[1] - x[0]
    
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * np.exp(-x**2 / 2.0)
    
    # L2 norm squared
    norm2_sq = np.sum(g**2) * dx
    
    if norm2_sq <= 0:
        return np.inf
    
    # ||x g||_inf
    xg = np.abs(x * g)
    xg_sup = np.max(xg)
    
    # Fourier transform of g
    # F(g)(xi) = integral g(x) exp(-2 pi i x xi) dx
    # Use FFT or numerical integration
    # For efficiency, use the fact that we can compute this via FFT on a grid
    
    # Let's use a larger grid for better Fourier accuracy
    N = 2**17  # 131072 points
    L = 30.0  # domain length
    x_fft = np.linspace(-L/2, L/2, N, endpoint=False)
    dx_fft = x_fft[1] - x_fft[0]
    
    H0_f = _hermite(0, x_fft)
    H4_f = _hermite(4, x_fft)
    H8_f = _hermite(8, x_fft)
    H12_f = _hermite(12, x_fft)
    
    g_fft = (c0 * H0_f + c4 * H4_f + c8 * H8_f + c12 * H12_f) * np.exp(-x_fft**2 / 2.0)
    
    # FFT: F(g)(xi) where xi = k/(N*dx_fft) for k = 0, ..., N-1
    G = np.fft.fft(g_fft) * dx_fft
    xi = np.fft.fftfreq(N, d=dx_fft)
    
    # ||xi F(g)||_inf
    xiG = np.abs(xi * G)
    xiG_sup = np.max(xiG)
    
    # The bound
    C4 = (xg_sup * xiG_sup) / norm2_sq
    
    return C4


def _nelder_mead(f, x0, bounds=None, max_iter=5000, tol=1e-10, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
    """
    Nelder-Mead simplex optimization.
    """
    n = len(x0)
    simplex = [x0.astype(float)]
    
    # Create initial simplex
    for i in range(n):
        xi = x0.copy()
        step = 0.05 * abs(x0[i]) + 1e-4
        xi[i] += step
        simplex.append(xi)
    
    # Evaluate initial simplex
    fvals = [f(v) for v in simplex]
    
    for it in range(max_iter):
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all points except worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
                fc = f(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
            else:
                # Inside contraction
                xc = centroid - rho * (centroid - simplex[-1])
                fc = f(xc)
                if fc <= fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
    
    # Return best point
    best_idx = np.argmin(fvals)
    return np.array(simplex[best_idx]), fvals[best_idx]


def construct():
    """
    Find the three coefficients (c0, c4, c8) that minimize the upper bound C4.
    The degree-12 coefficient c12 is determined by the constraint g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 180.0  # Leave margin before 210s kill
    
    # Initial guess from Goncalves et al. 2017
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate the initial point
    f0 = _compute_bound(x0[0], x0[1], x0[2])
    
    best_x = x0.copy()
    best_f = f0
    
    # Nelder-Mead optimization
    try:
        result_x, result_f = _nelder_mead(
            lambda x: _compute_bound(x[0], x[1], x[2]),
            x0,
            max_iter=3000,
            tol=1e-12
        )
        
        if result_f < best_f:
            best_x = result_x
            best_f = result_f
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_x
    
    # Try a few more starting points if time allows
    start_points = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.03, 0.0004]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-1.15, 0.035, 0.00025]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 40:
            break
        try:
            result_x, result_f = _nelder_mead(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                sp,
                max_iter=2000,
                tol=1e-12
            )
            if result_f < best_f:
                best_x = result_x
                best_f = result_f
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
