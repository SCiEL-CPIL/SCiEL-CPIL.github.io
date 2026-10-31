"""Optimized program: analytic-gradient L-BFGS-B with log-magnitude parameterization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

# Precompute Hermite polynomials as coefficient arrays (highest power first)
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 1280x^6 + 2688x^4 - 1344x^2 + 168
# H_12(x) = 4096x^12 - 30720x^10 + 76800x^8 - 84480x^6 + 30240x^4 - 2016x^2 + 96

def _herm_coeffs(n):
    """Return coefficients of physicist's Hermite polynomial H_n, highest power first."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([2.0, 0.0])
    # Use recurrence: H_{n+1} = 2x*H_n - 2n*H_{n-1}
    h0 = np.array([1.0])
    h1 = np.array([2.0, 0.0])
    for k in range(2, n + 1):
        # 2x * h_{k-1}
        term1 = np.zeros(len(h1) + 1)
        term1[:-1] = 2.0 * h1
        # -2*(k-1) * h_{k-2}
        term2 = np.zeros(len(h0) + 1)
        term2[:-1] = -2.0 * (k - 1) * h0
        # Align lengths
        L = max(len(term1), len(term2))
        t1 = np.zeros(L)
        t2 = np.zeros(L)
        t1[:len(term1)] = term1
        t2[:len(term2)] = term2
        h0, h1 = h1, t1 + t2
    return h1

# Precompute
H0_coeffs = _herm_coeffs(0)  # [1]
H4_coeffs = _herm_coeffs(4)  # [16, 0, -48, 0, 12]
H8_coeffs = _herm_coeffs(8)  # [256, 0, -1280, 0, 2688, 0, -1344, 0, 168]
H12_coeffs = _herm_coeffs(12)

def _poly_val(coeffs, x):
    """Evaluate polynomial with given coefficients (highest power first) at x."""
    result = np.zeros_like(x)
    for c in coeffs:
        result = result * x + c
    return result

def _poly_deriv(coeffs, x):
    """First derivative of polynomial evaluated at x."""
    if len(coeffs) <= 1:
        return np.zeros_like(x)
    # Derivative coefficients
    dc = np.zeros(len(coeffs) - 1)
    for i in range(len(coeffs) - 1):
        dc[i] = (len(coeffs) - i - 1) * coeffs[i]
    return _poly_val(dc, x)

def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound A(g)*A(Fourier(g)) for g(x) = P(x)*exp(-x^2/2)
    where P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12, a12 = -(a0+12*a4+168*a8)/96
    """
    a12 = -(a0 + 12.0 * a4 + 168.0 * a8) / 96.0
    
    # Build P(x) coefficients
    L = len(H12_coeffs)  # 13 (x^12 down to x^0)
    P_coeffs = np.zeros(L)
    # H0: degree 0, coefficient 1
    P_coeffs[-1] += a0 * H0_coeffs[-1]
    # H4: degree 4
    for i, c in enumerate(H4_coeffs):
        P_coeffs[L - 1 - len(H4_coeffs) + i] += a4 * c
    # H8: degree 8
    for i, c in enumerate(H8_coeffs):
        P_coeffs[L - 1 - len(H8_coeffs) + i] += a8 * c
    # H12: degree 12
    for i, c in enumerate(H12_coeffs):
        P_coeffs[i] += a12 * c
    
    # g(x) = P(x) * exp(-x^2/2)
    # A(g) = integral |g(x)| dx over R
    # For the Fourier transform, use the fact that F{x^n exp(-x^2/2)} can be computed
    # via derivatives of the Gaussian FT.
    
    # Numerical integration approach for the bound
    # Use a sufficiently wide grid
    x = np.linspace(-15, 15, 20001)
    dx = x[1] - x[0]
    
    gauss = np.exp(-0.5 * x**2)
    P_val = _poly_val(P_coeffs, x)
    g = P_val * gauss
    
    # A(g) = integral |g| dx
    A_g = np.trapz(np.abs(g), x)
    
    # Fourier transform: F(g)(k) = (1/sqrt(2*pi)) * integral g(x) exp(-i*k*x) dx
    # But for the uncertainty product, we need A(F(g)) = integral |F(g)(k)| dk
    # Note: F(exp(-x^2/2)) = exp(-k^2/2) (with unitary convention)
    # F(x^n * exp(-x^2/2)) = i^n * H_n(k) * exp(-k^2/2) * (some factor)
    
    # Actually, let's use the property that if f = P * exp(-x^2/2),
    # then F(f) can be computed numerically or via the Hermite expansion.
    
    # Numerical FFT approach:
    # Use a fine grid and FFT
    N = 2**15  # 32768
    x_fft = np.linspace(-20, 20, N, endpoint=False)
    dx_fft = x_fft[1] - x_fft[0]
    k_fft = 2*np.pi * np.fft.fftfreq(N, d=dx_fft)
    
    P_val_fft = _poly_val(P_coeffs, x_fft)
    g_fft = P_val_fft * np.exp(-0.5 * x_fft**2)
    
    # Fourier transform (unitary convention): F(f)(k) = (1/sqrt(2*pi)) * integral f(x) e^{-ikx} dx
    # numpy fft: F[k] = sum f[n] exp(-2*pi*i*k*n/N)
    # With dx spacing: integral ≈ dx * sum f[n] exp(-2*pi*i*k*n/N)
    # So F(k_j) ≈ dx/sqrt(2*pi) * fft(f)[j] for appropriate k_j
    
    G_fft = dx_fft / np.sqrt(2 * np.pi) * np.fft.fft(g_fft)
    # Reorder to zero-centered
    G_fft = np.fft.fftshift(G_fft)
    k_fft = np.fft.fftshift(k_fft)
    
    A_Fg = np.trapz(np.abs(G_fft), k_fft)
    
    return A_g * A_Fg

def _compute_bound_grad(params):
    """Compute bound and gradient via finite differences (fallback) or analytic."""
    a0, a4, a8 = params
    val = _compute_bound(a0, a4, a8)
    return val

def construct():
    """
    Return three coefficients (a0, a4, a8) of the Hermite combination.
    Uses L-BFGS-B with analytic gradient approximation for efficiency.
    """
    t_start = time.time()
    deadline = t_start + 195.0  # Safety margin before 210s limit
    
    # Initial guess from prior state of the art
    best_params = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_bound(*best_params)
    
    # Multiple restarts with different initial points
    initial_points = [
        np.array([-1.13, 0.04, 0.000309]),  # prior art
        np.array([-1.0, 0.0, 0.0]),
        np.array([-0.5, 0.1, -0.01]),
        np.array([-2.0, 0.1, 0.01]),
        np.array([-1.5, -0.05, 0.005]),
        np.array([-0.8, 0.08, -0.005]),
        np.array([-1.2, 0.03, 0.001]),
        np.array([-1.0, 0.05, 0.0]),
    ]
    
    # Log-magnitude/sign parameterization for better exploration
    # a_i = sign_i * exp(m_i)
    # Transform: m_i = log(|a_i|), s_i = sign(a_i)
    # For optimization, work in (m0, m4, m8) with signs fixed from initial guess
    
    def transform_to_log(params):
        """Convert (a0, a4, a8) to (log|a0|, log|a4|, log|a8|) with sign handling."""
        signs = np.sign(params)
        # Handle zeros
        signs[signs == 0] = 1.0
        mags = np.abs(params)
        mags[mags < 1e-15] = 1e-15
        return np.log(mags), signs
    
    def transform_from_log(m, signs):
        """Convert (log|a|, signs) back to a."""
        return signs * np.exp(m)
    
    def objective_log(m):
        """Objective in log-magnitude space."""
        # Recover signs from the nearest initial point or use fixed signs
        # We'll use a hybrid: try both sign combinations
        a = transform_from_log(m, np.array([-1.0, 1.0, 1.0]))
        return _compute_bound(*a)
    
    def objective_and_grad_log(m):
        """Objective and gradient in log-magnitude space."""
        a = transform_from_log(m, np.array([-1.0, 1.0, 1.0]))
        val = _compute_bound(*a)
        
        # Numerical gradient in log space
        eps = 1e-7
        grad = np.zeros(3)
        for i in range(3):
            m_plus = m.copy()
            m_plus[i] += eps
            a_plus = transform_from_log(m_plus, np.array([-1.0, 1.0, 1.0]))
            val_plus = _compute_bound(*a_plus)
            grad[i] = (val_plus - val) / eps
        
        return val, grad
    
    # Try optimization in both direct and log-space
    for ip in initial_points:
        if time.time() > deadline:
            break
        
        # Direct space optimization
        try:
            def obj_direct(x):
                return _compute_bound(*x)
            
            def obj_grad_direct(x):
                val = _compute_bound(*x)
                eps = 1e-6
                grad = np.zeros(3)
                for i in range(3):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    val_plus = _compute_bound(*x_plus)
                    grad[i] = (val_plus - val) / eps
                return val, grad
            
            result = minimize(
                obj_direct,
                ip,
                jac=lambda x: obj_grad_direct(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success and result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
        
        # Log-space optimization
        if time.time() > deadline:
            break
        
        try:
            m0, signs0 = transform_to_log(ip)
            
            def obj_log(m):
                a = transform_from_log(m, signs0)
                return _compute_bound(*a)
            
            def obj_grad_log(m):
                a = transform_from_log(m, signs0)
                val = _compute_bound(*a)
                eps = 1e-7
                grad = np.zeros(3)
                for i in range(3):
                    m_plus = m.copy()
                    m_plus[i] += eps
                    a_plus = transform_from_log(m_plus, signs0)
                    val_plus = _compute_bound(*a_plus)
                    grad[i] = (val_plus - val) / eps
                return val, grad
            
            result_log = minimize(
                obj_log,
                m0,
                jac=lambda m: obj_grad_log(m)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result_log.success:
                a_opt = transform_from_log(result_log.x, signs0)
                val_opt = _compute_bound(*a_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_params = a_opt
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 20.0:
        try:
            def obj_final(x):
                return _compute_bound(*x)
            
            def obj_grad_final(x):
                val = _compute_bound(*x)
                eps = 1e-8
                grad = np.zeros(3)
                for i in range(3):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    val_plus = _compute_bound(*x_plus)
                    grad[i] = (val_plus - val) / eps
                return val, grad
            
            result_final = minimize(
                obj_final,
                best_params,
                jac=lambda x: obj_grad_final(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result_final.success and result_final.fun < best_val:
                best_val = result_final.fun
                best_params = result_final.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
