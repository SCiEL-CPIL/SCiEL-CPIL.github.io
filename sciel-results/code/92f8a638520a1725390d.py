"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sympy as sp


def _hermite_poly(n, x):
    """Compute physicist's Hermite polynomial Hn(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _compute_bound(c0, c4, c8, x):
    """Compute the upper bound given coefficients and grid."""
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    
    # g(0) = 0 constraint: c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    c12 = -(c0 * 1 + c4 * 12 + c8 * 1680) / 40320
    
    gauss = np.exp(-x**2 / 2)
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
    
    # Compute norms
    dx = x[1] - x[0]
    norm2 = np.sum(g**2) * dx
    x2_norm = np.sum(x**2 * g**2) * dx
    
    # Fourier transform using FFT
    N = len(x)
    G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))) * dx
    
    # Frequency grid
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    freq2_norm = np.sum(freqs**2 * np.abs(G)**2) * dx
    
    if norm2 <= 0:
        return np.inf
    
    delta_x = np.sqrt(x2_norm / norm2)
    delta_xi = np.sqrt(freq2_norm / norm2)
    
    return delta_x * delta_xi


def _hermite_moment(m, n, power):
    """Compute ∫ x^power H_m(x) H_n(x) exp(-x²) dx analytically."""
    if power == 0:
        if m == n:
            return float(2**n * sp.factorial(n) * sp.sqrt(sp.pi))
        return 0.0
    elif power == 2:
        result = 0.0
        if m + 2 == n:
            result += 0.25 * float(2**n * sp.factorial(n) * sp.sqrt(sp.pi))
        if m == n:
            result += (2 * m + 1) / 2.0 * float(2**n * sp.factorial(n) * sp.sqrt(sp.pi))
        if m >= 2 and m - 2 == n:
            result += m * (m - 1) / 2.0 * float(2**n * sp.factorial(n) * sp.sqrt(sp.pi))
        return result
    else:
        raise ValueError(f"Unsupported power: {power}")


def _compute_bound_analytic(c0, c4, c8):
    """Compute the bound using analytical integrals."""
    c12 = -(c0 + 12 * c4 + 1680 * c8) / 40320.0
    coeffs = [c0, c4, c8, c12]
    degrees = [0, 4, 8, 12]
    
    norm2 = 0.0
    for c, k in zip(coeffs, degrees):
        norm2 += c**2 * float(2**k * sp.factorial(k) * sp.sqrt(sp.pi))
    
    x2_norm = 0.0
    for i, (ci, ki) in enumerate(zip(coeffs, degrees)):
        for j, (cj, kj) in enumerate(zip(coeffs, degrees)):
            x2_norm += ci * cj * _hermite_moment(ki, kj, 2)
    
    freq2_norm = 0.0
    for i, (ci, ki) in enumerate(zip(coeffs, degrees)):
        for j, (cj, kj) in enumerate(zip(coeffs, degrees)):
            phase = (1j) ** (ki - kj)
            freq2_norm += ci * cj * phase * _hermite_moment(ki, kj, 2)
    freq2_norm = float(np.real(freq2_norm))
    
    if norm2 <= 0 or x2_norm <= 0 or freq2_norm <= 0:
        return np.inf
    
    delta_x = np.sqrt(x2_norm / norm2)
    delta_xi = np.sqrt(freq2_norm / norm2)
    
    return delta_x * delta_xi


def construct():
    start_time = time.time()
    deadline = 170  # seconds, keep margin for evaluator kill at 210s
    
    # Setup grid for numerical verification
    x = np.linspace(-20, 20, 4096)
    
    # Initial guess
    c0_init = -113 / 100
    c4_init = 1 / 25
    c8_init = 1 / 3240
    
    best_bound = _compute_bound_analytic(c0_init, c4_init, c8_init)
    best_coeffs = np.array([c0_init, c4_init, c8_init])
    
    # Structural change: Use sympy to find critical points analytically
    # Since the objective is homogeneous, set c0 = 1 and solve for u = c4/c0, v = c8/c0
    try:
        if time.time() - start_time < deadline - 120:
            u, v = sp.symbols('u v', real=True)
            
            coeffs_sym = [sp.Integer(1), u, v, -(sp.Integer(1) + 12 * u + 1680 * v) / sp.Integer(40320)]
            degrees = [0, 4, 8, 12]
            
            norm2_sym = sp.Integer(0)
            for c, k in zip(coeffs_sym, degrees):
                norm2_sym += c**2 * 2**k * sp.factorial(k) * sp.sqrt(sp.pi)
            
            x2_sym = sp.Integer(0)
            for i, (ci, ki) in enumerate(zip(coeffs_sym, degrees)):
                for j, (cj, kj) in enumerate(zip(coeffs_sym, degrees)):
                    term = sp.Integer(0)
                    if ki + 2 == kj:
                        term += sp.Rational(1, 4) * 2**kj * sp.factorial(kj) * sp.sqrt(sp.pi)
                    if ki == kj:
                        term += sp.Rational(2 * ki + 1, 2) * 2**kj * sp.factorial(kj) * sp.sqrt(sp.pi)
                    if ki >= 2 and ki - 2 == kj:
                        term += ki * (ki - 1) / 2 * 2**kj * sp.factorial(kj) * sp.sqrt(sp.pi)
                    x2_sym += ci * cj * term
            
            freq2_sym = sp.Integer(0)
            for i, (ci, ki) in enumerate(zip(coeffs_sym, degrees)):
                for j, (cj, kj) in enumerate(zip(coeffs_sym, degrees)):
                    phase = sp.I ** (ki - kj)
                    term = sp.Integer(0)
                    if ki + 2 == kj:
                        term += sp.Rational(1, 4) * 2**kj * sp.factorial(kj) * sp.sqrt(sp.pi)
                    if ki == kj:
                        term += sp.Rational(2 * ki + 1, 2) * 2**kj * sp.factorial(kj) * sp.sqrt(sp.pi)
                    if ki >= 2 and ki - 2 == kj:
                        term += ki * (ki - 1) / 2 * 2**kj * sp.factorial(kj) * sp.sqrt(sp.pi)
                    freq2_sym += ci * cj * phase * term
            
            freq2_sym = sp.re(freq2_sym)
            
            obj = x2_sym * freq2_sym / norm2_sym**2
            
            grad_u = sp.simplify(sp.diff(obj, u))
            grad_v = sp.simplify(sp.diff(obj, v))
            
            print("Solving critical point equations...")
            sol_start = time.time()
            solutions = sp.solve([grad_u, grad_v], [u, v], dict=True)
            sol_time = time.time() - sol_start
            print(f"Found {len(solutions)} solutions in {sol_time:.1f}s")
            
            for sol in solutions:
                if time.time() - start_time > deadline - 20:
                    break
                try:
                    u_val = float(sp.re(sol[u]))
                    v_val = float(sp.re(sol[v]))
                    
                    if not (np.isfinite(u_val) and np.isfinite(v_val)):
                        continue
                    
                    c0_val = 1.0
                    c4_val = u_val
                    c8_val = v_val
                    
                    bound = _compute_bound_analytic(c0_val, c4_val, c8_val)
                    if bound < best_bound:
                        best_bound = bound
                        best_coeffs = np.array([c0_val, c4_val, c8_val])
                        print(f"New best: {best_bound:.10f} at u={u_val:.6f}, v={v_val:.6f}")
                except Exception:
                    continue
    except Exception as e:
        print(f"Sympy approach failed: {e}")
    
    # Fallback: numerical optimization
    if time.time() - start_time < deadline - 30:
        def objective(coeffs):
            elapsed = time.time() - start_time
            if elapsed > deadline - 10:
                return best_bound
            c0, c4, c8 = coeffs
            return _compute_bound_analytic(c0, c4, c8)
        
        start_points = [
            list(best_coeffs),
            [c0_init, c4_init, c8_init],
            [c0_init * 1.1, c4_init * 0.9, c8_init * 1.2],
        ]
        
        for sp_ in start_points:
            if time.time() - start_time > deadline - 20:
                break
            try:
                result = minimize(objective, sp_, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
                if result.fun < best_bound:
                    best_bound = result.fun
                    best_coeffs = result.x
            except Exception:
                pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
