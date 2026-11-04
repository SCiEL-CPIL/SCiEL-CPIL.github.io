import numpy as np
from scipy.optimize import minimize
import time


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


def construct():
    start_time = time.time()
    deadline = 170  # seconds, keep margin for evaluator kill at 210s

    # Chebyshev-based approach: represent g(x) as a Chebyshev series on [-L, L]
    # g(x) = sum_{k=0}^{K} a_k T_k(x/L) * exp(-x^2/2)
    # We optimize over the coefficients a_k subject to g(0) = 0
    
    L = 8.0  # interval half-width
    K = 12   # Chebyshev order
    
    # Chebyshev nodes on [-1, 1]
    N = 4096
    t = np.cos(np.pi * (np.arange(N) + 0.5) / N)  # Chebyshev-Lobatto-like nodes
    x = L * t
    dx = x[1] - x[0]
    
    # FFT setup
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    # Precompute Chebyshev polynomials at the nodes
    T = np.zeros((K + 1, N))
    T[0] = 1.0
    if K >= 1:
        T[1] = t
    for k in range(2, K + 1):
        T[k] = 2 * t * T[k-1] - T[k-2]
    
    gauss = np.exp(-x**2 / 2)
    
    # Effective basis: phi_k(x) = T_k(x/L) * gauss
    # Constraint: g(0) = 0 => sum a_k T_k(0) * gauss(0) = 0
    # T_k(0) = cos(k*pi/2) for k even, 0 for k odd
    # So constraint: a_0 + a_2*T_2(0) + a_4*T_4(0) + ... = 0
    # T_0(0)=1, T_2(0)=-1, T_4(0)=1, T_6(0)=-1, etc.
    
    # We'll parameterize with free coefficients and solve for one
    # Use a_0 as the dependent variable: a_0 = -sum_{k=2,4,...} a_k T_k(0)
    
    # Build the constraint matrix
    # For even k: T_k(0) = (-1)^(k/2)
    constraint_coeff = np.zeros(K + 1)
    for k in range(0, K + 1, 2):
        constraint_coeff[k] = (-1) ** (k // 2)
    
    # Free coefficients: a_1, a_2, ..., a_K (K free variables)
    # a_0 is determined by the constraint
    
    def compute_bound(a_free):
        """Compute uncertainty product for Chebyshev coefficients."""
        a = np.zeros(K + 1)
        a[1:] = a_free
        # Enforce g(0) = 0: a_0 * T_0(0) + sum_{k>=2 even} a_k T_k(0) = 0
        # a_0 = -sum_{k>=2 even} a_k T_k(0)
        a[0] = -np.sum(a[2::2] * constraint_coeff[2::2])
        
        # Compute g(x) = sum a_k T_k(x/L) * gauss
        g = T @ a * gauss
        
        # Norm
        norm2 = np.sum(g**2) * dx
        if norm2 <= 1e-30 or not np.isfinite(norm2):
            return np.inf
        
        # Position variance: <x^2> - <x>^2
        x_mean = np.sum(x * g**2) * dx / norm2
        x2_mean = np.sum(x**2 * g**2) * dx / norm2
        delta_x2 = x2_mean - x_mean**2
        
        # Frequency variance via FFT
        G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))) * dx
        freq2_weighted = np.sum(freqs**2 * np.abs(G)**2) * dx
        delta_xi2 = freq2_weighted / norm2
        
        if not np.isfinite(delta_x2) or delta_x2 < 0:
            return np.inf
        if not np.isfinite(delta_xi2) or delta_xi2 < 0:
            return np.inf
        
        return np.sqrt(delta_x2) * np.sqrt(delta_xi2)
    
    # Initial guess: try a few simple configurations
    best_bound = np.inf
    best_a = np.zeros(K)
    
    # Try several initial configurations
    rng = np.random.RandomState(42)
    
    # Configuration 1: mostly a_2 and a_4
    init1 = np.zeros(K)
    init1[1] = 1.0  # a_2
    init1[3] = -0.5  # a_4
    for init in [init1, rng.normal(0, 0.5, size=K)]:
        b = compute_bound(init)
        if b < best_bound:
            best_bound = b
            best_a = init.copy()
    
    # Powell optimization
    if time.time() - start_time < deadline - 30:
        try:
            res = minimize(
                compute_bound,
                best_a,
                method='Powell',
                options={'xtol': 1e-12, 'ftol': 1e-14, 'maxiter': 500, 'maxfev': 5000}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_a = res.x.copy()
        except Exception:
            pass
    
    # Second round with different starting points
    if time.time() - start_time < deadline - 20:
        for _ in range(5):
            if time.time() - start_time > deadline - 15:
                break
            perturbation = rng.normal(0, 0.3, size=K)
            start = best_a + perturbation
            try:
                res = minimize(
                    compute_bound,
                    start,
                    method='Powell',
                    options={'xtol': 1e-12, 'ftol': 1e-14, 'maxiter': 300, 'maxfev': 3000}
                )
                if res.fun < best_bound:
                    best_bound = res.fun
                    best_a = res.x.copy()
            except Exception:
                pass
    
    # Convert Chebyshev coefficients back to the required Hermite combination format
    # We need to express g(x) as c0*H0 + c4*H4 + c8*H8 + c12*H12 (with g(0)=0)
    # This is an approximation: project the Chebyshev-based g onto the Hermite basis
    
    # Compute the optimal g from Chebyshev coefficients
    a = np.zeros(K + 1)
    a[1:] = best_a
    a[0] = -np.sum(a[2::2] * constraint_coeff[2::2])
    g_opt = T @ a * gauss
    
    # Project onto Hermite basis {H0, H4, H8, H12} with g(0)=0 constraint
    # Use the same effective basis as before
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    
    v0 = (H0 - H12 / 10395.0) * gauss
    v4 = (H4 - 3.0 * H12 / 10395.0) * gauss
    v8 = (H8 - 105.0 * H12 / 10395.0) * gauss
    
    # Solve least squares: g_opt ≈ c0*v0 + c4*v4 + c8*v8
    V = np.column_stack([v0, v4, v8])
    # Weighted least squares with weight 1/dx
    W = np.ones(N) / dx
    VtWV = V.T @ (W[:, None] * V)
    VtWg = V.T @ (W * g_opt)
    
    try:
        c_hermite = np.linalg.solve(VtWV, VtWg)
    except np.linalg.LinAlgError:
        c_hermite = np.linalg.lstsq(VtWV, VtWg, rcond=None)[0]
    
    # Verify the bound with the Hermite representation
    def compute_bound_hermite(c):
        c = np.asarray(c, dtype=float)
        c12 = -(c[0] * 1 + c[1] * 3 + c[2] * 105) / 10395.0
        g = (c[0]*H0 + c[1]*H4 + c[2]*H8 + c12*H12) * gauss
        norm2 = np.sum(g**2) * dx
        if norm2 <= 1e-30:
            return np.inf
        x_mean = np.sum(x * g**2) * dx / norm2
        x2_mean = np.sum(x**2 * g**2) * dx / norm2
        delta_x2 = x2_mean - x_mean**2
        G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))) * dx
        delta_xi2 = np.sum(freqs**2 * np.abs(G)**2) * dx / norm2
        if delta_x2 < 0 or delta_xi2 < 0:
            return np.inf
        return np.sqrt(delta_x2) * np.sqrt(delta_xi2)
    
    final_bound = compute_bound_hermite(c_hermite)
    
    # If the projection is bad, fall back to direct optimization in Hermite space
    if not np.isfinite(final_bound) or final_bound > best_bound * 1.1:
        # Direct optimization in Hermite space
        def compute_bound_direct(c):
            c = np.asarray(c, dtype=float)
            c12 = -(c[0] * 1 + c[1] * 3 + c[2] * 105) / 10395.0
            g = (c[0]*H0 + c[1]*H4 + c[2]*H8 + c12*H12) * gauss
            norm2 = np.sum(g**2) * dx
            if norm2 <= 1e-30:
                return np.inf
            x_mean = np.sum(x * g**2) * dx / norm2
            x2_mean = np.sum(x**2 * g**2) * dx / norm2
            delta_x2 = x2_mean - x_mean**2
            G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))) * dx
            delta_xi2 = np.sum(freqs**2 * np.abs(G)**2) * dx / norm2
            if delta_x2 < 0 or delta_xi2 < 0:
                return np.inf
            return np.sqrt(delta_x2) * np.sqrt(delta_xi2)
        
        c_start = np.array([-1.13, 0.04, 0.0003])
        if time.time() - start_time < deadline - 10:
            try:
                res = minimize(
                    compute_bound_direct,
                    c_start,
                    method='Powell',
                    options={'xtol': 1e-14, 'ftol': 1e-16, 'maxiter': 1000, 'maxfev': 10000}
                )
                if res.fun < final_bound:
                    c_hermite = res.x
            except Exception:
                pass
    
    return np.array(c_hermite)
