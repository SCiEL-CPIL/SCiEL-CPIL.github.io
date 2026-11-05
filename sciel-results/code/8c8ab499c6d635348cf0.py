"""Combined approach: Chebyshev basis + differential evolution for autocorrelation minimization."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400
_DEADLINE = 145.0  # safety margin under 150s limit

# Precompute FFT size for fast autoconvolution
_FFT_SIZE = 1 << (2 * N - 1).bit_length()  # next power of 2 >= 2N-1
_FFT_SIZE = max(_FFT_SIZE, 1024)

def _value(h):
    """Compute the objective value (lower is better)."""
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s * s))


def _fast_autoconv(h):
    """Fast autoconvolution using FFT."""
    H = np.fft.rfft(h, _FFT_SIZE)
    C = np.fft.irfft(H * H, _FFT_SIZE)
    return C[:2 * N - 1]


def _fast_value(h):
    """Compute objective using FFT-based autoconvolution."""
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    c = _fast_autoconv(h)
    return abs(2.0 * N * np.max(c) / (s * s))


def _cheb_basis(k, n_samples=N):
    """
    Generate Chebyshev polynomial basis vectors on [-1, 1].
    T_0 = 1, T_1 = x, T_k = 2x*T_{k-1} - T_{k-2}
    """
    x = np.linspace(-1, 1, n_samples)
    basis = np.zeros((k, n_samples))
    basis[0] = 1.0
    if k > 1:
        basis[1] = x
    for j in range(2, k):
        basis[j] = 2.0 * x * basis[j-1] - basis[j-2]
    return basis


# Precompute Chebyshev bases for various K values
_CHEB_BASIS_CACHE = {}

def _get_cheb_basis(K):
    """Get or compute Chebyshev basis for given K."""
    if K not in _CHEB_BASIS_CACHE:
        _CHEB_BASIS_CACHE[K] = _cheb_basis(K, N)
    return _CHEB_BASIS_CACHE[K]


def _make_h_from_cheb(coeffs, K):
    """
    Construct h as linear combination of Chebyshev polynomials.
    h = B_K @ coeffs where B_K is the Chebyshev basis matrix.
    """
    B = _get_cheb_basis(K)
    return B.T @ coeffs


def _obj_cheb(coeffs, K):
    """Objective function for Chebyshev coefficient optimization."""
    h = _make_h_from_cheb(coeffs, K)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e12
    c = _fast_autoconv(h)
    max_c = np.max(c)
    val = 2.0 * N * max_c / (s * s)
    return abs(val)


def _soft_max(c, beta=10.0):
    """Smooth approximation of max using log-sum-exp."""
    c_max = np.max(c)
    if c_max < 0:
        # All negative, max is the least negative
        return c_max
    # log-sum-exp trick for stability
    shifted = c - c_max
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return c_max + log_sum


def _obj_cheb_smooth(coeffs, K, beta=10.0):
    """Smooth objective using soft-max surrogate."""
    h = _make_h_from_cheb(coeffs, K)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e12
    c = _fast_autoconv(h)
    max_c = _soft_max(c, beta)
    val = 2.0 * N * max_c / (s * s)
    return abs(val)


def construct():
    """Construct step heights minimizing the objective value."""
    start = time.time()
    
    best_h = np.ones(N)
    best_val = 2.0
    
    def _update(h):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h)):
            return
        v = _fast_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Try different numbers of Chebyshev basis functions
    for K in [3, 5, 7, 10, 15]:
        if time.time() - start > _DEADLINE - 15:
            break
        
        # Multiple initializations for diversity
        inits = [
            np.zeros(K),  # will set [0]=1 below
            np.random.default_rng(42 + K).uniform(-0.5, 0.5, K),
            np.random.default_rng(100 + K).normal(0, 0.3, K),
        ]
        for init in inits:
            init = init.copy()
            init[0] = 1.0  # ensure nonzero sum
            
            bounds = [(0.1, 10.0)] + [(-5.0, 5.0)] * (K - 1)
            
            dim = K
            remaining = _DEADLINE - (time.time() - start)
            if remaining < 5:
                break
            
            maxiter = max(30, int(remaining * 1.5))
            popsize = max(10, min(40, 35 - K))
            
            try:
                res = differential_evolution(
                    _obj_cheb, bounds, args=(K,),
                    x0=init,
                    maxiter=maxiter, popsize=popsize,
                    tol=0.005, mutation=(0.5, 1.0), recombination=0.7,
                    seed=42 + K, polish=False, init='sobol' if dim <= 20 else 'latinhypercube'
                )
                if res.x is not None:
                    h = _make_h_from_cheb(res.x, K)
                    _update(h)
                    
                    # Local polish with smooth surrogate first, then exact
                    if time.time() - start < _DEADLINE - 10:
                        try:
                            r2 = minimize(_obj_cheb_smooth, res.x, args=(K, 10.0),
                                          method='Nelder-Mead',
                                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                            if r2.x is not None:
                                h2 = _make_h_from_cheb(r2.x, K)
                                _update(h2)
                        except Exception:
                            pass
                        
                        try:
                            r3 = minimize(_obj_cheb, res.x, args=(K,),
                                          method='Nelder-Mead',
                                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
                            if r3.x is not None:
                                h3 = _make_h_from_cheb(r3.x, K)
                                _update(h3)
                        except Exception:
                            pass
            except Exception:
                pass
    
    # Fourier approach as backup
    if time.time() - start < _DEADLINE - 5:
        n = np.arange(N)
        cos_basis = []
        sin_basis = []
        for j in range(1, 7):
            cos_basis.append(np.cos(2 * np.pi * j * n / N))
            sin_basis.append(np.sin(2 * np.pi * j * n / N))
        cos_mat = np.column_stack(cos_basis)
        sin_mat = np.column_stack(sin_basis)
        
        def _fourier_h(coeffs, M):
            h = np.full(N, 1.0)
            for j in range(M):
                a = coeffs[2 * j]
                b = coeffs[2 * j + 1]
                h += a * cos_mat[:, j] + b * sin_mat[:, j]
            return h
        
        def _fourier_obj(coeffs, M):
            h = _fourier_h(coeffs, M)
            s = np.sum(h)
            if abs(s) < 1e-12:
                return 1e12
            c = _fast_autoconv(h)
            return abs(2.0 * N * np.max(c) / (s * s))
        
        for M in [2, 3, 4, 6]:
            if time.time() - start > _DEADLINE - 3:
                break
            dim = 2 * M
            bounds = [(-1.5, 1.5)] * dim
            remaining = _DEADLINE - (time.time() - start)
            if remaining < 3:
                break
            try:
                res_f = differential_evolution(
                    _fourier_obj, bounds, args=(M,),
                    maxiter=max(50, int(remaining * 2)), popsize=max(10, min(30, 25)),
                    tol=0.005, seed=123 + M, polish=False
                )
                if res_f.x is not None:
                    h_f = _fourier_h(res_f.x, M)
                    _update(h_f)
                    
                    if time.time() - start < _DEADLINE - 5:
                        try:
                            r_f = minimize(_fourier_obj, res_f.x, args=(M,),
                                           method='Nelder-Mead',
                                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
                            if r_f.x is not None:
                                h_f2 = _fourier_h(r_f.x, M)
                                _update(h_f2)
                        except Exception:
                            pass
            except Exception:
                pass
    
    return best_h
