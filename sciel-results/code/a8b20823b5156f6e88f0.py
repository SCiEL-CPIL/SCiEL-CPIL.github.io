"""
Design: Low-rank DCT representation of step heights.
Strategy: Use Nelder-Mead optimization on the DCT coefficients.
Reasoning: The non-negativity constraint on h creates non-smooth boundaries in the
coefficient space. Gradient-based methods (L-BFGS-B) can struggle with these.
Nelder-Mead is derivative-free, robust to non-smoothness, and efficient for the
low dimensionality (K ~ 10-12) of the DCT coefficient space.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)

# Number of DCT basis components to use
K = 12

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    """Exact evaluator metric for given heights."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _dct_decode(coeffs):
    """Decode DCT-II coefficients back to heights."""
    n = N
    m = np.arange(n)
    h = np.zeros(n)
    for k_idx in range(len(coeffs)):
        scale = np.sqrt(1.0 / n) if k_idx == 0 else np.sqrt(2.0 / n)
        h += coeffs[k_idx] * scale * np.cos(np.pi / n * (m + 0.5) * k_idx)
    return h

def _objective(coeffs):
    """Objective function for minimization (negative value)."""
    h = _dct_decode(coeffs)
    h = np.maximum(0.0, h)
    v = compute_value(h)
    return -v

def _generate_starts():
    """Generate diverse starting points in DCT coefficient space."""
    starts = []
    
    # 1. Uniform (DC only)
    h_uniform = np.ones(N)
    # Compute DCT coeffs for uniform
    c = np.zeros(K)
    c[0] = np.sqrt(N) # Since orthonormal DCT of ones is sqrt(N) at k=0
    # Actually, let's just use a helper to encode
    def encode(h):
        n = N
        m = np.arange(n)
        coeffs = np.zeros(K)
        for k_idx in range(K):
            scale = np.sqrt(1.0 / n) if k_idx == 0 else np.sqrt(2.0 / n)
            coeffs[k_idx] = np.sum(h * np.cos(np.pi / n * (m + 0.5) * k_idx)) / scale
        return coeffs

    starts.append(encode(h_uniform))
    
    # 2. Gaussian bump
    x = np.linspace(0, N - 1, N)
    h_gauss = np.exp(-0.5 * ((x - N / 2) / (N / 5)) ** 2)
    starts.append(encode(h_gauss))
    
    # 3. Semicircle
    h_semi = np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2))
    starts.append(encode(h_semi))
    
    # 4. Triangle
    h_tri = 1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1)
    starts.append(encode(h_tri))
    
    # 5. Cosine bump
    h_cos = 0.5 * (1 + np.cos(2 * np.pi * x / N))
    starts.append(encode(h_cos))

    # 6. Random smooth profiles
    for _ in range(30):
        coeffs = np.zeros(K)
        coeffs[0] = RNG.uniform(0.5, 1.5)
        for k in range(1, K):
            coeffs[k] = RNG.uniform(-0.3, 0.3)
        starts.append(coeffs)
        
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 10.0)  # 10s safety margin

    best_val = 0.0
    best_h = np.ones(N)
    
    starts = _generate_starts()
    
    # Initial evaluation of starts
    for s in starts:
        h = _dct_decode(s)
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # Optimization Phase
    # Use Nelder-Mead for robustness against non-smoothness from max(0, h)
    
    # We will run Nelder-Mead from the best starts and some random ones
    # To save time, we limit the number of full optimizations.
    
    # Sort starts by initial value to prioritize good ones
    start_vals = []
    for s in starts:
        h = _dct_decode(s)
        h = np.maximum(0.0, h)
        v = compute_value(h)
        start_vals.append((v, s))
        
    start_vals.sort(reverse=True)
    
    # Run optimization on top 10 starts
    for i, (v, s) in enumerate(start_vals[:10]):
        if time.time() > deadline:
            break
        
        try:
            res = opt.minimize(
                _objective,
                s,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            h = _dct_decode(res.x)
            h = np.maximum(0.0, h)
            v_res = compute_value(h)
            if v_res > best_val:
                best_val = v_res
                best_h = h.copy()
        except Exception:
            pass

    # Polish Phase: Refine the best solution with tighter tolerances
    if time.time() < deadline:
        # Get coefficients for best_h
        def encode(h):
            n = N
            m = np.arange(n)
            coeffs = np.zeros(K)
            for k_idx in range(K):
                scale = np.sqrt(1.0 / n) if k_idx == 0 else np.sqrt(2.0 / n)
                coeffs[k_idx] = np.sum(h * np.cos(np.pi / n * (m + 0.5) * k_idx)) / scale
            return coeffs
        
        best_coeffs = encode(best_h)
        
        # Run a few more Nelder-Mead steps from the best point with very tight tolerances
        for _ in range(5):
            if time.time() > deadline:
                break
            try:
                res = opt.minimize(
                    _objective,
                    best_coeffs,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                h = _dct_decode(res.x)
                h = np.maximum(0.0, h)
                v_res = compute_value(h)
                if v_res > best_val:
                    best_val = v_res
                    best_h = h.copy()
                    best_coeffs = res.x.copy()
            except Exception:
                break

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
