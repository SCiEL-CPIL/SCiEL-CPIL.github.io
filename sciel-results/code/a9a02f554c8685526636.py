"""Second autocorrelation inequality: maximize value = l2/(norm_1*norm_inf).

Strategy: Low-dimensional DCT coefficient optimization.
We parameterize the 50 heights h as |IDCT(c)| where c is a vector of DCT coefficients.
By restricting the number of non-zero coefficients (or effectively low-pass filtering c),
we constrain h to be smooth and low-complexity, which is likely optimal for such
autocorrelation-based functionals. This reduces the search space from 50 dimensions
to K dimensions (K << 50), making global optimization via Nelder-Mead feasible and
robust. We then polish with L-BFGS-B in the full 50-dim space starting from the
best DCT-reconstructed h.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fftpack import dct, idct

N = 50

def _eval(h):
    """Exact evaluator (mirrors the public contract)."""
    h = np.asarray(h, dtype=float)
    # Ensure non-negative
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] * y[:-1] + y[:-1] * y[1:] + y[1:] * y[1:])
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _dct_to_h(c, K):
    """
    Reconstruct h from DCT coefficients c (first K are active, rest 0).
    c is of length N. We only optimize the first K.
    h = |IDCT(c_padded)|
    """
    c_full = np.zeros(N)
    c_full[:K] = c
    # IDCT normalization: scipy.fftpack.idct uses orthonormal normalization?
    # Actually, scipy.fftpack.dct and idct are inverse of each other with proper scaling.
    # idct(dct(x)) = x.
    # We want h to be non-negative.
    h = np.abs(idct(c_full, type=2))
    # Scale h to have sum 1 (scale invariance)
    s = np.sum(h)
    if s < 1e-12:
        return np.ones(N) / N
    h = h / s
    return h

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin
    
    best_h = np.ones(N)
    best_val = _eval(best_h)
    
    # Try different numbers of DCT coefficients K
    # K=1 corresponds to constant (seed)
    # K=2, 3, ... allow more shape
    Ks = [2, 3, 4, 5, 6, 8, 10]
    
    # For each K, do a multi-start Nelder-Mead in K-dim space
    # Then polish with L-BFGS-B in 50-dim space
    
    rng = np.random.default_rng(50)
    
    for K in Ks:
        if time.time() > deadline - 30.0:
            break
            
        # Multiple random starts in K-dim DCT coefficient space
        # Coefficients can be positive or negative (IDCT can be negative, we take abs)
        # But since we take abs, the sign of the DC component (c[0]) doesn't matter for h shape much,
        # but higher freqs sign flips the wave shape.
        # Let's start with random coefficients in [-1, 1]
        
        n_starts = 10 if K <= 5 else 5
        
        for _ in range(n_starts):
            if time.time() > deadline - 10.0:
                break
                
            # Random start
            c0 = rng.uniform(-1, 1, K)
            
            # Objective for Nelder-Mead: maximize _eval(_dct_to_h(c, K))
            # We minimize -value
            def obj(c):
                h = _dct_to_h(c, K)
                return -_eval(h)
            
            try:
                # Nelder-Mead is derivative-free and good for low-dim
                res = minimize(
                    obj,
                    c0,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                c_opt = res.x
                h_opt = _dct_to_h(c_opt, K)
                v_opt = _eval(h_opt)
                
                if v_opt > best_val:
                    best_val = v_opt
                    best_h = h_opt
                    
                    # Polish with L-BFGS-B in full 50-dim space
                    if time.time() < deadline - 15.0:
                        try:
                            res2 = minimize(
                                lambda h: -_eval(h),
                                h_opt,
                                method='L-BFGS-B',
                                bounds=[(1e-6, 10.0)] * N,
                                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                            )
                            h_polished = res2.x
                            v_polished = _eval(h_polished)
                            if v_polished > best_val:
                                best_val = v_polished
                                best_h = h_polished
                        except Exception:
                            pass
            except Exception:
                pass

    # Final coordinate ascent polish
    if time.time() < deadline - 5.0:
        for _ in range(10):
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                orig = best_h[i]
                for f in [1.2, 0.8, 1.5, 0.5, 1.1, 0.9]:
                    cand = best_h.copy()
                    cand[i] = orig * f
                    if cand[i] < 0: continue
                    v = _eval(cand)
                    if v > best_val:
                        best_val = v
                        best_h = cand
                        improved = True
                        break
            if not improved:
                break

    return list(best_h)
