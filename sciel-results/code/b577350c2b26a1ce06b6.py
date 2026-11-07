import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
DEADLINE = 145.0  # Safety margin before 150s evaluator kill

_start = time.perf_counter()

def _time_left():
    return DEADLINE - (time.perf_counter() - _start)

def compute_value(h):
    """
    Computes the value exactly as defined in the task contract.
    Vectorized for speed.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        return 0.0
    if not np.all(np.isfinite(h)):
        return 0.0
    if np.any(h < 0):
        h = np.maximum(h, 0.0)
    
    # Convolution
    g = np.convolve(h, h, mode='full')
    m = len(g)  # 99
    
    # y = [0, g[0], ..., g[m-1], 0]
    # y has length m + 2
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    # Spacing w
    # Points are np.linspace(-0.5, 0.5, m + 2)
    # So w = 1.0 / (m + 1) = 1.0 / 100
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # y has indices 0..m+1, so i goes from 0 to m
    yi = y[:m+1]
    yip1 = y[1:m+2]
    terms = yi**2 + yi*yip1 + yip1**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    
    return l2 / (norm_1 * norm_inf)

def _objective(x):
    """Negative value for minimization."""
    h = np.maximum(x, 0.0)
    # Normalize to fix scale (scale-invariant)
    s = np.sum(h)
    if s < 1e-12:
        return 1e10
    h_norm = h / s * N  # keep sum = N for numerical stability
    return -compute_value(h_norm)

def _initial_candidates():
    """Generate diverse initial candidates."""
    candidates = []
    idx = np.arange(N)
    center = N / 2.0
    
    # 1. Uniform
    candidates.append(np.ones(N))
    
    # 2. Gaussian with various widths
    for sigma in [3, 5, 7, 10, 15, 20, 25]:
        candidates.append(np.exp(-((idx - center)**2) / (2 * sigma**2)))
    
    # 3. Super-Gaussian (flatter top)
    for p in [4, 6, 8, 12]:
        candidates.append(np.exp(-((idx - center)**2) / (2 * 10.0**2))**0.5)  # sqrt of gaussian = wider
        candidates.append(np.maximum(1 - ((idx - center) / 15.0)**p, 0))
    
    # 4. Asymmetric Gaussians
    for shift in [-5, -3, 3, 5]:
        c = center + shift
        for sigma in [5, 8, 12]:
            candidates.append(np.exp(-((idx - c)**2) / (2 * sigma**2)))
    
    # 5. Trapezoid
    for half_width in [5, 10, 15, 20]:
        for rise in [2, 5, 10]:
            h = np.zeros(N)
            left_plateau = center - half_width
            right_plateau = center + half_width
            h[(idx >= left_plateau - rise) & (idx < left_plateau)] = (idx - (left_plateau - rise)) / rise
            h[(idx >= left_plateau) & (idx <= right_plateau)] = 1.0
            h[(idx > right_plateau) & (idx <= right_plateau + rise)] = ((right_plateau + rise) - idx) / rise
            h = np.maximum(h, 0)
            if np.sum(h) > 0:
                candidates.append(h)
    
    # 6. Tent / Triangle
    for width in [5, 10, 15, 20, 25]:
        h = np.maximum(1 - np.abs(idx - center) / width, 0)
        candidates.append(h)
    
    # 7. Cosine bumps
    for width in [5, 10, 15, 20, 25]:
        h = np.maximum(np.cos(np.pi * (idx - center) / width), 0)
        candidates.append(h)
        h2 = np.maximum(np.cos(np.pi * (idx - center) / width)**2, 0)
        candidates.append(h2)
    
    # 8. Bimodal
    for sep in [5, 10, 15]:
        for sigma in [3, 5, 8]:
            h = np.exp(-((idx - (center - sep))**2) / (2 * sigma**2)) + np.exp(-((idx - (center + sep))**2) / (2 * sigma**2))
            candidates.append(h)
    
    # 9. Power law
    for alpha in [1, 2, 3, 5]:
        h = np.maximum(1 - np.abs(idx - center) / 20.0, 0)**alpha
        candidates.append(h)
    
    # 10. Step functions
    for k in [5, 10, 15, 20]:
        h = np.zeros(N)
        h[N//2 - k:N//2 + k] = 1.0
        candidates.append(h)
        # Two steps
        h2 = np.zeros(N)
        h2[N//2 - k-2:N//2 - 2] = 1.0
        h2[N//2 + 2:N//2 + k + 2] = 1.0
        candidates.append(h2)
    
    # 11. Alternating / oscillatory
    for freq in [1, 2, 3, 5]:
        h = 0.5 + 0.5 * np.cos(2 * np.pi * freq * idx / N)
        candidates.append(np.maximum(h, 0))
        h = 0.5 + 0.5 * np.sin(2 * np.pi * freq * idx / N)
        candidates.append(np.maximum(h, 0))
    
    # 12. Random with smoothing
    rng = np.random.RandomState(42)
    for _ in range(20):
        h = rng.rand(N)
        # Smooth
        kernel = np.ones(5) / 5
        h = np.convolve(h, kernel, mode='same')
        h = np.maximum(h, 0)
        candidates.append(h)
    
    # 13. Spike pairs
    for pos in [5, 10, 15]:
        for width in [1, 2, 3]:
            h = np.zeros(N)
            h[pos:pos+width] = 1.0
            h[N-1-pos-width:N-1-pos] = 1.0
            candidates.append(h)
    
    # 14. Plateau with tails
    for plateau in [5, 10, 15]:
        for tail in [5, 10, 15]:
            h = np.zeros(N)
            h[N//2 - plateau:N//2 + plateau] = 1.0
            # Add exponential tails
            for i in range(N//2 - plateau - tail, N//2 - plateau):
                if i >= 0:
                    h[i] = np.exp(-(N//2 - plateau - i) / 5.0)
            for i in range(N//2 + plateau, N//2 + plateau + tail):
                if i < N:
                    h[i] = np.exp(-(i - (N//2 + plateau)) / 5.0)
            candidates.append(h)
    
    # 15. DCT-based candidates
    for k in [1, 2, 3, 4, 5, 7, 10]:
        h = np.abs(np.cos(np.pi * k * (idx + 0.5) / N))
        candidates.append(h)
    
    return candidates

def construct():
    # Seed result
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    candidates = _initial_candidates()
    
    # First pass: evaluate all candidates
    for h0 in candidates:
        if _time_left() < 5:
            break
        h0 = np.maximum(h0, 0.0)
        if np.sum(h0) < 1e-12:
            continue
        val = compute_value(h0)
        if val > best_val:
            best_val = val
            best_h = h0.copy()
    
    # Second pass: local optimization from best candidates
    # Sort candidates by value
    cand_vals = []
    for h0 in candidates:
        h0 = np.maximum(h0, 0.0)
        if np.sum(h0) < 1e-12:
            continue
        val = compute_value(h0)
        cand_vals.append((val, h0))
    
    cand_vals.sort(key=lambda x: -x[0])
    
    # Optimize top candidates
    top_k = min(15, len(cand_vals))
    for val, h0 in cand_vals[:top_k]:
        if _time_left() < 10:
            break
        h0 = np.maximum(h0, 0.0)
        if np.sum(h0) < 1e-12:
            continue
        
        # L-BFGS-B
        try:
            res = minimize(_objective, h0, method='L-BFGS-B',
                           bounds=[(0.0, None)] * N,
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            h_opt = np.maximum(res.x, 0.0)
            if np.sum(h_opt) > 1e-12:
                v = compute_value(h_opt)
                if v > best_val:
                    best_val = v
                    best_h = h_opt.copy()
        except Exception:
            pass
        
        # Powell method
        if _time_left() < 10:
            break
        try:
            res2 = minimize(_objective, h_opt if 'h_opt' in locals() else h0, method='Powell',
                            bounds=[(0.0, None)] * N,
                            options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-10})
            h_pow = np.maximum(res2.x, 0.0)
            if np.sum(h_pow) > 1e-12:
                v = compute_value(h_pow)
                if v > best_val:
                    best_val = v
                    best_h = h_pow.copy()
        except Exception:
            pass
    
    # Coordinate descent on best
    if _time_left() > 15:
        h_cd = best_h.copy()
        for _ in range(3):
            if _time_left() < 10:
                break
            improved = True
            while improved and _time_left() > 10:
                improved = False
                for i in range(N):
                    # Try adjusting h[i]
                    old_h = h_cd[i]
                    # Try a few perturbations
                    for delta in [0.1, 0.05, 0.02, -0.1, -0.05, -0.02]:
                        new_h = h_cd.copy()
                        new_h[i] = max(0, old_h + delta)
                        if np.sum(new_h) < 1e-12:
                            continue
                        v = compute_value(new_h)
                        if v > best_val + 1e-10:
                            best_val = v
                            best_h = new_h.copy()
                            h_cd = new_h.copy()
                            improved = True
                            break
                    # Also try scaling
                    for scale in [1.1, 0.9, 1.05, 0.95]:
                        new_h = h_cd.copy()
                        new_h[i] = max(0, old_h * scale)
                        if np.sum(new_h) < 1e-12:
                            continue
                        v = compute_value(new_h)
                        if v > best_val + 1e-10:
                            best_val = v
                            best_h = new_h.copy()
                            h_cd = new_h.copy()
                            improved = True
                            break
    
    # Final: differential evolution on a reduced set if time allows
    if _time_left() > 30:
        try:
            # Reduce dimensionality by optimizing in a subspace
            # Use the best shape and perturb it
            base = best_h.copy()
            base = base / np.sum(base) * N
            
            def _reduced_objective(params):
                # params: (center, width, height, asymmetry)
                center, width, height, asym = params
                idx = np.arange(N)
                h = height * np.exp(-((idx - center)**2) / (2 * width**2))
                # Add asymmetry
                if asym > 0:
                    h[idx > center] *= np.exp(-asym * (idx[idx > center] - center))
                else:
                    h[idx <= center] *= np.exp(asym * (center - idx[idx <= center]))
                h = np.maximum(h, 0)
                s = np.sum(h)
                if s < 1e-12:
                    return 1e10
                return -compute_value(h / s * N)
            
            bounds_reduced = [(5, 45), (1, 30), (0.1, 5), (-1, 1)]
            res_de = differential_evolution(_reduced_objective, bounds_reduced,
                                            seed=42, maxiter=100, tol=1e-8,
                                            polish=True)
            center, width, height, asym = res_de.x
            idx = np.arange(N)
            h_de = height * np.exp(-((idx - center)**2) / (2 * width**2))
            if asym > 0:
                h_de[idx > center] *= np.exp(-asym * (idx[idx > center] - center))
            else:
                h_de[idx <= center] *= np.exp(asym * (center - idx[idx <= center]))
            h_de = np.maximum(h_de, 0)
            s = np.sum(h_de)
            if s > 1e-12:
                h_de = h_de / s * N
                v = compute_value(h_de)
                if v > best_val:
                    best_val = v
                    best_h = h_de.copy()
        except Exception:
            pass
    
    return best_h.tolist()
