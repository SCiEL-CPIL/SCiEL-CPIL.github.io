"""Optimized search for flattest ±1 polynomial with 24 coefficients."""
import numpy as np
import time
import random

_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin before 150s kill

N = 24
SQRTN = np.sqrt(N)
_KS = np.arange(N)
_TAU = 2.0 * np.pi

def _eval_poly(coeffs, theta):
    """Evaluate |P(e^{i theta})|."""
    z = np.exp(1j * theta)
    return float(np.abs(np.dot(coeffs, z ** _KS)))

def _supnorm_grid(coeffs, n_grid=4096):
    """Fast sup-norm estimate using FFT on a uniform grid."""
    padded = np.zeros(n_grid, dtype=float)
    padded[:N] = coeffs
    vals = np.fft.fft(padded)
    return float(np.max(np.abs(vals)))

def _refine_peak(coeffs, ang, iters=50):
    """Golden-section refine max of |P(e^{i theta})| near ang."""
    # Initial bracket: small window around ang
    delta = 0.01
    a = ang - delta
    b = ang + delta
    # Ensure we're bracketing a local max
    fa = _eval_poly(coeffs, a)
    fb = _eval_poly(coeffs, b)
    fmid = _eval_poly(coeffs, ang)
    
    # Expand if needed
    for _ in range(20):
        if fmid > fa and fmid > fb:
            break
        delta *= 2
        if delta > 0.5:
            break
        a = ang - delta
        b = ang + delta
        fa = _eval_poly(coeffs, a)
        fb = _eval_poly(coeffs, b)
        fmid = _eval_poly(coeffs, ang)
    
    # Golden section search for maximum
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = _eval_poly(coeffs, c)
    fd = _eval_poly(coeffs, d)
    
    for _ in range(iters):
        if fc > fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = _eval_poly(coeffs, c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = _eval_poly(coeffs, d)
    
    mid = (a + b) / 2.0
    return float(max(_eval_poly(coeffs, a), _eval_poly(coeffs, b), _eval_poly(coeffs, mid)))

def _supnorm_refined(coeffs, n_grid=2048):
    """Sup-norm with grid scan + refinement of top peaks."""
    padded = np.zeros(n_grid, dtype=float)
    padded[:N] = coeffs
    vals = np.fft.fft(padded)
    mags = np.abs(vals)
    
    # Get top 10 peaks
    top_idx = np.argsort(mags)[-10:]
    top_idx = np.sort(top_idx)
    
    best = 0.0
    for idx in top_idx:
        theta = (idx / n_grid) * _TAU
        refined = _refine_peak(coeffs, theta, iters=40)
        if refined > best:
            best = refined
    return best

def _flip_one(coeffs, i):
    """Return copy of coeffs with coefficient i flipped."""
    new = coeffs.copy()
    new[i] = -new[i]
    return new

def _local_search_1flip(coeffs, max_rounds=500):
    """Iterative 1-flip local search until convergence."""
    current = coeffs.copy()
    current_score = _supnorm_refined(current)
    
    for _ in range(max_rounds):
        if time.time() > _DEADLINE - 5.0:
            break
        best_flip = -1
        best_score = current_score
        for i in range(N):
            candidate = _flip_one(current, i)
            score = _supnorm_refined(candidate)
            if score < best_score - 1e-12:
                best_score = score
                best_flip = i
        if best_flip >= 0:
            current[best_flip] = -current[best_flip]
            current_score = best_score
        else:
            break
    
    return current, current_score

def _local_search_2flip_escape(coeffs, n_escapes=20):
    """Try 2-flip moves to escape local minima."""
    current = coeffs.copy()
    current_score = _supnorm_refined(current)
    
    for _ in range(n_escapes):
        if time.time() > _DEADLINE - 5.0:
            break
        # Pick two random positions to flip
        i, j = random.sample(range(N), 2)
        candidate = current.copy()
        candidate[i] = -candidate[i]
        candidate[j] = -candidate[j]
        candidate_score = _supnorm_refined(candidate)
        
        if candidate_score < current_score:
            current = candidate
            current_score = candidate_score
            # After escape, do 1-flip local search
            current, current_score = _local_search_1flip(current, max_rounds=100)
    
    return current, current_score

def _generate_golay_rudin_shapiro():
    """Generate Golay-Rudin-Shapiro polynomial truncated to 24 coefficients."""
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    return p[:N]

def _random_poly():
    """Generate random ±1 polynomial."""
    return np.random.choice([-1, 1], size=N).astype(float)

def construct():
    """Main search: multiple starts with local search."""
    random.seed(42)
    np.random.seed(42)
    
    best_poly = None
    best_score = float('inf')
    
    # Start with Golay-Rudin-Shapiro
    candidates = []
    grs = _generate_golay_rudin_shapiro()
    candidates.append(grs)
    
    # Add some random starts
    for _ in range(15):
        candidates.append(_random_poly())
    
    for start in candidates:
        if time.time() > _DEADLINE - 10.0:
            break
        
        # 1-flip local search
        poly, score = _local_search_1flip(start)
        
        if score < best_score:
            best_score = score
            best_poly = poly.copy()
        
        # 2-flip escape
        poly2, score2 = _local_search_2flip_escape(poly, n_escapes=10)
        if score2 < best_score:
            best_score = score2
            best_poly = poly2.copy()
    
    # Final refinement: do more 1-flip iterations on the best
    if best_poly is not None and time.time() < _DEADLINE - 5.0:
        best_poly, best_score = _local_search_1flip(best_poly, max_rounds=200)
    
    return best_poly.tolist()
