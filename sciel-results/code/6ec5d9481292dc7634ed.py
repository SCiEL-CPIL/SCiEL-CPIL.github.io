"""Signed autocorrelation maximization: sparse signed sequence with local sign-flip optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

_START_TIME = time.time()
_TIME_LIMIT = 115.0


def _autoconv_max_abs(h):
    """Compute max |autoconvolution| of h."""
    ac = fftconvolve(h, h, mode='full')
    return np.max(np.abs(ac))


def _squared_integral(h):
    """Compute (sum h)^2."""
    s = np.sum(h)
    return s * s


def _score(h):
    """Compute the objective: max|autoconv| / (sum h)^2. Lower is better."""
    si = _squared_integral(h)
    if si == 0:
        return np.inf
    return _autoconv_max_abs(h) / si


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (0/1 values)."""
    seq = np.array([0])
    while len(seq) < n:
        seq = np.concatenate([seq, 1 - seq])
    return seq[:n]


def _local_sign_flip_optimize(h, max_rounds=50, time_budget=None):
    """Iteratively flip signs of individual elements to reduce the objective."""
    best_h = h.copy().astype(float)
    best_score = _score(best_h)
    
    for _ in range(max_rounds):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        improved = False
        for i in range(len(best_h)):
            if time_budget is not None:
                if time.time() - _START_TIME > time_budget:
                    return best_h, best_score
            
            best_h[i] *= -1
            new_score = _score(best_h)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_h[i] *= -1
        
        if not improved:
            break
    
    return best_h, best_score


def _pair_flip_optimize(h, top_k=20, max_rounds=30, time_budget=None):
    """Try pair flips among top contributors to reduce the objective."""
    best_h = h.copy().astype(float)
    best_score = _score(best_h)
    
    for _ in range(max_rounds):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        improved = False
        # Try flipping pairs of elements
        idx = np.argsort(-np.abs(best_h))[:top_k]
        
        for a in range(len(idx)):
            if time_budget is not None:
                if time.time() - _START_TIME > time_budget:
                    return best_h, best_score
            
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                best_h[i] *= -1
                best_h[j] *= -1
                new_score = _score(best_h)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_h[i] *= -1
                    best_h[j] *= -1
        
        if not improved:
            break
    
    return best_h, best_score


def construct():
    """Construct a signed step function minimizing max|autoconv|/(sum h)^2."""
    np.random.seed(42)
    
    best_h = None
    best_score = np.inf
    
    # Generate Thue-Morse sequence and convert to ±1
    tm = _thue_morse(N)
    h_tm = 1.0 - 2.0 * tm
    
    # Try multiple Thue-Morse variants (shifts, sign flips)
    candidates = []
    
    # Original Thue-Morse
    candidates.append(h_tm.copy())
    
    # Negated Thue-Morse
    candidates.append(-h_tm.copy())
    
    # Shifted Thue-Morse (circular shift)
    for shift in [1, 2, 3]:
        candidates.append(np.roll(h_tm, shift).copy())
    
    # Also try a few structured patterns for comparison
    # Alternating
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    candidates.append(h_alt.copy())
    
    # Block pattern +1+1-1-1
    h_block = np.tile(np.array([1, 1, -1, -1]), N // 4 + 1)[:N]
    candidates.append(h_block.copy())
    
    # Random sparse 50%
    mask = np.random.rand(N) < 0.5
    h_rand = np.zeros(N)
    h_rand[mask] = np.random.choice([-1, 1], size=np.sum(mask))
    candidates.append(h_rand.copy())
    
    # Random sparse 30%
    mask = np.random.rand(N) < 0.3
    h_rand2 = np.zeros(N)
    h_rand2[mask] = np.random.choice([-1, 1], size=np.sum(mask))
    candidates.append(h_rand2.copy())
    
    # Evaluate and optimize each candidate
    for trial, h in enumerate(candidates):
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
        
        # Quick score before optimization
        sc = _score(h)
        
        # Local sign flip optimization
        h_opt, sc_opt = _local_sign_flip_optimize(h, max_rounds=30, time_budget=_TIME_LIMIT - 5)
        
        # Pair flip optimization
        h_opt, sc_opt = _pair_flip_optimize(h_opt, top_k=15, max_rounds=20, time_budget=_TIME_LIMIT - 5)
        
        if sc_opt < best_score:
            best_score = sc_opt
            best_h = h_opt.copy()
    
    # Final local polish
    if best_h is not None and time.time() - _START_TIME < _TIME_LIMIT - 5:
        best_h, best_score = _local_sign_flip_optimize(
            best_h, max_rounds=50, time_budget=_TIME_LIMIT - 2
        )
        best_h, best_score = _pair_flip_optimize(
            best_h, top_k=20, max_rounds=30, time_budget=_TIME_LIMIT - 2
        )
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
