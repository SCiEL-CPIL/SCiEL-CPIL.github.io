"""Optimized program: sparse signed sequence with local sign-flip optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s


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


def _local_sign_flip_optimize(h, max_rounds=50, time_budget=None):
    """Iteratively flip signs of individual elements to reduce the objective."""
    best_h = h.copy()
    best_score = _score(best_h)
    
    for _ in range(max_rounds):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        improved = False
        # Try flipping each position
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
                best_h[i] *= -1  # revert
        
        if not improved:
            break
    
    return best_h, best_score


def _pair_flip_optimize(h, max_rounds=20, time_budget=None):
    """Try flipping pairs of signs to escape local minima."""
    best_h = h.copy()
    best_score = _score(best_h)
    
    for _ in range(max_rounds):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        improved = False
        # Try flipping pairs (i, j) where i < j
        n = len(best_h)
        for i in range(n):
            if time_budget is not None:
                if time.time() - _START_TIME > time_budget:
                    return best_h, best_score
            for j in range(i+1, n):
                if time_budget is not None:
                    if time.time() - _START_TIME > time_budget:
                        return best_h, best_score
                
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
    
    # Generate diverse initial patterns
    patterns = []
    
    # Pattern 1: Dense alternating
    h1 = np.ones(N)
    h1[1::2] = -1
    patterns.append(h1)
    
    # Pattern 2: Sparse even positions with alternating signs
    h2 = np.zeros(N)
    h2[0::2] = 1
    h2[2::4] = -1
    patterns.append(h2)
    
    # Pattern 3: Random sparse 50%
    mask3 = np.random.rand(N) < 0.5
    h3 = np.zeros(N)
    h3[mask3] = np.random.choice([-1, 1], size=np.sum(mask3))
    patterns.append(h3)
    
    # Pattern 4: Sparse 30% with random signs
    mask4 = np.random.rand(N) < 0.3
    h4 = np.zeros(N)
    h4[mask4] = np.random.choice([-1, 1], size=np.sum(mask4))
    patterns.append(h4)
    
    # Pattern 5: Block pattern +1+1-1-1
    h5 = np.tile(np.array([1, 1, -1, -1]), N // 4 + 1)[:N]
    patterns.append(h5)
    
    # Pattern 6: Random sparse 40%
    mask6 = np.random.rand(N) < 0.4
    h6 = np.zeros(N)
    h6[mask6] = np.random.choice([-1, 1], size=np.sum(mask6))
    patterns.append(h6)
    
    # Pattern 7: Sparse 25%
    mask7 = np.random.rand(N) < 0.25
    h7 = np.zeros(N)
    h7[mask7] = np.random.choice([-1, 1], size=np.sum(mask7))
    patterns.append(h7)
    
    # Pattern 8: Alternating blocks of length 3
    h8 = np.tile(np.array([1, 1, -1]), N // 3 + 1)[:N]
    patterns.append(h8)
    
    # Pattern 9: Alternating blocks of length 5
    h9 = np.tile(np.array([1, 1, 1, -1, -1]), N // 5 + 1)[:N]
    patterns.append(h9)
    
    # Pattern 10: Random with bias toward zero (sparse)
    mask10 = np.random.rand(N) < 0.35
    h10 = np.zeros(N)
    h10[mask10] = np.random.choice([-1, 1], size=np.sum(mask10))
    patterns.append(h10)
    
    # Optimize each pattern
    for idx, h in enumerate(patterns):
        if time.time() - _START_TIME > _TIME_LIMIT - 20:
            break
        
        # Single sign flip optimization
        h_opt, sc = _local_sign_flip_optimize(h, max_rounds=30, time_budget=_TIME_LIMIT - 15)
        
        if sc < best_score:
            best_score = sc
            best_h = h_opt.copy()
        
        # If time allows, try pair flips on the best so far
        if time.time() - _START_TIME < _TIME_LIMIT - 30 and idx == 0:
            h_pair, sc_pair = _pair_flip_optimize(best_h, max_rounds=5, time_budget=_TIME_LIMIT - 25)
            if sc_pair < best_score:
                best_score = sc_pair
                best_h = h_pair.copy()
    
    # Final polish with more rounds
    if best_h is not None and time.time() - _START_TIME < _TIME_LIMIT - 10:
        best_h, best_score = _local_sign_flip_optimize(
            best_h, max_rounds=50, time_budget=_TIME_LIMIT - 5
        )
    
    # One more pair flip pass if time allows
    if best_h is not None and time.time() - _START_TIME < _TIME_LIMIT - 5:
        best_h, best_score = _pair_flip_optimize(
            best_h, max_rounds=3, time_budget=_TIME_LIMIT - 2
        )
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
