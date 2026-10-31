"""Combined program: Multi-start with Thue-Morse and Sparse structures."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s limit

def _autoconv_full(h):
    """Compute full autoconvolution of h."""
    return fftconvolve(h, h, mode='full')

def _autoconv_max_abs(h):
    """Compute max |autoconvolution| of h."""
    ac = _autoconv_full(h)
    return np.max(np.abs(ac))

def _squared_integral(h):
    """Compute (sum h)^2."""
    s = np.sum(h)
    return s * s

def _score(h):
    """Compute the objective: max|autoconv| / (sum h)^2. Lower is better."""
    si = _squared_integral(h)
    if si < 1e-12:
        return np.inf
    return _autoconv_max_abs(h) / si

def _local_sign_flip_optimize_fast(h, max_rounds=100, time_budget=None):
    """Iteratively flip signs using incremental autoconvolution update."""
    best_h = h.copy().astype(float)
    n = len(best_h)
    
    # Precompute initial autoconvolution
    ac = _autoconv_full(best_h)
    max_abs = np.max(np.abs(ac))
    s = np.sum(best_h)
    best_score = max_abs / (s * s) if abs(s) > 1e-12 else np.inf
    
    for round_num in range(max_rounds):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        # Periodically recompute ac to avoid floating-point drift
        if round_num > 0 and round_num % 20 == 0:
            ac = _autoconv_full(best_h)
            max_abs = np.max(np.abs(ac))
        
        improved = False
        # Try flipping each position
        for i in range(n):
            if time_budget is not None:
                if time.time() - _START_TIME > time_budget:
                    return best_h, best_score
            
            # Compute delta for flipping h[i]
            # Current value: v = best_h[i]
            # After flip: -v
            # ac[m] change: for each m, the terms involving h[i] change
            # Term j=i: h[i]*h[i+m] -> (-v)*h[i+m], change = -2*v*h[i+m]
            # Term j=i-m (i.e., j+m=i): h[i-m]*h[i] -> h[i-m]*(-v), change = -2*h[i-m]*v
            # For m=0: both terms are the same (j=i), so change = -2*v*h[i] = -2*v^2 = -2
            #   BUT ac[0] = sum h[j]^2, which is invariant under sign flips. So m=0 is unchanged.
            
            v = best_h[i]
            # Compute new ac values for affected lags
            # We'll compute the delta for each lag m
            # Lags affected: m such that 0<=i+m<n (gives j=i term) or 0<=i-m<n (gives j=i-m term)
            # m ranges from -(n-1) to (n-1), m!=0
            
            # Apply flip temporarily
            best_h[i] = -v
            s_new = s - 2*v  # s changes by -2*v
            
            # Update ac for all affected lags
            # For efficiency, compute deltas and track new max
            # We need to update ac and find new max
            # Affected lags: m where i+m in [0,n) or i-m in [0,n), m != 0
            # m range: max(-(i), -(n-1)) to min(n-1-i, n-1), excluding 0
            m_min = max(-i, -(n-1))
            m_max = min(n-1-i, n-1)
            
            new_max_abs = max_abs  # will update
            # We need to recompute max over all lags, but most are unchanged
            # The unchanged lags have the same |ac| as before
            # Only affected lags may have changed
            # So new_max = max(old_max_excluding_affected, max_affected_new)
            # This is complex. For n=400, let's just recompute max over all.
            
            for m in range(m_min, m_max + 1):
                if m == 0:
                    continue
                idx = m + n - 1
                delta = 0.0
                if 0 <= i + m < n:
                    delta += -2.0 * v * best_h[i + m]
                if 0 <= i - m < n:
                    delta += -2.0 * best_h[i - m] * v
                if delta != 0.0:
                    ac[idx] += delta
            
            new_max_abs = np.max(np.abs(ac))
            new_score = new_max_abs / (s_new * s_new) if abs(s_new) > 1e-12 else np.inf
            
            if new_score < best_score:
                best_score = new_score
                best_h[i] = -v  # keep flip
                s = s_new
                improved = True
            else:
                # Revert: undo the ac changes
                best_h[i] = v
                for m in range(m_min, m_max + 1):
                    if m == 0:
                        continue
                    idx = m + n - 1
                    delta = 0.0
                    if 0 <= i + m < n:
                        delta += -2.0 * v * best_h[i + m]
                    if 0 <= i - m < n:
                        delta += -2.0 * best_h[i - m] * v
                    if delta != 0.0:
                        ac[idx] -= delta  # subtract to revert
    
        if not improved:
            break
    
    return best_h, best_score

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    tm = np.zeros(n, dtype=int)
    for i in range(n):
        tm[i] = bin(i).count('1') % 2
    return tm

def _quadratic_phase(n, a=1):
    """Quadratic phase: sign of cos(pi*a*i^2/n), with zeros mapped to 1."""
    i = np.arange(n)
    result = np.sign(np.cos(np.pi * a * i**2 / n))
    result = np.where(result == 0, 1.0, result)
    return result

def _m_sequence_like(n, p=257):
    """Generate a sequence based on a primitive polynomial over GF(2), mapped to signs."""
    # Simple LFSR-based sequence
    # Use a known primitive polynomial for good autocorrelation
    # x^8 + x^4 + x^3 + x^2 + 1 (degree 8, period 255)
    # We'll generate a long sequence and take n elements
    length_needed = n + 10
    poly = 0b100111001  # x^8 + x^4 + x^3 + x^2 + 1
    degree = 8
    state = 1  # non-zero initial state
    seq = np.zeros(length_needed, dtype=int)
    for i in range(length_needed):
        seq[i] = state & 1
        # Shift and feedback
        lsb = state & 1
        state >>= 1
        if lsb:
            state ^= poly
    # Map 0->1, 1->-1
    return (1 - 2 * seq[:n]).astype(float)

def construct():
    """Construct a signed step function minimizing max|autoconv|/(sum h)^2."""
    np.random.seed(42)
    candidates = []
    
    # 1. Constant vector (Baseline)
    candidates.append(np.ones(N))
    
    # 2. Thue-Morse based
    tm = _thue_morse(N)
    h_tm = 1 - 2 * tm  # 0->1, 1->-1
    candidates.append(h_tm.astype(float))
    
    # 3. Quadratic phase
    candidates.append(_quadratic_phase(N, a=1).astype(float))
    candidates.append(_quadratic_phase(N, a=2).astype(float))
    
    # 4. M-sequence like
    candidates.append(_m_sequence_like(N))
    
    # 5. Sparse Random
    mask = np.random.rand(N) < 0.5
    signs = np.random.choice([-1, 1], N)
    h_sparse = mask * signs
    candidates.append(h_sparse.astype(float))
    
    # 6. Sparse Thue-Morse
    mask2 = np.random.rand(N) < 0.5
    h_sparse_tm = mask2 * h_tm
    candidates.append(h_sparse_tm.astype(float))
    
    # 7. Random signs
    candidates.append(np.random.choice([-1.0, 1.0], N))
    
    best_h = None
    best_score = np.inf
    
    time_budget = _TIME_LIMIT - 15
    
    for h in candidates:
        if time.time() - _START_TIME > time_budget:
            break
        
        # Optimize the candidate
        h_opt, score = _local_sign_flip_optimize_fast(h, max_rounds=50, time_budget=time_budget)
        
        if score < best_score:
            best_score = score
            best_h = h_opt
    
    # If no candidate was processed (timeout), return constant
    if best_h is None:
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
