"""
Combined strategy:
1. Structural Initialization: Use 'Pulse Comb' representations (sparse sums of shifted pulses) 
   to generate diverse, structured initial candidates that naturally have lower autoconvolution peaks.
2. Refinement: Use a fast, O(N) update rule for single-flip local search to efficiently 
   minimize the exact objective (max|autoconv| / sum^2) in the discrete binary space.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

def _score_sequence(s):
    """True score: max|autoconv| / (sum s)^2."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return max_ac / sq_int

def _soft_max_ac(s, temperature=1.0, K=128):
    """Differentiable surrogate for max|autoconv| via top-K log-sum-exp."""
    ac = fftconvolve(s, s, mode='full')
    ac_sq = ac ** 2
    max_val = np.max(ac_sq)
    if max_val < 1e-10:
        return 0.0
    # Only the largest K values matter for the log-sum-exp
    kk = min(K, ac_sq.size)
    idx = np.argpartition(ac_sq, -kk)[-kk:]
    top = ac_sq[idx]
    lse = max_val + np.log(np.sum(np.exp(top - max_val)))
    return np.sqrt(lse)

def _soft_objective(x, temperature=1.0, K=128):
    """Soft objective: log-sum-exp surrogate for max|ac| divided by (sum s)^2."""
    s = np.tanh(x)
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return 1e10
    return _soft_max_ac(s, temperature, K) / sq_int

def _generate_pulse_comb_init(rng, n_pulses=20, pulse_width=3):
    """
    Generate an initial signal as a sum of shifted rectangular pulses.
    This structural constraint helps escape 'random noise' local minima.
    """
    h = np.zeros(N)
    # Choose random positions and amplitudes
    positions = rng.integers(0, N, n_pulses)
    amplitudes = rng.choice([-1.0, 1.0], n_pulses)
    for pos, amp in zip(positions, amplitudes):
        # Add a rectangular pulse of width pulse_width centered at pos
        start = pos - pulse_width // 2
        end = start + pulse_width
        # Wrap around or clip? Clip for simplicity in linear conv context
        s_idx = max(0, start)
        e_idx = min(N, end)
        if s_idx < e_idx:
            h[s_idx:e_idx] += amp
    # Normalize to avoid zero sum
    if abs(np.sum(h)) < 1e-6:
        h += 0.1 * rng.standard_normal(N)
    return h

def _fast_flip_score(cur, cur_ac, cur_sum, j):
    """
    Calculate the score if we flip index j.
    Uses the O(N) update rule for autoconvolution.
    Returns (new_score, new_ac, new_sum) if flip is valid, else (inf, None, None).
    """
    val = cur[j]
    if val == 0:
        return np.inf, None, None
    
    delta = -2.0 * val
    new_sum = cur_sum + delta
    nsq = new_sum ** 2
    if nsq < 1e-10:
        return np.inf, None, None
    
    # Update autoconvolution
    # A_new[k] = A[k] - 4 * val * h[k-j] + 4 * val^2 * delta_{k, 2j}
    # Valid range for h[k-j] is k-j in [0, N-1] => k in [j, j+N-1]
    # We need to handle boundaries carefully.
    
    # Create a copy of the autoconvolution to update
    # To save time, we don't copy the whole array if we can avoid it, 
    # but we need the max of the modified array.
    # The modification is sparse in the sense that it's a shifted copy of h.
    
    # Let's compute the new max directly without full copy if possible?
    # No, the shift changes which elements are affected.
    # We must update the array.
    
    # Optimization: We can pre-allocate new_ac
    new_ac = cur_ac.copy()
    
    # Range of k affected:
    # k-j must be in [0, N-1]
    # k_min = j
    # k_max = j + N - 1
    # Also the delta term at k=2j
    
    k_start = j
    k_end = j + N
    
    # Clip to valid autoconv indices [0, 2N-2]
    valid_start = max(0, k_start)
    valid_end = min(2*N - 1, k_end)
    
    if valid_start < valid_end:
        # Indices in h: k - j
        # k ranges from valid_start to valid_end
        # h indices range from valid_start - j to valid_end - j - 1
        h_start = valid_start - j
        h_end = valid_end - j
        
        new_ac[valid_start:valid_end] += -4.0 * val * cur[h_start:h_end]
        
        # Add the squared term if 2j is within bounds
        k_sq = 2 * j
        if 0 <= k_sq < 2*N - 1:
             # Note: if k_sq is in the range [valid_start, valid_end), 
             # we already added the linear term part.
             # The formula is A_new = A - 4v h + 4v^2 delta.
             # So we just add 4v^2 at k_sq.
             new_ac[k_sq] += 4.0 * val * val
             
    new_max = np.max(np.abs(new_ac))
    new_score = new_max / nsq
    return new_score, new_ac, new_sum

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin
    
    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline
    base_score = _score_sequence(best_h)
    
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    rng = np.random.default_rng(42)
    
    # Phase 1: Continuous Optimization with Diverse Inits
    # We use a mix of random and structured (pulse comb) inits.
    
    n_restarts = 15
    for it in range(n_restarts):
        if time.time() > deadline - 30:
            break
            
        # Choose init strategy
        if it % 3 == 0:
            # Random Gaussian
            x0 = rng.standard_normal(N) * 0.5
        elif it % 3 == 1:
            # Pulse Comb
            h_init = _generate_pulse_comb_init(rng, n_pulses=30, pulse_width=2)
            # Map to x space (arctanh) for tanh parameterization
            # Clip to avoid inf
            h_init = np.clip(h_init / (np.max(np.abs(h_init)) + 1e-9), -0.9, 0.9)
            x0 = np.arctanh(h_init)
        else:
            # Chirp / Structured
            t = np.arange(N)
            freq = rng.uniform(0.1, 1.0)
            h_init = np.cos(2 * np.pi * freq * t + rng.uniform(0, 2*np.pi))
            h_init = np.clip(h_init, -0.9, 0.9)
            x0 = np.arctanh(h_init)
            
        temp = 2.0 * (0.7 ** (it / 5.0))
        
        try:
            result = minimize(
                _soft_objective,
                x0,
                method='L-BFGS-B',
                args=(temp, 128),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            s_cont = np.tanh(result.x)
            
            # Evaluate continuous
            sc_c = _score_sequence(s_cont)
            if sc_c < best_score:
                best_score = sc_c
                best_h = s_cont.copy()
                
            # Binarize
            s_bin = np.sign(s_cont)
            s_bin[s_bin == 0] = 1.0
            
            sc_b = _score_sequence(s_bin)
            if sc_b < best_score:
                best_score = sc_b
                best_h = s_bin.copy()
                
        except Exception:
            continue

    # Phase 2: Discrete Local Search (Fast Flip)
    # Start from the best binary solution found so far
    if np.all(np.abs(best_h) > 0.5): # If it's binary
        cur = best_h.copy()
    else:
        cur = np.sign(best_h)
        cur[cur == 0] = 1.0
        
    cur_ac = fftconvolve(cur, cur, mode='full')
    cur_sum = float(np.sum(cur))
    cur_score = _score_sequence(cur)
    
    if cur_score < best_score:
        best_score = cur_score
        best_h = cur.copy()
    else:
        cur_score = best_score
        cur = best_h.copy()
        cur_ac = fftconvolve(cur, cur, mode='full')
        cur_sum = float(np.sum(cur))
        
    # Greedy Local Search
    max_passes = 500
    for _pass in range(max_passes):
        if time.time() > deadline - 20:
            break
            
        best_j = -1
        best_js = cur_score
        best_new_ac = None
        best_new_sum = cur_sum
        best_val = cur[best_j] if best_j != -1 else 0
        
        # Evaluate all single flips
        for j in range(N):
            if time.time() > deadline - 20:
                break
            ns, n_ac, n_sum = _fast_flip_score(cur, cur_ac, cur_sum, j)
            if ns < best_js:
                best_js = ns
                best_j = j
                best_new_ac = n_ac
                best_new_sum = n_sum
                best_val = cur[j]
                
        if best_j == -1 or best_js >= cur_score - 1e-12:
            break
            
        # Apply the best flip
        cur[best_j] = -best_val
        cur_ac = best_new_ac
        cur_sum = best_new_sum
        cur_score = best_js
        
        if cur_score < best_score:
            best_score = cur_score
            best_h = cur.copy()

    # Phase 3: Simulated Annealing
    # Use the fast flip evaluation to explore
    T_start = 0.05
    sa_steps = 2000
    for it in range(sa_steps):
        if time.time() > deadline - 5:
            break
            
        idx = int(rng.integers(N))
        ns, n_ac, n_sum = _fast_flip_score(cur, cur_ac, cur_sum, idx)
        
        if ns == np.inf:
            continue
            
        d = ns - cur_score
        T = max(T_start * (0.995 ** it), 1e-6)
        
        if d < 0 or rng.random() < np.exp(-d / T):
            val = cur[idx]
            cur[idx] = -val
            cur_ac = n_ac
            cur_sum = n_sum
            cur_score = ns
            
            if cur_score < best_score:
                best_score = cur_score
                best_h = cur.copy()

    # Final check
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
