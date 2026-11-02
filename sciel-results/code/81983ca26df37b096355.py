"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_full(f):
    """Full autoconvolution (length 2N-1) via FFT."""
    n = len(f)
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    F = np.fft.rfft(f, fft_size)
    return np.fft.irfft(F * np.conj(F), fft_size)[:2*n-1]

def _score_from_autoconv(ac, S):
    """Compute max|autoconv| / S^2 given precomputed autoconv and sum."""
    if abs(S) < 1e-12:
        return float('inf')
    return np.max(np.abs(ac)) / (S * S)

def _build_state(f):
    """Build (autoconv, S, S2, f) for a binary ±1 vector."""
    ac = _autoconv_full(f)
    S = float(np.sum(f))
    return ac, S, S*S, f.copy()

def _flip_delta(i, f):
    """
    Return delta array for autoconv when flipping position i in a ±1 vector.
    delta[k] = ac_new[k] - ac_old[k] for k in [0, 2N-2].
    When f[i] changes by d = -2*f[i], the autoconv ac[k] = sum_j f[j]*f[j+k]
    changes by d*f[i+k] (if valid) + d*f[i-k] (if valid), except at k=0 where
    it's just d*f[i].
    """
    n = N
    delta = np.zeros(2*n-1)
    fi = f[i]
    d = -2.0 * fi
    # Term 1: j=i, so k such that i+k in [0,n-1], i.e., k in [max(0,-i), min(2n-2, n-1-i)]
    k_min1 = max(0, -i)
    k_max1 = min(2*n-2, n-1-i)
    if k_min1 <= k_max1:
        delta[k_min1:k_max1+1] += d * f[i+k_min1:i+k_max1+1]
    # Term 2: j+k=i, so j=i-k in [0,n-1], i.e., k in [max(0, i-n+1), min(2n-2, i)]
    k_min2 = max(0, i-n+1)
    k_max2 = min(2*n-2, i)
    if k_min2 <= k_max2:
        # f[i-k] for k from k_min2 to k_max2
        # i-k goes from i-k_min2 down to i-k_max2
        delta[k_min2:k_max2+1] += d * f[i-k_max2:i-k_min2+1][::-1]
    # At k=0, both terms refer to f[i], so we double-counted. Fix:
    delta[0] = d * fi
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    initializations = []
    # All ones
    initializations.append(np.ones(N))
    # Random ±1
    for seed in range(5):
        r = np.random.RandomState(seed)
        initializations.append(r.choice([-1, 1], size=N))
    # Structured patterns
    for freq in [1, 2, 3, 5, 8, 13]:
        x = np.linspace(0, freq * np.pi, N)
        initializations.append(np.sign(np.sin(x)))
        initializations.append(np.sign(np.cos(x)))
    # Block patterns
    for block in [2, 4, 8, 16, 32]:
        pat = np.tile([1]*block + [-1]*block, N//(2*block) + 1)[:N]
        initializations.append(pat)
    
    best_f = None
    best_score = float('inf')
    best_ac = None
    best_S = None
    best_S2 = None
    
    for init in initializations:
        ac, S, S2, f = _build_state(init)
        s = _score_from_autoconv(ac, S)
        if s < best_score:
            best_score = s
            best_f = f
            best_ac = ac
            best_S = S
            best_S2 = S2
    
    # Local search with fast incremental updates
    f = best_f.copy()
    ac = best_ac.copy()
    S = best_S
    S2 = best_S2
    current_score = best_score
    
    # Precompute: for each possible flip, we need delta. But delta depends on f,
    # so we compute it on the fly.
    
    # Use a priority-based or random search with incremental updates
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Try a random flip
        i = rng.randint(N)
        delta = _flip_delta(i, f)
        
        ac_new = ac + delta
        S_new = S - 2.0 * f[i]
        S2_new = S_new * S_new
        
        if abs(S_new) < 1e-12:
            # Skip this flip
            continue
        
        score_new = np.max(np.abs(ac_new)) / S2_new
        
        if score_new < current_score:
            # Accept
            ac = ac_new
            S = S_new
            S2 = S2_new
            f[i] = -f[i]
            current_score = score_new
            no_improve_count = 0
            
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
                best_ac = ac.copy()
                best_S = S
                best_S2 = S2
        else:
            no_improve_count += 1
            # If stuck for too long, try a random restart from a new initialization
            if no_improve_count > max_no_improve:
                # Pick a random new initialization
                r = np.random.RandomState(rng.randint(100000))
                new_init = r.choice([-1, 1], size=N)
                ac_new, S_new, S2_new, f_new = _build_state(new_init)
                s_new = _score_from_autoconv(ac_new, S_new)
                
                # Also try some structured inits
                for freq in [rng.randint(1, 20)]:
                    x = np.linspace(0, freq * np.pi, N)
                    for fn in [np.sign(np.sin(x)), np.sign(np.cos(x))]:
                        ac_t, S_t, S2_t, f_t = _build_state(fn)
                        s_t = _score_from_autoconv(ac_t, S_t)
                        if s_t < s_new:
                            s_new = s_t
                            ac_new, S_new, S2_new, f_new = ac_t, S_t, S2_t, f_t
                
                if s_new < current_score:
                    f = f_new
                    ac = ac_new
                    S = S_new
                    S2 = S2_new
                    current_score = s_new
                    no_improve_count = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_f = f.copy()
                        best_ac = ac.copy()
                        best_S = S
                        best_S2 = S2
                else:
                    # Reset counter, keep searching
                    no_improve_count = 0
    
    return best_f
# EVOLVE-BLOCK-END
