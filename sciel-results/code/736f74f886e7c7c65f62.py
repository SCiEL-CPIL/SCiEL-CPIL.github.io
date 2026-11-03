import numpy as np
import time

N = 400

def _get_fft_size(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize

def _autoconv_fft(h):
    n = len(h)
    fsize = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:2*n-1]

def _score(h):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _generate_zadoff_chu(rng, N, root=1):
    n = np.arange(N)
    phase = -np.pi * root * n * (n + 1) / N
    h = np.sign(np.cos(phase))
    h[h == 0] = 1.0
    return h

def _generate_chirp(rng, N):
    n = np.arange(N)
    freq = 1.0 + rng.random() * 4.0
    phase = 2 * np.pi * freq * n**2 / N
    h = np.sign(np.cos(phase))
    h[h == 0] = 1.0
    return h

def _generate_random_binary(rng, N):
    return rng.choice([1.0, -1.0], N)

def _refine_binary(h, deadline, max_no_improve=20):
    """
    Local search on binary sequence h.
    Tries to reduce the score by flipping bits.
    """
    h = h.copy()
    current_score = _score(h)
    no_improve_count = 0
    
    # Pre-compute FFT size
    fsize = _get_fft_size(N)
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
            
        ac = _autoconv_fft(h)
        s = np.sum(h)
        
        # Find the lag with maximum absolute autocorrelation
        k_max = np.argmax(np.abs(ac))
        max_val = np.abs(ac[k_max])
        
        # If the peak is at lag 0, it's the energy, which is N for binary.
        # We care about sidelobes (k > 0).
        # But the objective is max over ALL lags.
        # For binary, AC[0] = N.
        # So max_val is at least N.
        # The score is N / s^2.
        # To minimize score, we want s^2 to be large (close to N^2) and sidelobes small.
        
        # Strategy:
        # 1. If k_max == 0, we are dominated by the main lobe.
        #    We cannot reduce AC[0] (it's fixed at N for binary).
        #    We can only increase s^2.
        #    So we should flip bits to make s closer to N or -N.
        # 2. If k_max > 0, we are dominated by a sidelobe.
        #    We should flip bits to reduce this sidelobe.
        
        if k_max == 0:
            # Try to increase |sum(h)|
            # Flip bit j: s_new = s - 2*h[j]
            # We want |s_new| > |s|
            # If s > 0, we want h[j] = -1 to flip to 1, increasing s.
            # If s < 0, we want h[j] = 1 to flip to -1, decreasing s (making more negative).
            
            target_sign = np.sign(s)
            if target_sign == 0:
                target_sign = 1.0
                
            # Find indices where h[j] is opposite to target_sign
            # Flipping these will increase |s|
            candidates = np.where(h == -target_sign)[0]
            
            if len(candidates) == 0:
                # All bits are already aligned with target sign.
                # We can't increase s further.
                # We might have to accept this or try to reduce sidelobes even if it reduces s slightly?
                # No, if s is maxed out, we are stuck with score = N / s^2.
                # We should break or try random perturbation.
                no_improve_count += 1
                # Try a random block flip to escape
                if np.random.rand() < 0.1:
                    start = np.random.randint(0, N - 10)
                    h[start:start+10] = -h[start:start+10]
                    new_score = _score(h)
                    if new_score < current_score:
                        current_score = new_score
                        no_improve_count = 0
                    else:
                        h[start:start+10] = -h[start:start+10]
                        no_improve_count += 1
                else:
                    break
            else:
                # Pick a random candidate to flip
                j = candidates[np.random.randint(len(candidates))]
                h[j] = -h[j]
                new_score = _score(h)
                if new_score < current_score:
                    current_score = new_score
                    no_improve_count = 0
                else:
                    h[j] = -h[j]
                    no_improve_count += 1
        else:
            # k_max > 0: Sidelobe is dominant.
            # We want to reduce |ac[k_max]|.
            # ac[k_max] = sum_{i} h[i] h[i+k_max]
            # Flipping h[j] changes ac[k_max] by -2 * h[j] * (h[j+k_max] + h[j-k_max])
            
            # Calculate the change for all j
            # We need to handle boundaries for j+k_max and j-k_max
            # h is length N.
            # ac[k] is defined for k in [0, N-1].
            # ac[k] = sum_{i=0}^{N-1-k} h[i] h[i+k]
            
            # Let's compute the "influence" of each h[j] on ac[k_max]
            # h[j] appears in ac[k_max] if:
            # 1. j is the first index: i=j, so j+k_max < N -> j < N-k_max
            # 2. j is the second index: i+k_max=j, so i=j-k_max >= 0 -> j >= k_max
            
            # Term 1: h[j] * h[j+k_max] for j in [0, N-1-k_max]
            # Term 2: h[j-k_max] * h[j] for j in [k_max, N-1]
            
            # Change in ac[k_max] if we flip h[j]:
            # delta = -2 * h[j] * (h[j+k_max] (if valid) + h[j-k_max] (if valid))
            
            delta = np.zeros(N)
            
            # Term 1
            valid_1 = np.arange(N - k_max)
            delta[valid_1] += -2 * h[valid_1] * h[valid_1 + k_max]
            
            # Term 2
            valid_2 = np.arange(k_max, N)
            delta[valid_2] += -2 * h[valid_2] * h[valid_2 - k_max]
            
            # New value at k_max: ac[k_max] + delta[j]
            # We want to minimize |ac[k_max] + delta[j]|
            # But we also need to check if this creates a new peak elsewhere.
            # Heuristic: Pick the j that minimizes |ac[k_max] + delta[j]|
            # Among those, pick the one that is most "central" or random?
            
            new_vals = np.abs(ac[k_max] + delta)
            
            # Find the best flip
            # We might want to avoid flips that don't change the score much
            # Let's pick the top 5 candidates
            # Sort indices by new_vals
            top_indices = np.argsort(new_vals)[:10]
            
            improved = False
            for j in top_indices:
                # Try flipping j
                h[j] = -h[j]
                new_score = _score(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    h[j] = -h[j]
            
            if not improved:
                no_improve_count += 1
                # If stuck, try a random perturbation
                if np.random.rand() < 0.2:
                    block_size = np.random.randint(2, 10)
                    start = np.random.randint(0, N - block_size)
                    h[start:start+block_size] = -h[start:start+block_size]
                    new_score = _score(h)
                    if new_score < current_score:
                        current_score = new_score
                        no_improve_count = 0
                    else:
                        h[start:start+block_size] = -h[start:start+block_size]
                        # Keep no_improve_count high to eventually break
                else:
                    # Try a single random flip
                    j = np.random.randint(N)
                    h[j] = -h[j]
                    new_score = _score(h)
                    if new_score < current_score:
                        current_score = new_score
                        no_improve_count = 0
                    else:
                        h[j] = -h[j]
                        
    return h, current_score

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin
    
    best_score = np.inf
    best_h = np.ones(N)
    
    rng = np.random.default_rng(42)
    
    # Generate a pool of initial candidates
    candidates = []
    
    # 1. Zadoff-Chu variants
    for root in [1, 2, 3, 5, 7, 11, 13]:
        h = _generate_zadoff_chu(rng, N, root)
        candidates.append(h)
        
    # 2. Chirp variants
    for _ in range(10):
        h = _generate_chirp(rng, N)
        candidates.append(h)
        
    # 3. Random binary
    for _ in range(10):
        h = _generate_random_binary(rng, N)
        candidates.append(h)
        
    # 4. Sorted/Block random
    for _ in range(5):
        h = np.zeros(N)
        # Random blocks
        pos = 0
        while pos < N:
            block_len = np.random.randint(1, 10)
            sign = np.random.choice([1.0, -1.0])
            end = min(pos + block_len, N)
            h[pos:end] = sign
            pos = end
        candidates.append(h)

    # Refine each candidate
    for i, h_init in enumerate(candidates):
        if time.time() > deadline:
            break
            
        # Allocate time for refinement
        # Earlier candidates get more time? Or equal?
        # Let's give each candidate a time slice, but break if total time is exceeded.
        # Simple: run until deadline - 5s for the last one, or just run all with a shared deadline.
        # To ensure we don't timeout, we check time inside the loop.
        
        h_opt, score_opt = _refine_binary(h_init, deadline)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
    return best_h

# EVOLVE-BLOCK-END
