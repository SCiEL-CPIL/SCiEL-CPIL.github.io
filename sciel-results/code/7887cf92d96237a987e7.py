import numpy as np
import time

N = 400

def _autocorr(h):
    """
    Compute linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k] for k=0..N-1.
    Using FFT for efficiency.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    return ac

def _score(h):
    """
    Compute the metric: peak autoconvolution concentration.
    Lower is better.
    """
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +1/-1."""
    t = [1]
    for _ in range(int(np.log2(n)) + 1):
        t = t + [-x for x in t]
    return np.array(t[:n])

def _local_search(h, time_limit, rng):
    """
    Deterministic hill climbing with single-bit flips and small block flips.
    Returns the best h found.
    """
    best_h = h.copy()
    best_score = _score(best_h)
    t_start = time.time()
    
    # Precompute indices for fast access
    N = len(h)
    
    # Single-bit flip optimization
    # Flipping h[i] changes ac[k] by -2*h[i]*(h[i-k] + h[i+k])
    # We can update ac incrementally.
    
    h_curr = best_h.copy()
    ac_curr = _autocorr(h_curr)
    dc = ac_curr[0]
    
    # To speed up the check, we can maintain the current peak value and its index,
    # but since N is small, recomputing max is O(N) which is fine.
    
    # We will try flipping each bit. If it improves, we keep it.
    # We do multiple passes until no improvement or time limit.
    
    pass_count = 0
    while time.time() - t_start < time_limit:
        pass_count += 1
        improved = False
        
        # Randomize the order of checks to avoid bias
        order = rng.permutation(N)
        
        for i in order:
            if time.time() - t_start > time_limit:
                break
            
            # Flip h[i]
            old_val = h_curr[i]
            h_curr[i] = -old_val
            
            # Update autocorrelation
            # ac_new[k] = ac_old[k] - 2 * old_val * (h_curr[i-k] + h_curr[i+k])
            # Note: h_curr[i] is now -old_val, so the term is -2 * old_val * (new_val_at_i-k + new_val_at_i+k)
            # Actually, the change is:
            # ac[k] = sum h[j] h[j+k]
            # Terms involving i:
            # j=i: h[i] h[i+k] -> -h[i] h[i+k] (change: -2 h[i] h[i+k])
            # j+k=i: h[i-k] h[i] -> h[i-k] (-h[i]) (change: -2 h[i] h[i-k])
            # So ac_new[k] = ac_old[k] - 2 * old_val * (h_curr[i-k] + h_curr[i+k])
            # Wait, h_curr[i-k] and h_curr[i+k] are the CURRENT values (after flip? No, only i flipped).
            # h_curr[i-k] is unchanged. h_curr[i+k] is unchanged.
            # So yes, the formula holds.
            
            # Vectorized update
            # Indices: i-k and i+k
            # k ranges from 0 to N-1
            # i-k is valid if 0 <= i-k < N
            # i+k is valid if 0 <= i+k < N
            
            # Create masks
            k = np.arange(N)
            
            # Term 1: h[i-k]
            idx1 = i - k
            mask1 = (idx1 >= 0) & (idx1 < N)
            
            # Term 2: h[i+k]
            idx2 = i + k
            mask2 = (idx2 >= 0) & (idx2 < N)
            
            delta = np.zeros(N)
            if np.any(mask1):
                delta[mask1] += h_curr[idx1[mask1]]
            if np.any(mask2):
                delta[mask2] += h_curr[idx2[mask2]]
            
            ac_curr = ac_curr - 2.0 * old_val * delta
            
            # Check score
            # dc = ac_curr[0] (should be same as before, sum h[i]^2 = N)
            # But due to floating point, might drift slightly.
            # For now, use ac_curr[0].
            new_dc = ac_curr[0]
            if new_dc <= 1e-30:
                h_curr[i] = old_val
                ac_curr = ac_curr + 2.0 * old_val * delta
                continue
            
            new_score = np.max(np.abs(ac_curr)) / new_dc
            
            if new_score < best_score - 1e-12:
                best_score = new_score
                best_h = h_curr.copy()
                improved = True
            else:
                # Revert
                h_curr[i] = old_val
                ac_curr = ac_curr + 2.0 * old_val * delta
        
        if not improved:
            break

    return best_h

def construct():
    try:
        t0 = time.time()
        deadline = t0 + 115.0
        rng = np.random.default_rng(20240517)
        
        # 1. Generate Initial Candidates
        # Thue-Morse and shifts
        tm = _thue_morse(N)
        candidates = [tm]
        for shift in range(1, 10):
            candidates.append(np.roll(tm, shift))
        
        # Random
        for _ in range(10):
            candidates.append(rng.choice(np.array([-1.0, 1.0]), size=N))
            
        # Cosine based
        idx = np.arange(N)
        for freq in [1, 2, 3, 5, 10]:
            candidates.append(np.sign(np.cos(2 * np.pi * freq * idx / N)))
            candidates.append(np.sign(np.sin(2 * np.pi * freq * idx / N)))
            
        # 2. Multi-Resolution Coarse-to-Fine
        # Resolutions: 50, 100, 200, 400
        # We will run local search on each resolution level.
        
        best_overall = None
        best_overall_score = 1e9
        
        # Evaluate all initial candidates at full resolution to get a baseline
        # But we will use the multi-res process starting from the best initial.
        
        # First, find the best initial candidate at full resolution
        for h in candidates:
            if time.time() > deadline - 5.0:
                break
            s = _score(h)
            if s < best_overall_score:
                best_overall_score = s
                best_overall = h.copy()
                
        # Now, apply multi-resolution refinement to the best candidate
        # Start with the best candidate, but we can also try refining other good candidates.
        # Let's pick the top 3 initial candidates and refine them.
        
        scored_candidates = []
        for h in candidates:
            s = _score(h)
            scored_candidates.append((s, h))
        
        scored_candidates.sort(key=lambda x: x[0])
        top_k = 5
        top_candidates = [h for _, h in scored_candidates[:top_k]]
        
        for h_init in top_candidates:
            if time.time() > deadline - 10.0:
                break
            
            h_curr = h_init.copy()
            
            # Refinement levels: 100, 200, 400
            # At each level, we fix the lower bits and optimize the higher bits?
            # No, standard coarse-to-fine:
            # Level 1: 100 bins. Each bin is the average of 4 original bins?
            # Or simply, we take every 4th bin?
            # Let's use a decimation approach.
            
            # Actually, a simpler approach for "coarse-to-fine" in 1D is:
            # 1. Optimize on a subsampled version (e.g., every 4th point).
            # 2. Upsample by repeating each point 4 times.
            # 3. Optimize on the next level (every 2nd point).
            # 4. Upsample.
            # 5. Optimize on full resolution.
            
            h_sub = h_curr[::4].copy()
            # Local search on 100 points
            h_sub = _local_search(h_sub, time_limit=min(10.0, deadline - time.time() - 20.0), rng=rng)
            
            # Upsample to 400
            h_up = np.repeat(h_sub, 4)
            
            # Local search on 400 points (full resolution)
            # But we can also do a 200-point intermediate step?
            # Let's stick to 100 -> 400 for simplicity, or 100 -> 200 -> 400.
            
            # Let's do 100 -> 200 -> 400
            h_sub2 = h_up[::2].copy() # 200 points
            h_sub2 = _local_search(h_sub2, time_limit=min(10.0, deadline - time.time() - 10.0), rng=rng)
            
            h_up2 = np.repeat(h_sub2, 2) # 400 points
            
            # Final polish at 400
            h_final = _local_search(h_up2, time_limit=min(20.0, deadline - time.time() - 5.0), rng=rng)
            
            s_final = _score(h_final)
            if s_final < best_overall_score:
                best_overall_score = s_final
                best_overall = h_final.copy()
                
        # Final global polish on the best result
        if best_overall is not None and time.time() < deadline - 5.0:
            best_overall = _local_search(best_overall, time_limit=min(30.0, deadline - time.time() - 2.0), rng=rng)
            
        return best_overall

    except Exception:
        return np.ones(N)
