import numpy as np
import time

N = 400

def _compute_psl(h):
    """
    Computes the Peak Sidelobe Level (PSL) of the aperiodic autocorrelation of h.
    PSL = max_{tau != 0} |R(tau)|
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.fft(h, n=n_fft)
    R = np.real(np.fft.ifft(H * np.conj(H)))
    # Extract lags 1 to N-1
    sidelobes = R[1:N]
    return np.max(np.abs(sidelobes))

def _compute_autocorr(h):
    """Compute full aperiodic autocorrelation R[0:N] where R[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k]"""
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.fft(h, n=n_fft)
    R = np.real(np.fft.ifft(H * np.conj(H)))
    return R[:N]

def _update_psl_after_flip(h, R, idx):
    """
    After flipping h[idx], update R in-place and return new PSL.
    The autocorrelation R[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k]
    When h[idx] flips, R[k] changes by 2*h[idx]*(h[idx+k] - h[idx-k]) for valid indices.
    """
    old_val = h[idx]
    h[idx] = -old_val
    
    # Update R[k] for all lags k
    # For lag k: R[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k]
    # Terms involving h[idx]:
    # - If idx < N-k: h[idx]*h[idx+k] is in the sum -> change by -2*old_val*h[idx+k]
    # - If idx >= k: h[idx-k]*h[idx] is in the sum -> change by -2*old_val*h[idx-k]
    # Note: if idx+k < N and idx-k >= 0, both terms exist and change by -2*old_val*(h[idx+k]+h[idx-k])
    # But wait, we flipped h[idx], so new h[idx] = -old_val
    # Old contribution: old_val*h[idx+k] and/or old_val*h[idx-k]
    # New contribution: (-old_val)*h[idx+k] and/or (-old_val)*h[idx-k]
    # Change: -2*old_val*h[idx+k] and/or -2*old_val*h[idx-k]
    
    for k in range(1, N):
        delta = 0.0
        # h[idx] appears as first element if idx <= N-1-k, i.e., idx+k < N
        if idx + k < N:
            delta += -2.0 * old_val * h[idx + k]
        # h[idx] appears as second element if idx >= k, i.e., idx-k >= 0
        if idx - k >= 0:
            delta += -2.0 * old_val * h[idx - k]
        if delta != 0.0:
            R[k] += delta
    
    return np.max(np.abs(R[1:N]))

def construct():
    """
    Constructs a signed step function (sequence of +/- 1) of length N
    that minimizes the peak sidelobe level of its autocorrelation.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    np.random.seed(42)
    
    best_h = None
    best_psl = np.inf
    
    # Generate diverse initial seeds
    seeds = []
    
    # Random seeds
    for _ in range(20):
        h_rand = np.random.choice([-1, 1], size=N)
        seeds.append(h_rand)
        
    # Thue-Morse
    tm = np.zeros(N, dtype=np.int8)
    for i in range(1, N):
        tm[i] = tm[i >> 1] ^ (i & 1)
    seeds.append(1 - 2 * tm)
    
    # Quadratic residue sequence
    qr = np.ones(N, dtype=np.int8)
    for i in range(1, N):
        # Check if i is a quadratic residue mod some prime
        # Simple version: use Legendre symbol mod 397 (largest prime < 400)
        p = 397
        if i % p != 0:
            if pow(i, (p-1)//2, p) == 1:
                qr[i] = 1
            else:
                qr[i] = -1
    seeds.append(qr)
    
    # Evaluate seeds
    for h in seeds:
        psl = _compute_psl(h)
        if psl < best_psl:
            best_psl = psl
            best_h = h.copy()
    
    # Greedy local search with restarts
    h = best_h.copy()
    R = _compute_autocorr(h)
    current_psl = best_psl
    
    # Multiple rounds of greedy local search
    max_rounds = 50
    for round_num in range(max_rounds):
        if time.time() >= deadline:
            break
        
        improved = True
        while improved:
            if time.time() >= deadline:
                break
            
            improved = False
            # Try flipping each position and keep the best improvement
            best_flip_idx = -1
            best_flip_psl = current_psl
            
            # Sample a subset of positions to speed up (or try all if fast enough)
            # With O(N) update per flip, trying all N=400 positions is 400*400=160k ops per round
            # That's fast enough. Let's try all positions.
            for idx in range(N):
                # Compute PSL after flipping idx
                new_psl = _update_psl_after_flip(h, R, idx)
                
                if new_psl < best_flip_psl:
                    best_flip_psl = new_psl
                    best_flip_idx = idx
                else:
                    # Revert the flip
                    h[idx] *= -1
                    # Revert R
                    old_val = h[idx]  # This is the original value before flip
                    # Actually, we need to revert properly
                    # h[idx] was flipped, so current h[idx] = -original
                    # We need to restore R
                    # The change was -2*old_val*(...) where old_val was the value before flip
                    # After flip, h[idx] = -old_val
                    # To revert, we add back 2*old_val*(...)
                    # But we already changed h[idx] back, so let's just recompute
                    pass
                
                # Revert h[idx] for next trial (if it was flipped)
                h[idx] *= -1
                # Revert R: we need to undo the change
                # The change to R[k] was -2*orig_val*h[idx±k]
                # After reverting h[idx], we need to add back 2*orig_val*h[idx±k]
                # But h[idx] is now back to orig_val, and h[idx±k] haven't changed
                # So we add 2*orig_val*h[idx±k] to R[k]
                orig_val = h[idx]  # This is the original value
                for k in range(1, N):
                    delta = 0.0
                    if idx + k < N:
                        delta += 2.0 * orig_val * h[idx + k]
                    if idx - k >= 0:
                        delta += 2.0 * orig_val * h[idx - k]
                    if delta != 0.0:
                        R[k] += delta
            
            if best_flip_idx >= 0 and best_flip_psl < current_psl:
                # Apply the best flip
                h[best_flip_idx] *= -1
                # Update R
                old_val = h[best_flip_idx]  # This is now -original
                # We need to update R for the flip
                # Change: -2*orig_val*h[idx±k] where orig_val is the value BEFORE flip
                # orig_val = -h[best_flip_idx] (since we just flipped)
                orig_val = -h[best_flip_idx]
                for k in range(1, N):
                    delta = 0.0
                    if best_flip_idx + k < N:
                        delta += -2.0 * orig_val * h[best_flip_idx + k]
                    if best_flip_idx - k >= 0:
                        delta += -2.0 * orig_val * h[best_flip_idx - k]
                    if delta != 0.0:
                        R[k] += delta
                
                current_psl = best_flip_psl
                improved = True
                
                if current_psl < best_psl:
                    best_psl = current_psl
                    best_h = h.copy()
        
        # If no improvement in this round, do a random restart
        if not improved:
            # Perturb current best or random restart
            if np.random.rand() < 0.3:
                h = np.random.choice([-1, 1], size=N)
            else:
                h = best_h.copy()
                num_flips = max(1, N // 10)
                flip_indices = np.random.choice(N, num_flips, replace=False)
                h[flip_indices] *= -1
            
            R = _compute_autocorr(h)
            current_psl = _compute_psl(h)
    
    return best_h
