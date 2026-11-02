"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    Overlap at shift k is sum_i h[i] * (1 - h[(i+k) % N]).
    We want to minimize the max over k of this quantity.
    """
    one_minus_h = 1.0 - h
    # Circular correlation: for each shift k, compute dot(h, roll(one_minus_h, k))
    # Using FFT for efficiency
    N = len(h)
    H = np.fft.fft(h)
    G = np.fft.fft(one_minus_h)
    # Circular cross-correlation
    # corr[k] = sum_i h[i] * one_minus_h[(i+k) % N]
    # This is the circular cross-correlation of h and one_minus_h
    # Using FFT: corr = ifft(fft(h) * conj(fft(one_minus_h))) gives sum_i h[i+k] * one_minus_h[i]
    # We want sum_i h[i] * one_minus_h[i+k]
    # Let's compute directly for correctness
    max_ov = 0.0
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _max_overlap_fast(h):
    """
    Faster computation using FFT.
    Overlap at shift k: sum_i h[i] * (1-h)[(i+k) % N]
    = sum_i h[i] - sum_i h[i] * h[(i+k) % N]
    = sum(h) - circular_autocorr_h[k]
    
    where circular_autocorr_h[k] = sum_i h[i] * h[(i+k) % N]
    """
    N = len(h)
    H = np.fft.fft(h)
    # Circular autocorrelation
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]
    # Using FFT: autocorr = ifft(H * conj(H)) gives sum_i h[i] * h[(i-k) % N]
    # So autocorr_neg[k] = sum_i h[i] * h[(i-k) % N]
    # autocorr_pos[k] = sum_i h[i] * h[(i+k) % N] = autocorr_neg[N-k]
    power = H * np.conj(H)
    autocorr_neg = np.fft.ifft(power).real
    # autocorr_neg[k] = sum_i h[i] * h[(i-k) % N]
    # We want autocorr_pos[k] = sum_i h[i] * h[(i+k) % N]
    # autocorr_pos[k] = autocorr_neg[-k % N] = autocorr_neg[(N-k) % N]
    autocorr_pos = np.roll(autocorr_neg, -1)[1:]  # This might be off by one
    # Let's be more careful
    # autocorr_neg[k] = sum_i h[i] * h[(i-k) % N]
    # Let j = i-k, then i = j+k
    # autocorr_neg[k] = sum_j h[j+k] * h[j]
    # This is the cross-correlation with lag k
    # We want sum_i h[i] * h[(i+k) % N]
    # Let j = i+k, i = j-k
    # = sum_j h[j-k] * h[j]
    # This is autocorr_neg[-k % N]
    # autocorr_neg[m] = sum_j h[j+m] * h[j]
    # So sum_j h[j-k] * h[j] = autocorr_neg[-k % N]
    autocorr_pos = np.zeros(N)
    for k in range(N):
        autocorr_pos[k] = autocorr_neg[(-k) % N]
    
    sum_h = np.sum(h)
    overlaps = sum_h - autocorr_pos
    return np.max(overlaps)

def _score(h):
    """Score is the upper bound on C5, which is the max overlap."""
    return _max_overlap_fast(h)

def construct():
    """
    Construct step heights to minimize max overlap.
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Initial solution: uniform
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _score(h)
    
    # Gradient descent with projection
    # Objective: min max_k (sum(h) - autocorr_pos[k])
    # = min max_k (N/2 - autocorr_pos[k])  [since sum(h) = N/2 is fixed]
    # = N/2 - max_k autocorr_pos[k]
    # So minimizing max overlap is equivalent to maximizing min_k autocorr_pos[k]
    
    # Let's use a smoothed max approximation
    # f(h) = logsumexp of the overlaps, or just the max
    
    lr = 0.01
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Compute overlaps
        N = len(h)
        H = np.fft.fft(h)
        power = H * np.conj(H)
        autocorr_neg = np.fft.ifft(power).real
        autocorr_pos = np.zeros(N)
        for k in range(N):
            autocorr_pos[k] = autocorr_neg[(-k) % N]
        
        sum_h = np.sum(h)
        overlaps = sum_h - autocorr_pos
        max_ov = np.max(overlaps)
        
        # Gradient of max overlap w.r.t. h
        # The max is achieved at some k*, so gradient is gradient of overlap at k*
        # overlap[k] = sum_i h[i] * (1-h)[(i+k)%N]
        # d(overlap[k])/dh[j] = (1-h)[(j+k)%N] - h[j] * 1 (from the (1-h) term)
        # Wait, let's be more careful
        # overlap[k] = sum_i h[i] * (1 - h[(i+k)%N])
        # = sum_i h[i] - sum_i h[i] * h[(i+k)%N]
        # d/dh[j] of sum_i h[i] = 1
        # d/dh[j] of sum_i h[i] * h[(i+k)%N]:
        #   If i = j: h[(j+k)%N]
        #   If (i+k)%N = j, i.e., i = (j-k)%N: h[j]
        # So d/dh[j] of autocorr_pos[k] = h[(j+k)%N] + h[j] (if k != 0)
        # For k=0: d/dh[j] of sum_i h[i]^2 = 2*h[j]
        
        # Gradient of max overlap:
        # Find k* that achieves the max
        k_star = np.argmax(overlaps)
        
        grad = np.ones(N)  # from sum(h)
        if k_star == 0:
            grad -= 2 * h
        else:
            grad -= np.roll(h, -k_star)  # h[(j+k_star)%N]
            grad -= h  # h[j] from the other term
            # Wait, let me re-derive
            # autocorr_pos[k] = sum_i h[i] * h[(i+k)%N]
            # d/dh[j] = h[(j+k)%N] (from i=j term) + h[(j-k)%N] (from (i+k)%N=j term)
            # So d/dh[j] of autocorr_pos[k] = h[(j+k)%N] + h[(j-k)%N]
            grad -= np.roll(h, -k_star)  # h[(j+k_star)%N]
            grad -= np.roll(h, k_star)   # h[(j-k_star)%N]
        
        # Update
        h_new = h - lr * grad
        
        # Project to constraints: 0 <= h[i] <= 1, sum(h) = N/2
        # Simple projection: clip, then adjust sum
        h_new = np.clip(h_new, 0, 1)
        # Adjust to satisfy sum constraint
        diff = N / 2 - np.sum(h_new)
        if diff > 0:
            # Need to add diff, distribute to elements with room
            room = 1 - h_new
            total_room = np.sum(room)
            if total_room > 0:
                h_new += diff * room / total_room
        elif diff < 0:
            # Need to subtract -diff, distribute from elements above 0
            space = h_new
            total_space = np.sum(space)
            if total_space > 0:
                h_new += diff * space / total_space
        
        h_new = np.clip(h_new, 0, 1)
        
        # Check score
        new_score = _score(h_new)
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
        
        h = h_new
        
        # Adaptive learning rate
        if iteration % 100 == 0:
            lr = max(0.001, lr * 0.99)
    
    return best_h
# EVOLVE-BLOCK-END

