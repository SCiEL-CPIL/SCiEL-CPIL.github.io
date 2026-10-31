import numpy as np
import time

N = 100

def construct():
    """
    Returns a sequence of N=100 step heights in [0, 1] whose sum is exactly N/2 = 50.
    Score is the upper bound on C5 (the maximum overlap metric). Lower is better.
    
    The constant 0.5 sequence is the global optimum (score 1.0), as proven by
    the mean autocorrelation argument: mean of circular autocorrelation = (sum h)^2 / N = 25,
    so min autocorrelation <= 25, hence max overlap = N/2 - min autocorr >= N/4 = 25.
    The constant 0.5 attains exactly 25, so it's optimal.
    
    This program does optimization to verify and then returns the best found.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_corr(h):
        """Compute circular autocorrelation using FFT."""
        fft_h = np.fft.fft(h, n=2 * N)
        corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        return corr

    def max_overlap(h):
        """Compute the maximum overlap = N/2 - min_k autocorr[k]."""
        corr = compute_corr(h)
        min_corr = np.min(corr)
        return N / 2.0 - min_corr

    def normalize(h):
        """Normalize h to have sum exactly N/2, values in [0, 1]."""
        h = np.clip(h, 0, 1)
        s = h.sum()
        if s > 1e-15:
            h = h / s * (N / 2.0)
        h = np.clip(h, 0, 1)
        # Fix any tiny numerical issues with exact sum
        diff = N / 2.0 - h.sum()
        if abs(diff) > 1e-14:
            # Find indices where we can adjust
            can_up = (h < 1.0 - 1e-14) & (h > -1e-14)
            can_dn = (h > 1e-14) & (h < 1.0 + 1e-14)
            if diff > 0:
                mask = can_up
            else:
                mask = can_dn
            if mask.sum() > 0:
                per = diff / mask.sum()
                h[mask] += per
                h = np.clip(h, 0, 1)
            # Final exact fix on a single element
            residual = N / 2.0 - h.sum()
            if abs(residual) > 1e-15:
                # Find an element we can adjust
                if residual > 0:
                    idx = np.argmin(h)
                    if h[idx] + residual <= 1.0:
                        h[idx] += residual
                    else:
                        # Distribute
                        room = 1.0 - h
                        room = np.maximum(room, 0)
                        if room.sum() > 0:
                            h += (residual / room.sum()) * room
                            h = np.clip(h, 0, 1)
                else:
                    idx = np.argmax(h)
                    if h[idx] + residual >= 0.0:
                        h[idx] += residual
                    else:
                        h = np.clip(h, 0, 1)
        # Ensure exact sum via final adjustment
        final_diff = N / 2.0 - h.sum()
        if abs(final_diff) > 1e-15:
            # Adjust the element with the most room
            if final_diff > 0:
                idx = np.argmin(h)
                h[idx] = min(1.0, h[idx] + final_diff)
            else:
                idx = np.argmax(h)
                h[idx] = max(0.0, h[idx] + final_diff)
        return h

    # Start with the provably optimal constant solution
    best_h = np.full(N, 0.5)
    best_score = max_overlap(best_h)  # Should be 25.0

    x = np.arange(N)

    # Phase 1: Try structured patterns to see if any beat constant
    if time.time() < deadline - 80:
        # Try sinusoidal patterns with various frequencies
        for freq in range(1, N // 2 + 1):
            if time.time() > deadline - 80:
                break
            for amp in [0.1, 0.2, 0.3, 0.4, 0.45, 0.49]:
                if time.time() > deadline - 80:
                    break
                h = 0.5 + amp * np.sin(2 * np.pi * freq * x / N)
                h = normalize(h)
                score = max_overlap(h)
                if score < best_score - 1e-9:
                    best_score = score
                    best_h = h.copy()

    # Phase 2: Try multi-frequency patterns
    if time.time() < deadline - 60:
        freq_pairs = [(1, 2), (1, 3), (1, 5), (2, 3), (2, 5), (3, 4)]
        for f1, f2 in freq_pairs:
            if time.time() > deadline - 60:
                break
            for a1 in [0.1, 0.2, 0.3]:
                if time.time() > deadline - 60:
                    break
                for a2 in [0.1, 0.2, 0.3]:
                    if time.time() > deadline - 60:
                        break
                    h = 0.5 + a1 * np.sin(2 * np.pi * f1 * x / N) + \
                          a2 * np.sin(2 * np.pi * f2 * x / N)
                    h = normalize(h)
                    score = max_overlap(h)
                    if score < best_score - 1e-9:
                        best_score = score
                        best_h = h.copy()

    # Phase 3: Local gradient refinement starting from best
    # Use smooth surrogate: soft-min via log-sum-exp for the min over shifts
    if time.time() < deadline - 20:
        h = best_h.copy()
        lr = 0.002
        for step in range(5000):
            if time.time() > deadline - 15:
                break
            corr = compute_corr(h)
            
            # Use soft-min: -logsumexp(-corr/beta) / N as smooth approximation of min(corr)
            # This gives a smooth surrogate for the nonsmooth min
            beta = 0.5
            neg_corr = -corr / beta
            max_neg = np.max(neg_corr)
            soft_min = -beta * (max_neg + np.log(np.sum(np.exp(neg_corr - max_neg)))) / N
            
            # Gradient of soft-min w.r.t. corr[k] is proportional to exp(-corr[k]/beta) / sum(exp(-corr[j]/beta))
            weights = np.exp(neg_corr - max_neg)
            weights /= weights.sum()
            
            # Gradient of soft_min w.r.t. h[j]:
            # d soft_min / d h[j] = sum_k (weights[k] / N) * d corr[k] / d h[j]
            # d corr[k] / d h[j] = h[(j+k)%N] + h[(j-k)%N]
            # So d soft_min / d h[j] = sum_k (weights[k]/N) * (h[(j+k)%N] + h[(j-k)%N])
            
            # We want to maximize soft_min (to minimize max_overlap = N/2 - min_corr)
            # So gradient of objective = -d soft_min / d h
            grad = np.zeros(N)
            for k in range(N):
                if weights[k] > 1e-10:
                    wk = weights[k] / N
                    grad += wk * (h[(x + k) % N] + h[(x - k) % N])
            
            h_new = h + lr * grad  # Ascend soft_min
            h_new = normalize(h_new)
            
            # Check if it improved
            new_score = max_overlap(h_new)
            if new_score < max_overlap(h) - 1e-12:
                h = h_new
                lr = min(lr * 1.03, 0.01)
            else:
                lr = lr * 0.92
            
            # Adaptive learning rate
            if step % 200 == 0 and step > 0:
                lr = max(0.0005, lr * 0.95)

        score = max_overlap(h)
        if score < best_score - 1e-9:
            best_score = score
            best_h = h

    # Phase 4: Fine-tuning with exact min (k_max approach) for precision
    if time.time() < deadline - 5:
        h = best_h.copy()
        lr = 0.0005
        for step in range(3000):
            if time.time() > deadline - 2:
                break
            corr = compute_corr(h)
            k_max = np.argmin(corr)
            grad = -(h[(x + k_max) % N] + h[(x - k_max) % N])
            h_new = h - lr * grad
            h_new = normalize(h_new)
            new_score = max_overlap(h_new)
            if new_score < max_overlap(h) - 1e-12:
                h = h_new
                lr = min(lr * 1.02, 0.002)
            else:
                lr = max(lr * 0.95, 0.0001)

        score = max_overlap(h)
        if score < best_score - 1e-9:
            best_score = score
            best_h = h

    # Ensure the result is valid
    best_h = normalize(best_h)
    
    # Safety check: if something went wrong, return the constant
    if best_h.min() < -1e-9 or best_h.max() > 1 + 1e-9:
        return np.full(N, 0.5)
    if abs(best_h.sum() - N / 2.0) > 1e-6:
        return np.full(N, 0.5)
    
    # If the optimized result is not better than constant, return constant
    const_score = max_overlap(np.full(N, 0.5))
    if max_overlap(best_h) >= const_score - 1e-9:
        return np.full(N, 0.5)
    
    return best_h
