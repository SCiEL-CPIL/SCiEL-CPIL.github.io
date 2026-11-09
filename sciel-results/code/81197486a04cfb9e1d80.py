import numpy as np
import time
from scipy.optimize import minimize

N = 100

def compute_score_fast(h):
    """Compute max overlap using FFT-based circular cross-correlation."""
    # overlap(s) = sum_i h_i * (1 - h_{i+s}) = N/2 - sum_i h_i * h_{i+s}
    # sum_i h_i * h_{i+s} is the circular autocorrelation at lag s
    # Using FFT: autocorr = ifft(fft(h) * conj(fft(h)))
    hft = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(hft * np.conj(hft)))
    # autocorr[s] = sum_i h_i * h_{i+s} (circular)
    # We want max over s in 1..N-1 of (N/2 - autocorr[s])
    # = N/2 - min_{s in 1..N-1} autocorr[s]
    # autocorr[0] = sum h_i^2, skip it
    min_ac = np.min(autocorr[1:])
    return N / 2.0 - min_ac

def soft_min_overlap(h, beta=50.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    hft = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(hft * np.conj(hft)))
    # overlaps at shifts 1..N-1
    ov = N / 2.0 - autocorr[1:]
    # soft max via log-sum-exp
    m = np.max(ov)
    soft_max = m + np.log(np.sum(np.exp(beta * (ov - m)))) / beta
    return soft_max

def project_to_binary(h):
    """Project continuous h to binary with exactly N/2 ones, preserving order."""
    # Take top N/2 values as 1s
    threshold_idx = np.argpartition(h, N // 2 - 1)[N // 2 - 1]
    h_bin = np.where(h > threshold_idx, 1.0, 0.0)
    # Handle ties: if more than N/2 are >= threshold, need to be careful
    ones_count = np.sum(h_bin == 1.0)
    if ones_count > N // 2:
        # Reduce: set some 1s to 0
        ones_positions = np.where(h_bin == 1.0)[0]
        # Among the ones, set the smallest to 0
        h_vals_at_ones = h[ones_positions]
        to_zero = ones_positions[np.argsort(h_vals_at_ones)[:ones_count - N // 2]]
        h_bin[to_zero] = 0.0
    elif ones_count < N // 2:
        # Increase: set some 0s to 1
        zeros_positions = np.where(h_bin == 0.0)[0]
        h_vals_at_zeros = h[zeros_positions]
        to_one = zeros_positions[np.argsort(h_vals_at_zeros)[- (N // 2 - ones_count):]]
        h_bin[to_one] = 1.0
    return h_bin

def construct():
    deadline = time.time() + 145
    best_h = None
    best_score = np.inf
    
    np.random.seed(42)
    
    # Phase 1: Continuous optimization with soft-min surrogate
    # Start from several random initial points
    for restart in range(20):
        if time.time() > deadline - 30:
            break
        
        # Random initial point in [0,1]^N with sum = N/2
        # Generate random values, then scale/shift to meet sum constraint
        h0 = np.random.rand(N)
        # Adjust to have sum N/2
        h0 = (h0 - h0.mean()) * 0.5 + 0.5
        h0 = np.clip(h0, 0.0, 1.0)
        # Re-adjust sum
        if np.sum(h0) > 0:
            h0 = h0 * (N / 2.0) / np.sum(h0)
            h0 = np.clip(h0, 0.0, 1.0)
        
        # Simple projected gradient descent
        h_curr = h0.copy()
        lr = 0.1
        beta_schedule = [20.0, 50.0, 100.0, 200.0, 500.0]
        
        for beta in beta_schedule:
            if time.time() > deadline - 30:
                break
            for it in range(200):
                if time.time() > deadline - 30:
                    break
                # Numerical gradient of soft_min_overlap
                eps = 1e-5
                grad = np.zeros(N)
                for i in range(N):
                    h_plus = h_curr.copy()
                    h_plus[i] += eps
                    h_minus = h_curr.copy()
                    h_minus[i] -= eps
                    grad[i] = (soft_min_overlap(h_plus, beta) - soft_min_overlap(h_minus, beta)) / (2 * eps)
                
                # Gradient step
                h_new = h_curr - lr * grad
                
                # Project to [0,1]
                h_new = np.clip(h_new, 0.0, 1.0)
                
                # Project to sum = N/2 (simple: adjust mean)
                diff = N / 2.0 - np.sum(h_new)
                if diff > 0:
                    # Need to increase; add to elements not at 1
                    mask = h_new < 0.999
                    if np.sum(mask) > 0:
                        h_new[mask] += diff / np.sum(mask)
                else:
                    # Need to decrease; subtract from elements not at 0
                    mask = h_new > 0.001
                    if np.sum(mask) > 0:
                        h_new[mask] += diff / np.sum(mask)
                h_new = np.clip(h_new, 0.0, 1.0)
                
                # Ensure sum is approximately N/2
                s = np.sum(h_new)
                if s > 0:
                    h_new = h_new * (N / 2.0) / s
                    h_new = np.clip(h_new, 0.0, 1.0)
                
                h_curr = h_new
                lr *= 0.99
        
        # Snap to binary
        h_bin = project_to_binary(h_curr)
        score = compute_score_fast(h_bin)
        
        if score < best_score:
            best_score = score
            best_h = h_bin.copy()
        
        # Also try the continuous value's score as a reference
        cont_score = compute_score_fast(h_curr)
    
    # Phase 2: Fast discrete local search from the best binary solution
    if best_h is not None:
        h_bin = best_h.copy()
        current_score = best_score
        
        # Simulated annealing on binary sequences
        T = 1.0
        cooling = 0.995
        ones = np.where(h_bin == 1)[0]
        zeros = np.where(h_bin == 0)[0]
        
        while time.time() < deadline and T > 0.01:
            # Pick random swap
            if len(ones) == 0 or len(zeros) == 0:
                break
            i = np.random.choice(ones)
            j = np.random.choice(zeros)
            
            h_bin[i] = 0
            h_bin[j] = 1
            new_score = compute_score_fast(h_bin)
            delta = new_score - current_score
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
                # Update ones and zeros
                ones = np.where(h_bin == 1)[0]
                zeros = np.where(h_bin == 0)[0]
            else:
                h_bin[i] = 1
                h_bin[j] = 0
            
            T *= cooling
            
            if current_score < best_score:
                best_score = current_score
                best_h = h_bin.copy()
        
        # Phase 3: Hill climbing polish
        improved = True
        while improved and time.time() < deadline:
            improved = False
            h_bin = best_h.copy()
            current_score = best_score
            ones = np.where(h_bin == 1)[0]
            zeros = np.where(h_bin == 0)[0]
            
            for _ in range(500):
                if time.time() > deadline:
                    break
                if len(ones) == 0 or len(zzeros) == 0 if len(ones) == 0 else len(zeros) == 0:
                    break
                i = np.random.choice(ones)
                j = np.random.choice(zeros)
                
                h_bin[i] = 0
                h_bin[j] = 1
                new_score = compute_score_fast(h_bin)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    ones = np.where(h_bin == 1)[0]
                    zeros = np.where(h_bin == 0)[0]
                else:
                    h_bin[i] = 1
                    h_bin[j] = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_bin.copy()
    
    # Final validation
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Ensure sum is exactly N/2
    s = np.sum(best_h)
    if abs(s - N / 2.0) > 1e-10:
        # Adjust
        if best_h.sum() > N / 2.0:
            ones_pos = np.where(best_h == 1.0)[0]
            for p in ones_pos:
                if np.sum(best_h) <= N / 2.0:
                    break
                best_h[p] = 0.0
        else:
            zeros_pos = np.where(best_h == 0.0)[0]
            for p in zeros_pos:
                if np.sum(best_h) >= N / 2.0:
                    break
                best_h[p] = 1.0
    
    return best_h
