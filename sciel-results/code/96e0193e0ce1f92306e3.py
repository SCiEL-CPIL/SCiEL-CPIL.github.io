import numpy as np
import time

N = 100

def _time_remaining(start, deadline):
    return deadline - time.time()

def _compute_all_corrs_fft(h):
    """Compute all cyclic cross-correlations C[k] = sum_i h[i] * h[(i+k) % N] using FFT."""
    # C[k] = <h, roll(h, k)> = sum_i h[i] * h[(i+k)%N]
    # This is the circular cross-correlation of h with itself, i.e., autocorrelation.
    # Using FFT: C = ifft(fft(h) * conj(fft(h)))  -- but we need the right convention.
    # Actually, C[k] = sum_i h[i] * h[(i+k)%N] = (h * reverse(h))[k] in circular convolution sense.
    # Let's use: C[k] = real(ifft(fft(h) * conj(fft(h))))  gives the circular autocorrelation
    # where C[k] = sum_j h[j] * conj(h[j-k]) for complex, but for real it's sum_j h[j] h[j-k].
    # We want C[k] = sum_i h[i] h[i+k].
    # Note: C[k] = C[-k] for real signals (autocorrelation is even).
    # Let's just compute it directly to be safe for small N, or use the right FFT trick.
    # Actually for N=100, direct computation is fast enough.
    # But let's use FFT for efficiency in case we increase N or iterations.
    
    H = np.fft.fft(h)
    # The circular autocorrelation r[k] = sum_n x[n] x[n+k] can be computed as:
    # r = ifft(fft(x) * conj(fft(x)))  gives r[k] = sum_n x[n] conj(x[n-k])
    # For real x, r[k] = sum_n x[n] x[n-k]
    # We want C[k] = sum_n x[n] x[n+k] = r[-k] = r[N-k] (for k != 0), r[0] = r[0]
    # Actually r[k] = sum_n x[n] x[n-k], so r[-k] = sum_n x[n] x[n+k] = C[k]
    # Since r is even for real signals: r[k] = r[N-k]
    # So C[k] = r[k] actually? Let's check:
    # r[k] = sum_n x[n] x[n-k]. Let m = n-k, then r[k] = sum_m x[m+k] x[m] = C[k].
    # Yes! So C = ifft(|fft(h)|^2).real
    C = np.fft.ifft(np.abs(H)**2).real
    return C

def _max_overlap(h):
    """Compute the maximum overlap value, which is max_k (N/2 - C[k])."""
    C = _compute_all_corrs_fft(h)
    return N/2.0 - np.min(C)

def _soft_max_overlap(h, alpha):
    """Soft approximation of max overlap for differentiable optimization."""
    C = _compute_all_corrs_fft(h)
    g = N/2.0 - C  # overlap values for each shift
    max_g = np.max(g)
    g_shifted = g - max_g
    soft_max = max_g + (1.0/alpha) * np.log(np.sum(np.exp(alpha * g_shifted)))
    return soft_max

def _soft_max_overlap_grad(h, alpha):
    """Gradient of soft max overlap with respect to h."""
    C = _compute_all_corrs_fft(h)
    g = N/2.0 - C
    max_g = np.max(g)
    g_shifted = g - max_g
    exp_term = np.exp(alpha * g_shifted)
    softmax_weights = exp_term / np.sum(exp_term)
    
    # d(soft_max)/d(g[k]) = softmax_weights[k]
    # d(g[k])/d(h) = -d(C[k])/d(h)
    # C[k] = sum_i h[i] h[(i+k)%N]
    # d(C[k])/d(h[j]) = h[(j+k)%N] + h[(j-k)%N]  (for real signals, both terms contribute)
    # Wait, C[k] = sum_i h[i] h[i+k]. d/dh[j] = h[j+k] (from first term when i=j) + h[j-k] (from second term when i+k=j, i.e., i=j-k)
    # So d(C[k])/d(h[j]) = h[(j+k)%N] + h[(j-k)%N]
    # But for k=0: C[0] = sum h[i]^2, d/dh[j] = 2*h[j]. And h[(j+0)%N] + h[(j-0)%N] = 2*h[j]. Consistent.
    
    # grad_soft_max[h[j]] = sum_k softmax_weights[k] * d(g[k])/d(h[j])
    #                      = sum_k softmax_weights[k] * (-d(C[k])/d(h[j]))
    #                      = -sum_k softmax_weights[k] * (h[(j+k)%N] + h[(j-k)%N])
    
    # For k=0, the term h[j+0] + h[j-0] = 2*h[j], and this is weighted by softmax_weights[0]
    # For k>0, h[j+k] + h[j-k]
    
    # We can compute this efficiently using FFT or direct summation.
    # Let's do direct for N=100.
    grad = np.zeros(N)
    for k in range(N):
        w = softmax_weights[k]
        if w == 0:
            continue
        # d(C[k])/d(h[j]) = h[(j+k)%N] + h[(j-k)%N] for all j
        # So contribution to grad[j] from shift k is -w * (h[(j+k)%N] + h[(j-k)%N])
        shift_pos = np.roll(h, -k)  # h[(j+k)%N] for each j
        shift_neg = np.roll(h, k)   # h[(j-k)%N] for each j
        grad -= w * (shift_pos + shift_neg)
    
    return grad

def _project_to_box_simplex(v, lower=0.0, upper=1.0, target_sum=None):
    """Project v onto the box [lower, upper]^N with sum = target_sum."""
    if target_sum is None:
        target_sum = N/2.0
    
    # First project to box
    v_clipped = np.clip(v, lower, upper)
    
    # Then adjust to meet sum constraint using a water-filling approach
    # We need to find lambda such that sum(clip(v - lambda, lower, upper)) = target_sum
    # This is a 1D root-finding problem.
    
    def sum_at_lambda(lam):
        return np.sum(np.clip(v - lam, lower, upper))
    
    # Find bounds for lambda
    # If sum(v_clipped) == target_sum, we're done
    if np.allclose(np.sum(v_clipped), target_sum, atol=1e-12):
        return v_clipped
    
    # Binary search for lambda
    lo = lower - np.max(v) - 1.0
    hi = upper - np.min(v) + 1.0
    
    for _ in range(100):
        mid = (lo + hi) / 2.0
        s = sum_at_lambda(mid)
        if np.abs(s - target_sum) < 1e-12:
            break
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    
    lam = (lo + hi) / 2.0
    result = np.clip(v - lam, lower, upper)
    
    # Fine-tune to get exact sum
    diff = target_sum - np.sum(result)
    # Distribute the difference to elements that are not at bounds
    interior = (result > lower + 1e-12) & (result < upper - 1e-12)
    if np.any(interior):
        result[interior] += diff / np.sum(interior)
        result = np.clip(result, lower, upper)
        # Re-adjust if needed
        diff2 = target_sum - np.sum(result)
        if np.abs(diff2) > 1e-10:
            # Add/subtract from the element closest to the middle
            mid_idx = np.argmin(np.abs(result - 0.5))
            result[mid_idx] += diff2
            result = np.clip(result, lower, upper)
    
    return result

def _bang_bang_init(n_ones):
    """Create a bang-bang initialization with n_ones entries set to 1 and rest to 0, 
    then adjust to have sum = N/2."""
    # Start with a random pattern
    idx = np.random.choice(N, n_ones, replace=False)
    h = np.zeros(N)
    h[idx] = 1.0
    return h

def construct():
    start_time = time.time()
    deadline = 148  # 2s margin from 150s limit
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Strategy: Multiple restarts with different initializations, 
    # then gradient descent with soft max objective, then polishing.
    
    # Diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.full(N, 0.5))
    
    # 2. Bang-bang patterns (sum must be N/2 = 50)
    # Try different arrangements of 50 ones
    for trial in range(20):
        h_bb = np.zeros(N)
        # Random permutation of first 50 ones
        idx = np.random.permutation(N)[:50]
        h_bb[idx] = 1.0
        inits.append(h_bb)
    
    # 3. Block patterns
    for block_size in [1, 2, 3, 4, 5]:
        h_block = np.zeros(N)
        # Alternate blocks of 1 and 0
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        # Adjust to have sum = N/2
        if np.sum(h_block) != N/2:
            # Find where to flip
            current_sum = np.sum(h_block)
            if current_sum > N/2:
                # Set some 1s to 0
                ones = np.where(h_block == 1)[0]
                to_flip = np.random.choice(ones, int(current_sum - N/2), replace=False)
                h_block[to_flip] = 0
            else:
                zeros = np.where(h_block == 0)[0]
                to_flip = np.random.choice(zeros, int(N/2 - current_sum), replace=False)
                h_block[to_flip] = 1
        inits.append(h_block)
    
    # 4. Random smooth patterns
    for _ in range(10):
        # Generate random smooth pattern
        freqs = np.arange(N)
        coeffs = np.random.randn(N) * 0.3
        coeffs[0] = 0  # Remove DC offset
        h_rand = np.real(np.fft.ifft(coeffs * np.exp(2j * np.pi * np.random.randn(N) * 0.5)))
        h_rand = 0.5 + h_rand
        h_rand = np.clip(h_rand, 0, 1)
        # Normalize to have sum = N/2
        h_rand = _project_to_box_simplex(h_rand, 0, 1, N/2.0)
        inits.append(h_rand)
    
    # 5. Fourier-based: start with a power spectrum that has low autocorrelation peaks
    for _ in range(10):
        # Random non-negative power spectrum
        P = np.abs(np.random.randn(N) + 1j * np.random.randn(N))**2
        # Normalize so that sum(h) = N/2
        # h = ifft(P) should be real, non-negative, bounded
        h_fourier = np.fft.ifft(P).real
        # This might not be in [0,1], so project
        h_fourier = _project_to_box_simplex(h_fourier, 0, 1, N/2.0)
        inits.append(h_fourier)
    
    # Optimize each initialization
    alphas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    for init_h in inits:
        if time.time() > deadline - 5.0:
            break
        
        h = init_h.copy()
        
        # Phase 1: Gradient descent with soft max
        for alpha in alphas:
            if time.time() > deadline - 3.0:
                break
            
            # Adaptive learning rate
            lr = 0.1
            current_score = _soft_max_overlap(h, alpha)
            
            for it in range(500):
                if time.time() > deadline - 3.0:
                    break
                
                grad = _soft_max_overlap_grad(h, alpha)
                
                # Gradient descent step
                h_new = h - lr * grad
                h_new = _project_to_box_simplex(h_new, 0, 1, N/2.0)
                
                new_score = _soft_max_overlap(h_new, alpha)
                
                if new_score < current_score:
                    h = h_new
                    current_score = new_score
                    lr = min(lr * 1.05, 0.5)
                else:
                    lr *= 0.5
                    if lr < 1e-6:
                        break
                
                # Convergence check
                if np.linalg.norm(grad) < 1e-8:
                    break
            
            # Update best
            actual_score = _max_overlap(h)
            if actual_score < best_score:
                best_score = actual_score
                best_h = h.copy()
        
        # Phase 2: Coordinate descent polish
        if time.time() > deadline - 2.0:
            break
        
        h = best_h.copy() if best_score < _max_overlap(h) else h
        current_score = _max_overlap(h)
        
        # Coordinate descent: adjust individual elements
        for _ in range(200):
            if time.time() > deadline - 1.0:
                break
            
            improved = False
            for j in range(N):
                if time.time() > deadline - 1.0:
                    break
                
                # Try setting h[j] to various values
                for candidate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    h_trial = h.copy()
                    h_trial[j] = candidate
                    # Adjust other elements to maintain sum
                    diff = N/2.0 - np.sum(h_trial)
                    if np.abs(diff) < 1e-12:
                        pass
                    else:
                        # Distribute to other elements proportionally
                        others = np.arange(N) != j
                        h_trial[others] += diff / np.sum(others)
                        h_trial = np.clip(h_trial, 0, 1)
                        # Re-adjust if clipping changed the sum
                        diff2 = N/2.0 - np.sum(h_trial)
                        if np.abs(diff2) > 1e-10:
                            h_trial[j] += diff2
                            h_trial[j] = np.clip(h_trial[j], 0, 1)
                    
                    trial_score = _max_overlap(h_trial)
                    if trial_score < current_score - 1e-10:
                        h = h_trial
                        current_score = trial_score
                        improved = True
                        break
            
            if not improved:
                break
        
        # Update best
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
    
    # Final: ensure exact sum
    best_h = _project_to_box_simplex(best_h, 0, 1, N/2.0)
    
    return best_h

if __name__ == "__main__":
    h = construct()
    C = _compute_all_corrs_fft(h)
    min_C = np.min(C)
    max_overlap = N/2.0 - min_C
    print(f"Min C: {min_C}, Max Overlap: {max_overlap}")
    print(f"Sum h: {np.sum(h)}")
    print(f"Min h: {np.min(h)}, Max h: {np.max(h)}")
    print(f"Score (value): {max_overlap}")
