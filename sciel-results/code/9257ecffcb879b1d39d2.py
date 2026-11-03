"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.fft import fft, ifft

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # N determines the resolution. Higher N allows finer control but slower FFT.
    # N=256 is fast enough for FFT and allows good precision.
    N = 256
    step_width = 2.0 / N
    target_sum = N / 2.0
    
    # Precompute FFT setup for efficiency
    # We use linear convolution via FFT.
    # Size of FFT: needs to be at least 2*N for linear convolution of two N-length signals.
    # Let's use 2*N to be precise, or 4*N for safety. 2*N is sufficient for correlation.
    fft_size = 2 * N
    
    def compute_autocorr(h):
        """
        Compute the discrete autocorrelation C_k = sum_i h_i * h_{i+k} for k in 0..N-1.
        This approximates the integral overlap.
        """
        # Zero-pad h to fft_size
        h_padded = np.zeros(fft_size)
        h_padded[:N] = h
        
        H = fft(h_padded, n=fft_size)
        # Autocorrelation is IFFT(|H|^2)
        corr = np.real(ifft(H * np.conj(H), n=fft_size))
        
        # The correlation array 'corr' has length fft_size.
        # corr[k] corresponds to sum_i h_i h_{i+k} for positive lags k (in circular sense, but we zero-padded so it's linear for k < N).
        # Actually, for zero-padded signals, the first N elements of the IFFT result (shifted appropriately) give the linear autocorrelation.
        # Standard linear autocorrelation R[k] = sum_{i=0}^{N-1-k} x[i] x[i+k].
        # The IFFT of |H|^2 gives circular correlation. With zero padding of size 2N, the first N values correspond to the linear correlation for lags 0 to N-1.
        # Let's verify:
        # x = [a, b, 0, 0]
        # X = [a+b, a-b, -a-b, b-a] ...
        # |X|^2 IFFT -> [a^2+b^2, 0, a^2+b^2, 0] ...
        # Wait, linear autocorrelation of [a,b] is:
        # lag 0: a^2 + b^2
        # lag 1: a*b
        # lag -1: b*a
        # The IFFT of |FFT(x_padded)|^2 gives:
        # index 0: a^2+b^2
        # index 1: a*b
        # index 2: 0 (wrap around)
        # index 3: b*a
        # So for positive lags k=0..N-1, we take corr[0:N].
        # For negative lags, we would take corr[fft_size-N+1:], but due to symmetry of autocorrelation, C[k] = C[-k].
        # The problem asks for max overlap of h with shifted 1-h.
        # Overlap J(s) = 1 - C(s).
        # We want to minimize max J(s), which is 1 - min C(s).
        # So we want to maximize min C(s).
        # The shifts s are in [0, 2]. Discrete lags k=0..N correspond to s=k*step_width.
        # Due to symmetry, we only need to check k=0..N//2 potentially, but let's check all 0..N-1 to be safe, 
        # although C[k] = C[N-k] is not generally true for finite length non-periodic signals unless we define it carefully.
        # However, the physical shift s and 2-s are symmetric for the integral on [0,2] if we consider the geometry.
        # Let's just evaluate C[k] for k in 0..N-1.
        
        return corr[:N]

    def get_gradient_for_lag(k, h):
        """
        Compute gradient of C_k = sum_i h_i h_{i+k} with respect to h.
        C_k = sum_{i=0}^{N-1-k} h_i h_{i+k}
        dC_k/dh_j = h_{j+k} (if j+k < N) + h_{j-k} (if j-k >= 0)
        """
        grad = np.zeros(N)
        # Term 1: h_j appears as the first element in pair (j, j+k)
        # Valid for j such that j+k < N => j < N-k
        if k > 0:
            grad[:N-k] += h[k:]
        # Term 2: h_j appears as the second element in pair (j-k, j)
        # Valid for j such that j-k >= 0 => j >= k
        if k > 0:
            grad[k:] += h[:N-k]
        # If k==0, C_0 = sum h_i^2, grad = 2*h.
        # The formula above for k=0:
        # grad[:N] += h[0:] => grad += h
        # grad[0:] += h[:N] => grad += h
        # So grad = 2h. Correct.
        
        return grad

    def project(h):
        """
        Project h onto {x | 0 <= x_i <= 1, sum(x) = N/2}
        """
        h = np.asarray(h, dtype=float)
        target_sum = N / 2.0
        
        # Binary search for the threshold tau such that sum(clip(h - tau, 0, 1)) = target_sum
        lo = float(h.min()) - 1.0
        hi = float(h.max()) + 1.0
        
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            proj = np.clip(h - mid, 0.0, 1.0)
            s = np.sum(proj)
            if s > target_sum:
                lo = mid
            else:
                hi = mid
        
        tau = 0.5 * (lo + hi)
        h_proj = np.clip(h - tau, 0.0, 1.0)
        
        # Final adjustment for precision
        diff = target_sum - np.sum(h_proj)
        if abs(diff) > 1e-12:
            # Simple additive adjustment, then re-clip
            h_proj = h_proj + diff / N
            h_proj = np.clip(h_proj, 0.0, 1.0)
            
        return h_proj

    # Initializations
    rng = np.random.default_rng(42)
    inits = []
    
    # 1. Uniform
    inits.append(np.full(N, 0.5))
    
    # 2. Bang-bang random
    for _ in range(5):
        bb = np.zeros(N)
        bb[rng.choice(N, N // 2, replace=False)] = 1.0
        inits.append(bb)
        
    # 3. Structured periodic
    for period in [2, 3, 4, 5, 8, 16]:
        bb = np.zeros(N)
        ones = period // 2
        if ones == 0: continue
        pattern = np.zeros(period)
        pattern[:ones] = 1.0
        bb = np.tile(pattern, N // period + 1)[:N]
        # Adjust sum
        current_sum = np.sum(bb)
        target_sum = N / 2.0
        if current_sum > target_sum:
            idxs = np.where(bb==1)[0]
            to_remove = rng.choice(idxs, int(current_sum - target_sum), replace=False)
            bb[to_remove] = 0
        elif current_sum < target_sum:
            idxs = np.where(bb==0)[0]
            to_add = rng.choice(idxs, int(target_sum - current_sum), replace=False)
            bb[to_add] = 1
        inits.append(bb)
        
    # 4. Cosine waves
    idx = np.arange(N)
    for f in [1, 2, 3, 5, 10]:
        h_wave = 0.5 + 0.4 * np.cos(2.0 * np.pi * f * idx / N)
        inits.append(h_wave)

    best_h = None
    best_val = float('inf')
    
    # Optimization Loop for each init
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        h = project(init.copy())
        
        # Adaptive learning rate
        lr = 0.05
        best_h_local = h.copy()
        best_val_local = float('inf')
        
        # Pre-compute C for initial h
        C = compute_autocorr(h)
        min_val = np.min(C)
        best_val_local = 1.0 - min_val # Overlap is 1 - C
        
        # Track active set (lag with min C)
        active_lags = [np.argmin(C)]
        
        for it in range(1000):
            if time.time() > deadline - 5:
                break
                
            # Recompute C if necessary or use cached? 
            # For gradient step, we need C to find the min lag.
            C = compute_autocorr(h)
            min_val = np.min(C)
            current_overlap = 1.0 - min_val
            
            if current_overlap < best_val_local:
                best_val_local = current_overlap
                best_h_local = h.copy()
                
            # Find all lags that are close to the minimum (active set)
            # This stabilizes the gradient
            threshold = min_val + 1e-4
            active_indices = np.where(C <= threshold)[0]
            
            # If active set is too large, just take the single minimum
            if len(active_indices) > 10:
                active_indices = np.array([np.argmin(C)])
            
            # Compute average gradient over active set
            grad = np.zeros(N)
            for k in active_indices:
                grad += get_gradient_for_lag(k, h)
            grad /= len(active_indices)
            
            # Gradient ascent on C (to maximize min C)
            # h_new = h + lr * grad
            h_trial = h + lr * grad
            h_new = project(h_trial)
            
            # Check if we improved
            C_new = compute_autocorr(h_new)
            min_val_new = np.min(C_new)
            overlap_new = 1.0 - min_val_new
            
            if overlap_new < current_overlap:
                h = h_new
                # Increase lr slightly if progress
                lr = min(lr * 1.05, 0.5)
            else:
                # Decrease lr if no progress
                lr *= 0.8
                if lr < 1e-5:
                    break
                    
        if best_val_local < best_val:
            best_val = best_val_local
            best_h = best_h_local.copy()
            
    # Final Polish: Coordinate Descent on the best solution
    if best_h is not None and time.time() < deadline - 2:
        h = best_h.copy()
        C = compute_autocorr(h)
        min_val = np.min(C)
        best_val = 1.0 - min_val
        
        for _ in range(500):
            if time.time() > deadline - 1:
                break
                
            # Find worst lag
            k_worst = np.argmin(C)
            
            # Try small perturbations on a few random indices
            # Or just do a single gradient step with very small lr
            grad = get_gradient_for_lag(k_worst, h)
            
            # Try both directions? No, we want to increase C[k_worst].
            # So we go in direction of grad.
            for lr in [0.01, 0.001, 0.0001]:
                h_trial = project(h + lr * grad)
                C_trial = compute_autocorr(h_trial)
                min_val_trial = np.min(C_trial)
                overlap_trial = 1.0 - min_val_trial
                
                if overlap_trial < best_val:
                    h = h_trial
                    C = C_trial
                    min_val = min_val_trial
                    best_val = overlap_trial
                    break
            
    # Ensure exact sum constraint for return
    if best_h is not None:
        best_h = project(best_h)
        # One last check
        C_final = compute_autocorr(best_h)
        # The score is the max overlap. 
        # The task says "score is the upper bound on C5". 
        # Usually, the evaluator computes the actual max overlap.
        # We return the heights.
        
    return best_h

# EVOLVE-BLOCK-END
