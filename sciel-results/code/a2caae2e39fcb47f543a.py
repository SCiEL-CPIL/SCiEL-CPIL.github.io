"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct step heights h on [0, 2] (N steps, each in [0,1], sum = N/2)
    to minimize the maximum overlap of h with shifted 1-h.
    
    The overlap at shift k is: sum_i h[i] * (1 - h[(i+k) % N])
    We want to minimize max_k overlap_k.
    
    Strategy: Use projected gradient descent on a smooth surrogate.
    The objective is f(h) = max_k sum_i h[i]*(1-h[(i+k)%N])
    
    We use a soft-max (log-sum-exp) surrogate for the max, then do
    projected gradient descent with constraints 0 <= h[i] <= 1, sum(h) = N/2.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Precompute shift indices
    # For shift k, the overlap is sum_i h[i] * (1 - h[(i+k) % N])
    # = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    # = N/2 - sum_i h[i]*h[(i+k)%N]   (since sum h = N/2)
    # So minimizing max overlap = maximizing min_k sum_i h[i]*h[(i+k)%N]
    
    # Actually let's be more careful. The "overlap" of h with shifted 1-h:
    # overlap_k = sum_i h[i] * (1 - h[(i+k) % N])
    # We want to minimize max_k overlap_k
    
    # Let's use a smooth approach: minimize a soft-max of the overlaps
    
    def overlaps(h):
        """Compute overlap at each shift."""
        # overlap_k = sum_i h[i] * (1 - h[(i+k)%N])
        # Use FFT for efficiency: this is a circular cross-correlation
        # overlap_k = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
        # The second term is the circular autocorrelation of h at lag k
        # Actually: sum_i h[i]*h[(i+k)%N] is the circular cross-correlation
        # Let's compute via FFT
        H = np.fft.rfft(h, n=N)
        # circular autocorrelation: ifft(|H|^2) gives sum_i h[i]*h[(i+k)%N]
        ac = np.fft.irfft(H * np.conj(H), n=N)
        # ac[k] = sum_i h[i] * h[(i-k)%N] ... need to check convention
        # Actually np.fft.irfft(|H|^2) gives the circular autocorrelation
        # Let me just compute directly for correctness, N=100 is small
        ov = np.zeros(N)
        for k in range(N):
            shifted = np.roll(h, k)
            ov[k] = np.sum(h * (1.0 - shifted))
        return ov
    
    def soft_max(x, tau=0.01):
        """Smooth approximation of max(x) using log-sum-exp."""
        m = np.max(x)
        return m + tau * np.log(np.sum(np.exp((x - m) / tau)))
    
    def objective(h, tau=0.01):
        ov = overlaps(h)
        return soft_max(ov, tau)
    
    def project(h):
        """Project onto {h : 0 <= h[i] <= 1, sum(h) = N/2}."""
        # Simple approach: clip, then adjust to meet sum constraint
        h = np.clip(h, 0.0, 1.0)
        # Adjust to make sum = N/2
        for _ in range(50):
            s = np.sum(h)
            diff = N / 2.0 - s
            if abs(diff) < 1e-12:
                break
            # Distribute the difference, respecting bounds
            if diff > 0:
                # Need to increase; add to elements below 1
                mask = h < 1.0
                if not np.any(mask):
                    break
                per = diff / np.sum(mask)
                h[mask] = np.minimum(h[mask] + per, 1.0)
            else:
                # Need to decrease; subtract from elements above 0
                mask = h > 0.0
                if not np.any(mask):
                    break
                per = -diff / np.sum(mask)
                h[mask] = np.maximum(h[mask] - per, 0.0)
        return h
    
    # Initialize with a non-trivial pattern
    # Start with alternating high/low
    h = np.zeros(N)
    h[::2] = 1.0
    h[1::2] = 0.0
    # This gives sum = 50 = N/2. Good.
    
    best_h = h.copy()
    best_score = np.max(overlaps(h))
    
    # Projected gradient descent with annealing of tau
    tau_start = 0.1
    tau_end = 0.005
    lr_start = 0.05
    lr_end = 0.001
    n_iters = 5000
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Anneal parameters
        frac = it / max(n_iters - 1, 1)
        tau = tau_start * (tau_end / tau_start) ** frac
        lr = lr_start * (lr_end / lr_start) ** frac
        
        # Compute gradient of soft-max of overlaps
        ov = overlaps(h)
        m = np.max(ov)
        exps = np.exp((ov - m) / tau)
        weights = exps / np.sum(exps)  # softmax weights over shifts
        
        # Gradient of overlap_k w.r.t. h[i] = 1 - h[(i+k)%N]
        # d(overlap_k)/dh[i] = 1 - h[(i+k)%N]
        # Total gradient = sum_k weights[k] * (1 - h[(i+k)%N])
        grad = np.zeros(N)
        for k in range(N):
            if weights[k] > 1e-15:
                shifted = np.roll(h, k)
                grad += weights[k] * (1.0 - shifted)
        
        # Gradient step
        h_new = h - lr * grad
        
        # Project
        h_new = project(h_new)
        
        # Evaluate
        score = np.max(overlaps(h_new))
        if score < best_score:
            best_score = score
            best_h = h_new.copy()
        
        h = h_new
    
    # Final polish: try a few different initializations if time permits
    if time.time() < deadline - 20:
        # Try random initialization
        np.random.seed(42)
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            h_try = np.random.rand(N)
            h_try = project(h_try)
            
            # Quick optimization
            for it in range(2000):
                if time.time() > deadline - 5:
                    break
                frac = it / 2000.0
                tau = 0.05 * (0.005 / 0.05) ** frac
                lr = 0.03 * (0.001 / 0.03) ** frac
                
                ov = overlaps(h_try)
                m = np.max(ov)
                exps = np.exp((ov - m) / tau)
                weights = exps / np.sum(exps)
                
                grad = np.zeros(N)
                for k in range(N):
                    if weights[k] > 1e-15:
                        shifted = np.roll(h_try, k)
                        grad += weights[k] * (1.0 - shifted)
                
                h_try = project(h_try - lr * grad)
            
            score_try = np.max(overlaps(h_try))
            if score_try < best_score:
                best_score = score_try
                best_h = h_try.copy()
    
    # Ensure exact constraints
    best_h = project(best_h)
    
    # Verify sum
    s = np.sum(best_h)
    if abs(s - N/2.0) > 1e-10:
        # Force correction
        diff = N/2.0 - s
        # Add to elements with most room
        idx = np.argsort(best_h)[:int(abs(diff)) + 1]
        if diff > 0:
            for i in idx:
                if best_h[i] < 1.0:
                    add = min(1.0 - best_h[i], diff)
                    best_h[i] += add
                    diff -= add
                    if diff <= 1e-12:
                        break
        else:
            idx = np.argsort(-best_h)[:int(abs(diff)) + 1]
            for i in idx:
                if best_h[i] > 0.0:
                    sub = min(best_h[i], -diff)
                    best_h[i] -= sub
                    diff += sub
                    if diff >= -1e-12:
                        break
    
    return best_h
# EVOLVE-BLOCK-END
