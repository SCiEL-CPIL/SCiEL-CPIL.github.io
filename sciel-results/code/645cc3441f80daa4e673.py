import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _binary_sequence_search():
    """Search for good binary sequences with balanced sum."""
    best_h = None
    best_val = np.inf
    
    # Try various binary patterns
    # Pattern 1: Alternating
    h_alt = np.array([0.0, 1.0] * (N // 2))
    if len(h_alt) < N:
        h_alt = np.concatenate([h_alt, [0.0]])
    best_h = h_alt.copy()
    best_val = _max_overlap(h_alt)
    
    # Pattern 2: Blocks of 1s and 0s
    for block_size in [2, 3, 4, 5, 7, 10]:
        pattern = []
        while len(pattern) < N:
            pattern.extend([1.0] * block_size)
            pattern.extend([0.0] * block_size)
        h_block = np.array(pattern[:N])
        val = _max_overlap(h_block)
        if val < best_val:
            best_val = val
            best_h = h_block.copy()
    
    # Pattern 3: Pseudo-random binary with balanced sum
    np.random.seed(123)
    for seed in range(20):
        np.random.seed(seed)
        h_rand = np.random.randint(0, 2, N).astype(float)
        # Adjust to have exactly N/2 ones
        n_ones = np.sum(h_rand)
        if n_ones > N / 2:
            # Flip some 1s to 0s
            ones_idx = np.where(h_rand == 1.0)[0]
            to_flip = np.random.choice(ones_idx, int(n_ones - N / 2), replace=False)
            h_rand[to_flip] = 0.0
        elif n_ones < N / 2:
            # Flip some 0s to 1s
            zeros_idx = np.where(h_rand == 0.0)[0]
            to_flip = np.random.choice(zeros_idx, int(N / 2 - n_ones), replace=False)
            h_rand[to_flip] = 1.0
        
        val = _max_overlap(h_rand)
        if val < best_val:
            best_val = val
            best_h = h_rand.copy()
    
    return best_h, best_val

def _autocorr_from_h(h):
    """Compute circular autocorrelation of h via FFT."""
    H = np.fft.fft(h)
    ac = np.fft.ifft(H * np.conj(H)).real
    return ac

def _max_overlap(h):
    """Compute max overlap = max_k (N/2 - autocorr[k])."""
    ac = _autocorr_from_h(h)
    return np.max(N / 2.0 - ac)

def _soft_min(x, tau=0.05):
    """Soft-min via log-sum-exp: -logsumexp(-x/tau)*tau approximates min(x)."""
    x = np.asarray(x)
    m = np.min(x)
    return m - tau * np.log(np.mean(np.exp((x - m) / tau)))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Initial guess: uniform
    h0 = np.full(N, 0.5)
    
    # Try binary sequence search first
    h_binary, val_binary = _binary_sequence_search()
    best_h = h_binary.copy()
    best_val = val_binary
    
    def objective_and_grad(h):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            raise RuntimeError("timeout")
        
        h = np.clip(h, 0.0, 1.0)
        # Enforce sum constraint via projection
        s = np.sum(h)
        if s > 0:
            h = h * (N / 2.0) / s
            h = np.clip(h, 0.0, 1.0)
        
        ac = _autocorr_from_h(h)
        overlaps = N / 2.0 - ac
        
        # Use soft-min as smooth surrogate for min(ac) = -soft_min(overlaps)
        tau = 0.05
        m = np.min(overlaps)
        exp_vals = np.exp((overlaps - m) / tau)
        mean_exp = np.mean(exp_vals)
        soft_min_val = m - tau * np.log(mean_exp)
        
        # We want to minimize soft_min_val (which approximates min overlap)
        # Gradient: d(soft_min)/d(overlaps_i) = exp_vals[i] / (N * mean_exp)
        # overlaps_i = N/2 - ac_i, so d(overlaps_i)/d(h) = -d(ac_i)/d(h)
        
        # Gradient of autocorrelation wrt h:
        # ac = ifft(|FFT(h)|^2)
        # d(ac_k)/d(h_j) can be computed via FFT
        # ac_k = sum_j h_j * h_{j-k} (circular)
        # d(ac_k)/d(h_j) = h_{j-k} + h_{j+k} (circular)
        
        # So d(overlaps_k)/d(h_j) = -(h_{j-k} + h_{j+k})
        # grad_h = sum_k (exp_vals[k]/(N*mean_exp)) * (-(h_{j-k} + h_{j+k}))
        
        # Using FFT: for each k, the contribution is w_k * (h shifted by -k + h shifted by +k)
        # This is: ifft(FFT(w) * (FFT(h) + FFT(h)))... let me think more carefully.
        
        # Actually: sum_k w_k * h_{j-k} = (w * h)_j (circular convolution)
        # sum_k w_k * h_{j+k} = sum_k w_k * h_{j+k} = (w_rev * h)_j where w_rev[k] = w_{-k}
        
        w = exp_vals / (N * mean_exp)
        
        # Circular convolution: (w * h)[j] = sum_k w[k] * h[j-k]
        # In FFT: ifft(FFT(w) * FFT(h))
        # (w_rev * h)[j] = sum_k w[k] * h[j+k] = sum_k w[-k] * h[j-k] = ifft(FFT(w_rev) * FFT(h))
        # where w_rev[k] = w[-k mod N]
        
        w_rev = np.zeros_like(w)
        w_rev[0] = w[0]
        w_rev[1:] = w[:0:-1]
        
        W = np.fft.fft(w)
        W_rev = np.fft.fft(w_rev)
        H = np.fft.fft(h)
        
        conv1 = np.fft.ifft(W * H).real
        conv2 = np.fft.ifft(W_rev * H).real
        
        grad = -(conv1 + conv2)
        
        return soft_min_val, grad
    
    # Multi-start optimization from binary and continuous initializations
    np.random.seed(42)
    n_starts = 12
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 20.0:
            break
        
        if start_idx == 0:
            h_init = h_binary.copy()
        elif start_idx == 1:
            h_init = h0.copy()
        elif start_idx == 2:
            # Low-frequency cosine wave (inspired by cell findings)
            h_init = 0.5 + 0.3 * np.cos(2 * np.pi * np.arange(N) / N)
            h_init = np.clip(h_init, 0.0, 1.0)
            h_init = h_init * (N / 2.0) / np.sum(h_init)
            h_init = np.clip(h_init, 0.0, 1.0)
        else:
            # Random initialization with mean 0.5
            h_init = np.random.uniform(0, 1, N)
            h_init = (h_init - 0.5) * 2.0  # scale to have mean ~0
            h_init = h_init + 0.5
            h_init = np.clip(h_init, 0.01, 0.99)
            h_init = h_init * (N / 2.0) / np.sum(h_init)
            h_init = np.clip(h_init, 0.0, 1.0)
        
        try:
            result = minimize(
                objective_and_grad,
                h_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.success or result.fun < best_val:
                h_result = np.clip(result.x, 0.0, 1.0)
                # Project to satisfy sum constraint
                s = np.sum(h_result)
                if s > 0:
                    h_result = h_result * (N / 2.0) / s
                    h_result = np.clip(h_result, 0.0, 1.0)
                
                val = _max_overlap(h_result)
                if val < best_val:
                    best_val = val
                    best_h = h_result.copy()
        except Exception:
            pass
    
    # Final polish: tighten with smaller tau
    if time.time() < deadline - 10.0:
        def objective_and_grad_fine(h):
            elapsed = time.time() - start_time
            if elapsed > deadline - 3.0:
                raise RuntimeError("timeout")
            
            h = np.clip(h, 0.0, 1.0)
            s = np.sum(h)
            if s > 0:
                h = h * (N / 2.0) / s
                h = np.clip(h, 0.0, 1.0)
            
            ac = _autocorr_from_h(h)
            overlaps = N / 2.0 - ac
            
            tau = 0.01
            m = np.min(overlaps)
            exp_vals = np.exp((overlaps - m) / tau)
            mean_exp = np.mean(exp_vals)
            soft_min_val = m - tau * np.log(mean_exp)
            
            w = exp_vals / (N * mean_exp)
            w_rev = np.zeros_like(w)
            w_rev[0] = w[0]
            w_rev[1:] = w[:0:-1]
            
            W = np.fft.fft(w)
            W_rev = np.fft.fft(w_rev)
            H = np.fft.fft(h)
            
            conv1 = np.fft.ifft(W * H).real
            conv2 = np.fft.ifft(W_rev * H).real
            
            grad = -(conv1 + conv2)
            
            return soft_min_val, grad
        
        try:
            result = minimize(
                objective_and_grad_fine,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if result.fun < best_val:
                h_result = np.clip(result.x, 0.0, 1.0)
                s = np.sum(h_result)
                if s > 0:
                    h_result = h_result * (N / 2.0) / s
                    h_result = np.clip(h_result, 0.0, 1.0)
                
                val = _max_overlap(h_result)
                if val < best_val:
                    best_val = val
                    best_h = h_result.copy()
        except Exception:
            pass
    
    # Final projection to ensure exact sum
    best_h = np.clip(best_h, 0.0, 1.0)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h * (N / 2.0) / s
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
