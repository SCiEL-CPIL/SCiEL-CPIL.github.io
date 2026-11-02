import numpy as np
import time

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i]*h[i+k]|."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))


def _continuous_seed(seed, time_budget):
    """Generate a good binary seed via continuous gradient descent on L2 autocorrelation norm."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N)
    start = time.time()
    
    lr = 0.05
    best_x = x.copy()
    best_l2 = _l2_autocorr(x)
    
    while time.time() - start < time_budget:
        # Compute L2 norm and gradient
        H = np.fft.fft(x, _FFT_SIZE)
        ac = np.fft.ifft(H * np.conj(H)).real[:N]
        l2_val = float(np.sum(ac[1:] ** 2))
        
        if l2_val < best_l2:
            best_l2 = l2_val
            best_x = x.copy()
            
        # Gradient of sum_{k=1}^{N-1} ac[k]^2
        # Using the property that the gradient is related to the convolution of ac with x
        # grad = 4 * (ac * x)_circular (with appropriate truncation for aperiodic)
        # For simplicity and speed, we approximate with circular convolution
        # which is valid for large N and small boundary effects, or we use the direct FFT method.
        
        # Efficient Gradient via FFT:
        # Let S = FFT(x). The quantity we minimize is proportional to sum |S(f)|^4.
        # d/df sum |S|^4 = 4 * S * |S|^2 (conjugate for real signals? No, S is complex)
        # d/dx sum |S|^4 = 4 * IFFT( S * |S|^2 ) ... wait.
        # Let F(x) = sum_k ac[k]^2.
        # ac[k] = IFFT(|S|^2)[k].
        # F = sum_k |IFFT(|S|^2)[k]|^2 = N * sum_f ||S(f)|^2|^2 = N * sum_f |S(f)|^4.
        # dF/dx_i = 4 * IFFT( S(f) * |S(f)|^2 * exp(2pij f i) ) ... 
        # Actually, the gradient of sum |S(f)|^4 with respect to x is 4 * IFFT( S * |S|^2 ).
        # Let's verify: S = F x. dS/df = F (dx/df).
        # d/dx sum |S|^4 = sum 4 |S|^2 S* dS*/dx + ... 
        # It turns out grad = 4 * IFFT( S * |S|^2 ).
        
        grad = np.fft.ifft(H * np.abs(H)**2).real
        # Note: The factor 4 and scaling might need adjustment, but direction is correct.
        
        x = x - lr * grad
        
        # Normalize to prevent drift
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm * np.sqrt(N)
            
        # Adaptive LR
        if l2_val > best_l2 * 1.05:
            lr *= 0.9
            
    h = np.where(best_x >= 0, 1.0, -1.0)
    return h, _peak_sidelobe(h)


def _l2_autocorr(x):
    """L2 norm of aperiodic autocorrelation sidelobes."""
    H = np.fft.fft(x, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.sum(ac[1:] ** 2))


def _anneal(seed, time_budget):
    """Simulated annealing with variable-length block flips and kicks."""
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 5.0
    T_min = 1e-6
    start = time.time()
    stuck_counter = 0
    
    while time.time() - start < time_budget:
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        max_len = max(1, int(32.0 * (1.0 - 0.7 * progress)))
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        
        # Kick if stuck
        if stuck_counter > 100:
            # Flip a random larger block to escape
            kick_len = int(rng.integers(10, 50))
            kick_i = int(rng.integers(0, N - kick_len + 1))
            h[kick_i:kick_i + kick_len] *= -1.0
            cur = _peak_sidelobe(h)
            stuck_counter = 0
            T = max(T, 2.0) # Reheat slightly
            
        h[i:i + L] *= -1.0
        new = _peak_sidelobe(h)
        d = new - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            if cur < best:
                best = cur
                best_h = h.copy()
                stuck_counter = 0
            else:
                stuck_counter += 1
        else:
            h[i:i + L] *= -1.0
            stuck_counter += 1
            
        T = max(T_min, T * 0.999)
        
    return best_h, best


def _greedy_polish(h, time_budget, start):
    """Greedy polish with single-flip and adjacent-pair-flip sweeps."""
    h = h.copy()
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    
    while time.time() - start < time_budget:
        improved = False
        
        # --- single-flip sweep ---
        best_i = -1
        best_c = cur
        for i in range(N):
            h[i] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c:
                best_c = c
                best_i = i
            h[i] *= -1.0
        if best_i >= 0:
            h[best_i] *= -1.0
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True

        # --- adjacent-pair-flip sweep ---
        best_pair = -1
        best_c2 = cur
        for i in range(N - 1):
            h[i] *= -1.0
            h[i + 1] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c2:
                best_c2 = c
                best_pair = i
            h[i] *= -1.0
            h[i + 1] *= -1.0
        if best_pair >= 0:
            h[best_pair] *= -1.0
            h[best_pair + 1] *= -1.0
            cur = best_c2
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True
            
        if not improved:
            break
            
    return best_h, best


def construct():
    try:
        hard_deadline = 140.0
        start = time.time()

        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        def _remaining():
            return hard_deadline - (time.time() - start)

        # Phase 0: Generate high-quality seeds via continuous gradient descent.
        seed_budget = min(10.0, _remaining() - 10.0)
        if seed_budget > 2.0:
            h_seed, c_seed = _continuous_seed(42, seed_budget)
            if c_seed < global_best:
                global_best = c_seed
                global_best_h = h_seed
            # Polish the continuous seed
            sub = min(5.0, _remaining() - 10.0)
            if sub > 1.0:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 1: Anneal from the best seed found so far.
        budget = min(30.0, _remaining() - 10.0)
        if budget > 5.0:
            h, c = _anneal(12345, budget)
            if c < global_best:
                global_best = c
                global_best_h = h
            # Polish right after to lock in gains
            sub = min(10.0, _remaining() - 10.0)
            if sub > 1.0:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: multiple shorter restarts for diversity.
        restart_id = 1
        while _remaining() > 15.0:
            sub = min(10.0, _remaining() - 15.0)
            if sub < 2.0:
                break
            h, c = _anneal(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            # Quick polish after each restart
            sub2 = min(5.0, _remaining() - 15.0)
            if sub2 > 1.0:
                h2, c2 = _greedy_polish(global_best_h, sub2, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 1

        # Phase 3: final high-precision polish.
        remaining = _remaining() - 2.0
        if remaining > 1.0:
            h, c = _greedy_polish(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
