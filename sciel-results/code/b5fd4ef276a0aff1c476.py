import numpy as np
import time

N = 400

def _calc_acf(s):
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    def _eval(s):
        s_f = s.astype(np.float64)
        F = np.fft.rfft(s_f, n=n_fft)
        P = F * np.conj(F)
        acf = np.fft.irfft(P, n=n_fft)
        A = np.real(acf[:N])
        val = float(np.max(np.abs(A[1:])))
        return A, val
    
    def _spectral_init(freq, phase):
        x = np.linspace(0, 1, N, endpoint=False)
        sig = np.sin(2 * np.pi * freq * x + phase)
        s = np.sign(sig)
        s[s == 0] = 1
        return s.astype(np.int8)
    
    def _greedy_exact(s, A, val, max_iters=500):
        """Exact greedy with vectorized delta computation."""
        s = s.copy().astype(np.int8)
        A = A.copy().astype(np.float64)
        val = val
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            best_i = -1
            best_new_val = val
            
            for i in range(N):
                # Compute delta_A for flipping bit i
                # For k >= 1:
                #   delta_A[k] = -2*s[i]*s[i+k] (if i+k < N) + (-2*s[i-k]*s[i] if k <= i)
                # We only need to check k where delta_A[k] != 0
                # k range: 1 to max(i, N-1-i)
                
                k_max = max(i, N - 1 - i)
                if k_max < 1:
                    continue
                
                # Create delta_A for k=1..k_max
                delta_A = np.zeros(k_max + 1)  # index 0..k_max, we use 1..k_max
                
                si = s[i]
                
                # j=i term: k=1 to min(N-1-i, k_max)
                k_end1 = min(N - 1 - i, k_max)
                if k_end1 >= 1:
                    delta_A[1:k_end1+1] += -2 * si * s[i+1:i+1+k_end1]
                
                # j=i-k term: k=1 to min(i, k_max)
                k_end2 = min(i, k_max)
                if k_end2 >= 1:
                    delta_A[1:k_end2+1] += -2 * s[i-k_end2:i][::-1] * si
                
                # New A for affected k
                A_new_affected = A[1:k_max+1] + delta_A[1:k_max+1]
                
                # The unaffected k have |A[k]| <= val, and the max of unaffected is at most val
                # So new_val = max(max(|A_new_affected|), max(|A[unaffected]|))
                # max(|A[unaffected]|) <= val, so new_val = max(max(|A_new_affected|), val)
                # But we need the actual max of unaffected to be precise
                # For speed, use: new_val = max(np.max(np.abs(A_new_affected)), val)
                # This is an upper bound; if it's < val, we definitely improved
                # If it's >= val, we need to check more carefully
                
                new_affected_max = float(np.max(np.abs(A_new_affected)))
                
                # If the affected max is already >= val, the new overall max is at least new_affected_max
                # If new_affected_max < val, the new overall max is val (from unaffected)
                # So new_val = max(new_affected_max, val)
                # But this is only correct if the unaffected max is exactly val.
                # In practice, use this as a lower bound on the new max.
                
                # For correctness, compute the true new max:
                # The unaffected k are those > k_max, so A[k_max+1:]
                if k_max + 1 < N:
                    unaffected_max = float(np.max(np.abs(A[k_max+1:])))
                else:
                    unaffected_max = 0.0
                
                new_val = max(new_affected_max, unaffected_max)
                
                if new_val < best_new_val:
                    best_new_val = new_val
                    best_i = i
            
            if best_i >= 0 and best_new_val < val - 1e-12:
                i = best_i
                s[i] = -s[i]
                A, val = _eval(s)
            else:
                break
        
        return s, A, val
    
    def _greedy_block(s, A, val):
        s = s.copy()
        A = A.copy()
        val = val
        
        block_sizes = [2, 4, 8, 16, 32]
        improved = True
        while improved:
            if time.time() > deadline - 5.0:
                break
            improved = False
            best_move = None
            best_new_val = val
            
            for bs in block_sizes:
                for _ in range(50):
                    start_idx = np.random.randint(0, N - bs + 1)
                    old_block = s[start_idx:start_idx + bs].copy()
                    s[start_idx:start_idx + bs] = -old_block
                    A_new, new_val = _eval(s)
                    if new_val < best_new_val:
                        best_new_val = new_val
                        best_move = (start_idx, bs)
                    else:
                        s[start_idx:start_idx + bs] = old_block
            
            if best_move is not None and best_new_val < val - 1e-12:
                start_idx, bs = best_move
                s[start_idx:start_idx + bs] = -s[start_idx:start_idx + bs]
                A, val = _eval(s)
                improved = True
        
        return s, A, val
    
    def _sa_search(s, A, val):
        T = 30.0
        T_min = 0.05
        s = s.copy()
        A = A.copy()
        val = val
        
        block_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        while time.time() - start_time < deadline and T > T_min:
            r = np.random.random()
            if r < 0.2:
                block_size = 1
            elif r < 0.4:
                block_size = 2
            elif r < 0.6:
                block_size = 4
            elif r < 0.75:
                block_size = 8
            elif r < 0.85:
                block_size = 16
            elif r < 0.92:
                block_size = 32
            else:
                block_size = 64
            
            if block_size > N:
                block_size = N
            
            start_idx = np.random.randint(0, N - block_size + 1)
            old_block = s[start_idx:start_idx + block_size].copy()
            s[start_idx:start_idx + block_size] = -old_block
            
            A_new, val_new = _eval(s)
            delta = val_new - val
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                A = A_new
                val = val_new
            else:
                s[start_idx:start_idx + block_size] = old_block
            
            T *= 0.997
        
        return s, A, val
    
    best_val = float('inf')
    best_seq = np.ones(N, dtype=np.int8)
    
    num_restarts = 0
    
    while time.time() < deadline - 20.0:
        r = np.random.random()
        if r < 0.3:
            freq = np.random.uniform(1, 10)
            phase = np.random.uniform(0, 2 * np.pi)
            s = _spectral_init(freq, phase)
        elif r < 0.5:
            freq = np.random.uniform(0.5, 3)
            phase = np.random.uniform(0, 2 * np.pi)
            s = _spectral_init(freq, phase)
        elif r < 0.7:
            s = np.random.randint(0, 2, size=N) * 2 - 1
        else:
            s = np.random.randint(0, 2, size=N) * 2 - 1
            s_f = s.astype(np.float64)
            s_smooth = np.convolve(s_f, np.ones(5)/5, mode='same')
            s = np.sign(s_smooth)
            s[s == 0] = 1
            s = s.astype(np.int8)
        
        A, val = _eval(s)
        
        remaining = deadline - time.time()
        sa_time = min(remaining * 0.5, 10.0)
        s, A, val = _sa_search(s, A, val)
        
        if time.time() < deadline - 10.0:
            s, A, val = _greedy_exact(s, A, val)
        
        if time.time() < deadline - 5.0:
            s, A, val = _greedy_block(s, A, val)
        
        if val < best_val:
            best_val = val
            best_seq = s.copy()
        
        num_restarts += 1
        
        if time.time() > deadline - 15.0:
            break
    
    if time.time() < deadline - 3.0:
        A_best = _calc_acf(best_seq)
        best_seq, _, best_val = _greedy_exact(best_seq, A_best, best_val)
    
    if time.time() < deadline - 1.0:
        A_best = _calc_acf(best_seq)
        best_seq, _, best_val = _greedy_block(best_seq, A_best, best_val)
    
    return best_seq
