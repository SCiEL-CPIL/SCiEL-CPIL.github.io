import time
import numpy as np

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 145.0

def _linear_overlap_values(h):
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _compute_r(h):
    n = len(h)
    r = np.zeros(n)
    for k in range(n):
        if k == 0:
            r[k] = np.dot(h, h)
        else:
            r[k] = np.dot(h[k:], h[:-k])
    return r

def _local_search_binary(h):
    h = h.astype(np.float64)
    r = _compute_r(h)
    min_r = np.min(r)
    
    zeros = np.where(h == 0)[0]
    ones = np.where(h == 1)[0]
    
    no_improve_count = 0
    max_no_improve = 100
    
    while no_improve_count < max_no_improve:
        if time.time() > _DEADLINE - 5:
            break
            
        best_swap = None
        best_min_r = min_r
        best_r = r
        
        num_zeros = len(zeros)
        num_ones = len(ones)
        
        for i_idx in range(num_zeros):
            i = zeros[i_idx]
            for j_idx in range(num_ones):
                j = ones[j_idx]
                if i == j: continue
                
                h[i] = 1
                h[j] = 0
                
                new_r = _compute_r(h)
                new_min_r = np.min(new_r)
                
                if new_min_r > best_min_r + 1e-10:
                    best_min_r = new_min_r
                    best_swap = (i, j)
                    best_r = new_r.copy()
                
                h[i] = 0
                h[j] = 1
        
        if best_swap is not None:
            i, j = best_swap
            h[i] = 1
            h[j] = 0
            r = best_r
            min_r = best_min_r
            zeros = np.where(h == 0)[0]
            ones = np.where(h == 1)[0]
            no_improve_count = 0
        else:
            no_improve_count += 1
            
    return h, min_r

def _continuous_optimize(h0, n_iter=500):
    """Simple gradient descent on smooth max."""
    h = np.array(h0, dtype=np.float64)
    h = np.clip(h, 0.0, 1.0)
    # Project to sum N/2
    target = N / 2.0
    current_sum = np.sum(h)
    offset = target - current_sum
    h = h + offset / N
    h = np.clip(h, 0.0, 1.0)
    
    best_h = h.copy()
    best_score = _score(h)
    
    lr = 0.01
    beta = 50.0
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 10:
            break
            
        cs = _linear_overlap_values(h)
        cs_max = np.max(cs)
        exp_cs = np.exp(beta * (cs - cs_max))
        sum_exp = np.sum(exp_cs)
        w = exp_cs / sum_exp
        
        # Gradient
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
        w_pad = np.zeros(n_fft)
        w_pad[:N] = w
        h_pad = np.zeros(n_fft)
        h_pad[:N] = h
        W = np.fft.rfft(w_pad)
        H = np.fft.rfft(h_pad)
        conv = np.fft.irfft(W * H, n=n_fft)
        A = conv[:N]
        w_rev = w[::-1]
        w_rev_pad = np.zeros(n_fft)
        w_rev_pad[:N] = w_rev
        W_rev = np.fft.rfft(w_rev_pad)
        conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
        B = conv_rev[N-1 : N-1+N]
        grad = -(2.0 / N) * (A + B)
        
        h_new = h - lr * grad
        h_new = np.clip(h_new, 0.0, 1.0)
        # Project
        diff = target - np.sum(h_new)
        h_new = h_new + diff / N
        h_new = np.clip(h_new, 0.0, 1.0)
        
        score = _score(h_new)
        if score < best_score:
            best_score = score
            best_h = h_new.copy()
            h = h_new
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
                
    return best_h, best_score

def construct():
    best_h = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    # 1. Continuous optimization from random starts
    for _ in range(5):
        if time.time() > _DEADLINE - 20:
            break
        h_init = np.random.uniform(0.0, 1.0, N)
        h_opt, s_opt = _continuous_optimize(h_init)
        if s_opt < best_score:
            best_score = s_opt
            best_h = h_opt.copy()
            
    # 2. Binarize the best continuous solution and local search
    if best_h is not None:
        h_bin = (best_h > 0.5).astype(float)
        # Adjust sum
        sum_bin = np.sum(h_bin)
        if sum_bin > N/2:
            # Flip largest 0.5+ to 0
            idx = np.argsort(best_h)[::-1]
            for i in idx:
                if h_bin[i] == 1 and sum_bin > N/2:
                    h_bin[i] = 0
                    sum_bin -= 1
        elif sum_bin < N/2:
            idx = np.argsort(best_h)
            for i in idx:
                if h_bin[i] == 0 and sum_bin < N/2:
                    h_bin[i] = 1
                    sum_bin += 1
        
        h_bin_opt, min_r = _local_search_binary(h_bin)
        s_bin = _score(h_bin_opt)
        if s_bin < best_score:
            best_score = s_bin
            best_h = h_bin_opt.copy()

    # 3. Random binary local search
    for _ in range(10):
        if time.time() > _DEADLINE - 10:
            break
        h_rand = np.zeros(N)
        indices = np.random.choice(N, N//2, replace=False)
        h_rand[indices] = 1.0
        h_rand_opt, min_r = _local_search_binary(h_rand)
        s_rand = _score(h_rand_opt)
        if s_rand < best_score:
            best_score = s_rand
            best_h = h_rand_opt.copy()

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    return best_h
