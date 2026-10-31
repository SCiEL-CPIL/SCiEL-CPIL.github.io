import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin
    
    # Initial guess: Uniform distribution gives ratio 1.0
    # We want to spread mass to reduce peak convolution relative to L2 norm.
    # Actually, concentrating mass INCREASES L2 norm but also increases peak conv.
    # Let's analyze:
    # h = delta: L2^2 = 1, Peak = 1. Ratio = 1.
    # h = uniform: L2^2 = 1/N, Peak = 1/N. Ratio = 1.
    # h = [1/2, 1/2, 0...]: L2^2 = 1/2, Peak = 1/2. Ratio = 1.
    # h = [1/3, 1/3, 0, 1/3]: L2^2 = 1/3, Peak = 2/9. Ratio = 1.5.
    # It seems "gaps" help.
    
    best_h = np.ones(N) / N
    best_score = 1.0
    
    def calc_score(h):
        # h is normalized to sum 1
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        peak = np.max(conv)
        if peak == 0:
            return 0
        return l2_sq / peak

    # Local search
    h = np.ones(N) / N
    step_size = 0.1
    num_iterations = 50000 # Adjust based on time
    
    for i in range(num_iterations):
        if time.time() > deadline:
            break
            
        # Perturb
        noise = np.random.normal(0, step_size, N)
        h_new = h + noise
        h_new = np.maximum(h_new, 0) # Enforce non-negative
        sum_h = np.sum(h_new)
        if sum_h == 0:
            h_new = np.ones(N) / N
        else:
            h_new = h_new / sum_h
            
        score_new = calc_score(h_new)
        
        if score_new > best_score:
            best_score = score_new
            best_h = h_new.copy()
            h = h_new.copy()
            # Optionally increase step size or keep
        else:
            # Annealing: reduce step size if stuck
            if i % 1000 == 0:
                step_size *= 0.95
            # Or accept with probability exp((new-old)/T)
            # For now, simple hill climbing with restarts
    
    # Final polish: try to optimize the "gaps"
    # The optimal structure likely has non-zero elements separated by zeros.
    # We can try to set small elements to 0 and re-normalize.
    for _ in range(100):
        if time.time() > deadline:
            break
        # Zero out smallest elements
        idx = np.argsort(best_h)
        # Try zeroing out 1, 2, 3 ... elements
        for k in range(1, 10):
            h_test = best_h.copy()
            h_test[idx[:k]] = 0
            sum_h = np.sum(h_test)
            if sum_h == 0: continue
            h_test = h_test / sum_h
            s = calc_score(h_test)
            if s > best_score:
                best_score = s
                best_h = h_test.copy()
                
    return list(best_h)
