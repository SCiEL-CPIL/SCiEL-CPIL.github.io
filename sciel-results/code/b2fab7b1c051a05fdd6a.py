import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    def get_score_fixed(f):
        # Assumes sum(f) == 1
        c = np.convolve(f, f, 'full')
        return np.max(np.abs(c))
        
    def run_search(f_init, duration):
        f_init = f_init / np.sum(f_init)
        current_f = f_init.copy()
        current_score = get_score_fixed(current_f)
        best_f = current_f.copy()
        best_score = current_score
        
        T = 0.005
        step_size = 0.05
        
        while time.time() - start_time < deadline:
            if time.time() - start_time + duration > deadline:
                break # Not enough time for this restart? 
                # Actually, we should just check global deadline inside loop
            
            noise = np.random.normal(0, step_size, N)
            new_f = current_f + noise
            new_sum = np.sum(new_f)
            if abs(new_sum) < 1e-6:
                continue
            new_f = new_f / new_sum
            
            new_score = get_score_fixed(new_f)
            
            if new_score < current_score:
                current_f = new_f
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_f = new_f.copy()
            else:
                delta = new_score - current_score
                if np.random.rand() < np.exp(-delta / T):
                    current_f = new_f
                    current_score = new_score
            
            T *= 0.9995
            if T < 1e-5:
                T = 1e-5
                step_size *= 0.95
            if step_size < 1e-4:
                # Restart local search
                step_size = 0.05
                T = 0.005
                
        return best_f, best_score

    # Try a few initializations
    f_best = np.ones(N)
    score_best = float('inf')
    
    # Init 1: Constant
    f1, s1 = run_search(np.ones(N), 5)
    if s1 < score_best:
        score_best = s1
        f_best = f1
        
    # Init 2: Random
    f2, s2 = run_search(np.random.randn(N), 5)
    if s2 < score_best:
        score_best = s2
        f_best = f2
        
    # Init 3: Sine
    x = np.arange(N)
    f3, s3 = run_search(1 + np.cos(2 * np.pi * x / N), 5)
    if s3 < score_best:
        score_best = s3
        f_best = f3
        
    return f_best
