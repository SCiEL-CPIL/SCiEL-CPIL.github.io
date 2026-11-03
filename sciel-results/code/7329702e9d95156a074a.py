import numpy as np
import random

def construct():
    N = 50
    DELTA = 0.5 / N
    
    def calc_score(h):
        l1 = np.sum(h)
        if l1 < 1e-15:
            return -np.inf
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return -np.inf
        return l2_sq / (l1 * max_conv)

    best_h = None
    best_score = -np.inf
    
    # Increase restarts and steps for better coverage
    num_restarts = 200
    steps_per_restart = 3000
    
    for _ in range(num_restarts):
        # Random initialization
        # Mix of uniform and sparse starts
        if random.random() < 0.5:
            h = np.random.rand(N)
        else:
            h = np.zeros(N)
            idx = random.sample(range(N), 5)
            for i in idx:
                h[i] = random.random()
        
        h_sum = np.sum(h)
        if h_sum == 0:
            h = np.ones(N) / N
        else:
            h /= h_sum
            
        current_score = calc_score(h)
        
        # Simulated Annealing
        T_start = 1e-4
        T_end = 1e-6
        decay = (T_end / T_start) ** (1.0 / steps_per_restart)
        T = T_start
        
        for step in range(steps_per_restart):
            # Pick two indices
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            
            # Try moving mass from i to j
            # Scale epsilon relative to current mass
            eps = np.random.uniform(0.05, 0.2) * h[i]
            
            if h[i] < eps:
                continue
            
            h_new = h.copy()
            h_new[i] -= eps
            h_new[j] += eps
            
            new_score = calc_score(h_new)
            
            if new_score > current_score:
                h = h_new
                current_score = new_score
            else:
                # Accept worse solution
                delta = new_score - current_score
                if delta > 0:
                    h = h_new
                    current_score = new_score
                else:
                    # Avoid overflow in exp
                    if T > 1e-8:
                        p = np.exp(delta / T)
                        if random.random() < p:
                            h = h_new
                            current_score = new_score
            
            T *= decay
        
        if current_score > best_score:
            best_score = current_score
            best_h = h

    return list(best_h)
