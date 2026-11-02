import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def compute_ratio(h):
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        l_inf = np.max(h)
        if l1 < 1e-15 or l_inf < 1e-15:
            return 0.0
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return 0.0
        return (l2_sq) / (l1 * l_inf * max_conv)

    # Initialize with a known Sidon set
    # A Sidon set in [0, 49] with 8 elements
    # 0, 1, 4, 10, 18, 23, 30, 38 is a candidate
    best_h = np.zeros(N)
    best_h[[0, 1, 4, 10, 18, 23, 30, 38]] = 1.0
    best_ratio = compute_ratio(best_h)
    
    # Local search to improve the Sidon set
    import random
    random.seed(42)
    
    current_h = best_h.copy()
    current_ratio = best_ratio
    
    T = 0.1
    for i in range(5000):
        if time.time() > deadline:
            break
            
        # Random move
        move_type = random.random()
        new_h = current_h.copy()
        
        if move_type < 0.3:
            # Add
            zeros = np.where(current_h == 0)[0]
            if len(zeros) > 0:
                idx = random.choice(zeros)
                new_h[idx] = 1.0
        elif move_type < 0.6:
            # Remove
            ones = np.where(current_h == 1)[0]
            if len(ones) > 1:
                idx = random.choice(ones)
                new_h[idx] = 0.0
        else:
            # Swap
            ones = np.where(current_h == 1)[0]
            zeros = np.where(current_h == 0)[0]
            if len(ones) > 0 and len(zeros) > 0:
                idx_out = random.choice(ones)
                idx_in = random.choice(zeros)
                new_h[idx_out] = 0.0
                new_h[idx_in] = 1.0
        
        new_ratio = compute_ratio(new_h)
        
        if new_ratio > current_ratio:
            current_h = new_h
            current_ratio = new_ratio
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_h = current_h.copy()
        else:
            delta = current_ratio - new_ratio
            if random.random() < np.exp(-delta / T):
                current_h = new_h
                current_ratio = new_ratio
        
        T = max(T * 0.99, 0.01)

    return list(best_h)
