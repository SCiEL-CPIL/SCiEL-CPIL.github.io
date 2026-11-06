import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85
    n = 128
    targets = np.arange(128, 257)
    
    def compute_metrics(A):
        conv = np.convolve(A, A, 'full')
        diag = np.zeros(len(conv), dtype=np.int32)
        even_targets = targets[targets % 2 == 0]
        for t in even_targets:
            half = t // 2
            if 0 <= half <= n:
                diag[t] = A[half]
        unordered = (conv[targets] + diag[targets]) // 2
        return int(np.max(unordered)), int(np.min(unordered))
    
    # Initialize with seed
    A = np.zeros(n+1, dtype=np.int8)
    A[64:129] = 1
    best_peak, _ = compute_metrics(A)
    best_A = A.copy()
    current_peak = best_peak
    current_A = A.copy()
    
    rng = random.Random(42)
    temp = 3.0
    
    while time.time() < deadline:
        move_type = rng.random()
        if move_type < 0.5:
            idx = rng.randint(0, n)
            A = current_A.copy()
            A[idx] = 1 - A[idx]
        elif move_type < 0.85:
            idx1, idx2 = rng.sample(range(n+1), 2)
            A = current_A.copy()
            A[idx1] = 1 - A[idx1]
            A[idx2] = 1 - A[idx2]
        else:
            # Random restart-ish: flip 3-5 bits
            k = rng.randint(3, 5)
            idxs = rng.sample(range(n+1), k)
            A = current_A.copy()
            for idx in idxs:
                A[idx] = 1 - A[idx]
        
        new_peak, min_count = compute_metrics(A)
        
        if min_count >= 1:
            if new_peak < current_peak:
                current_peak = new_peak
                current_A = A
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_A = A.copy()
            elif new_peak == current_peak:
                if rng.random() < 0.4:
                    current_A = A
            else:
                delta = new_peak - current_peak
                if rng.random() < np.exp(-delta / max(temp, 0.1)):
                    current_peak = new_peak
                    current_A = A
        # else reject
        
        temp = max(0.1, temp * 0.99995)
    
    return [i for i in range(n+1) if best_A[i]]
