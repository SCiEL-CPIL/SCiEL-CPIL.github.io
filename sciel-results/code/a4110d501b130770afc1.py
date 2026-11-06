import numpy as np
import time

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for the 90s limit
    
    # 1. Precompute APs and Incidence Matrix A
    ap_indices = []
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            idx = a - 1
            if idx >= N:
                continue
            current_ap = []
            while idx < N:
                current_ap.append(idx)
                idx += d
            if current_ap:
                ap_indices.append(current_ap)
                
    num_aps = len(ap_indices)
    A = np.zeros((num_aps, N), dtype=np.float32)
    for i, ap in enumerate(ap_indices):
        A[i, ap] = 1.0
        
    # 2. Continuous Gradient Descent Initialization
    G = A.T @ A
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, N).astype(np.float32)
    
    lr = 0.01
    # Run GD for a short time
    for _ in range(100):
        if time.time() > deadline - 20:
            break
        grad = G @ x
        x -= lr * grad
        x = np.clip(x, -1, 1)
        
    x_discrete = np.where(x > 0, 1, -1).astype(np.float32)
    
    # 3. Simulated Annealing
    current_max = np.max(np.abs(A @ x_discrete))
    best_max = current_max
    best_x = x_discrete.copy()
    
    if best_max == 1:
        return [int(v) for v in best_x]
        
    T = 5.0
    cooling_rate = 0.99995
    min_T = 0.05
    
    # We can speed up by checking time more frequently if SA is slow
    # But A@x is fast.
    
    while time.time() < deadline - 2:
        if T < min_T:
            T = min_T
            
        i = rng.integers(N)
        
        x_discrete[i] *= -1
        
        sums = A @ x_discrete
        new_max = np.max(np.abs(sums))
        
        delta = new_max - current_max
        
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_x = x_discrete.copy()
                if best_max == 1:
                    break
        else:
            x_discrete[i] *= -1
            
        T *= cooling_rate
        
    return [int(v) for v in best_x]
