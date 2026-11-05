import numpy as np
import time

N = 400

def compute_value(h):
    S = np.sum(h)
    if np.isclose(S, 0):
        return float('inf')
    C = np.convolve(h, h)
    max_C = np.max(C)
    return abs(2 * N * max_C / (S ** 2))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Try some analytical candidates
    candidates = []
    
    # 1. Gaussian windows
    for sigma in [50, 100, 150, 200]:
        x = np.arange(N)
        h = np.exp(-(x - N/2)**2 / (2 * sigma**2))
        candidates.append(h)
        
    # 2. Sine + DC
    for P in [100, 200, 400, 800]:
        for amp in [0.1, 0.5, 1.0]:
            h = 1 + amp * np.sin(2 * np.pi * np.arange(N) / P)
            candidates.append(h)
            
    # 3. Random
    np.random.seed(42)
    for _ in range(10):
        h = np.random.randn(N)
        candidates.append(h)
        
    # Evaluate candidates
    for h in candidates:
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    # Local search on the best candidate
    # Simple random hill climbing
    step_size = 0.1
    for _ in range(5000):
        if time.time() > deadline:
            break
        h_new = best_h + np.random.randn(N) * step_size
        val_new = compute_value(h_new)
        if val_new < best_val:
            best_val = val_new
            best_h = h_new
            step_size *= 0.95 # decay step size
        else:
            step_size *= 0.99
            
    return best_h

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400

def compute_value(h):
    S = np.sum(h)
    if np.isclose(S, 0):
        return float('inf')
    C = np.convolve(h, h)
    max_C = np.max(C)
    return abs(2 * N * max_C / (S ** 2))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Try some analytical candidates
    candidates = []
    
    # 1. Gaussian windows
    for sigma in [50, 100, 150, 200]:
        x = np.arange(N)
        h = np.exp(-(x - N/2)**2 / (2 * sigma**2))
        candidates.append(h)
        
    # 2. Sine + DC
    for P in [100, 200, 400, 800]:
        for amp in [0.1, 0.5, 1.0]:
            h = 1 + amp * np.sin(2 * np.pi * np.arange(N) / P)
            candidates.append(h)
            
    # 3. Random
    np.random.seed(42)
    for _ in range(10):
        h = np.random.randn(N)
        candidates.append(h)
        
    # Evaluate candidates
    for h in candidates:
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    # Local search on the best candidate
    # Simple random hill climbing
    step_size = 0.1
    for _ in range(5000):
        if time.time() > deadline:
            break
        h_new = best_h + np.random.randn(N) * step_size
        val_new = compute_value(h_new)
        if val_new < best_val:
            best_val = val_new
            best_h = h_new
            step_size *= 0.95 # decay step size
        else:
            step_size *= 0.99
            
    return best_h
# EVOLVE-BLOCK-END
