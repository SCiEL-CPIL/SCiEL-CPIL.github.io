"""Optimized Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time

N = 100

def _compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _compute_all_overlaps(h):
    """Compute all N overlap values."""
    one_minus_h = 1.0 - h
    ovs = np.empty(N)
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        ovs[k] = np.dot(h, shifted)
    return ovs

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Start with a binary sequence: alternating blocks
    # Try to find a good binary pattern first
    best_h = None
    best_score = float('inf')
    
    # Initialize with a structured binary pattern
    # Use a pattern that balances 1s and 0s
    h = np.zeros(N)
    # Place N/2 ones in a spread-out pattern
    # Try several initial patterns
    patterns = []
    
    # Pattern 1: evenly spaced
    idx1 = np.arange(0, N, 2)
    p1 = np.zeros(N)
    p1[idx1] = 1.0
    patterns.append(p1)
    
    # Pattern 2: golden ratio based spacing
    phi = (1 + 5**0.5) / 2
    idx2 = np.array([int((i * N) / phi) % N for i in range(N//2)])
    p2 = np.zeros(N)
    p2[idx2] = 1.0
    patterns.append(p2)
    
    # Pattern 3: random
    for _ in range(5):
        idx = np.random.choice(N, N//2, replace=False)
        p = np.zeros(N)
        p[idx] = 1.0
        patterns.append(p)
    
    # Evaluate initial patterns
    for p in patterns:
        if time.time() > deadline:
            break
        score = _compute_max_overlap(p)
        if score < best_score:
            best_score = score
            best_h = p.copy()
    
    if best_h is None:
        best_h = patterns[0].copy()
        best_score = _compute_max_overlap(best_h)
    
    # Local search: flip individual bits
    h = best_h.copy()
    score = best_score
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Try random flips
        i = np.random.randint(N)
        old_val = h[i]
        h[i] = 1.0 - old_val
        
        new_score = _compute_max_overlap(h)
        
        if new_score < score - 1e-12:
            score = new_score
            no_improve = 0
        else:
            h[i] = old_val  # revert
            no_improve += 1
    
    # Now try to refine with continuous values
    # Use gradient-free optimization on the continuous version
    h_cont = h.copy()
    
    # Try small perturbations around the binary solution
    best_cont_score = score
    best_cont_h = h.copy()
    
    # Simulated annealing style refinement
    T = 0.01
    for iteration in range(10000):
        if time.time() > deadline:
            break
        if T < 0.0001:
            break
        
        # Pick a random position and adjust its value
        i = np.random.randint(N)
        delta = np.random.uniform(-0.1, 0.1)
        old_val = h_cont[i]
        h_cont[i] = np.clip(old_val + delta, 0.0, 1.0)
        
        # Check sum constraint
        current_sum = np.sum(h_cont)
        target_sum = N / 2.0
        
        new_score = _compute_max_overlap(h_cont)
        
        # Penalize if sum deviates too much
        sum_penalty = 10.0 * abs(current_sum - target_sum)
        total_score = new_score + sum_penalty
        
        if total_score < best_cont_score + 1e-12 or np.random.random() < np.exp(-(total_score - best_cont_score) / T):
            if total_score < best_cont_score - 1e-12:
                best_cont_score = total_score
                best_cont_h = h_cont.copy()
        
        T *= 0.999
    
    # Use the best continuous solution if it's better
    if best_cont_score < score:
        h_final = best_cont_h
    else:
        h_final = h
    
    # Final: project to ensure exact sum constraint
    # Adjust to make sum exactly N/2
    current_sum = np.sum(h_final)
    if abs(current_sum - N/2) > 1e-10:
        # Distribute the difference evenly
        diff = N/2 - current_sum
        h_final += diff / N
        # Clip and re-adjust
        h_final = np.clip(h_final, 0.0, 1.0)
        # Fine-tune to get exact sum
        for _ in range(100):
            current_sum = np.sum(h_final)
            if abs(current_sum - N/2) < 1e-12:
                break
            diff = (N/2 - current_sum) / N
            h_final += diff
            h_final = np.clip(h_final, 0.0, 1.0)
    
    # Verify
    final_score = _compute_max_overlap(h_final)
    
    return h_final

# Pre-compute to make construct fast
# Actually, let's do the optimization at module load time
_start = time.time()

def _optimize():
    start_time = time.time()
    deadline = start_time + 100.0
    
    best_h = None
    best_score = float('inf')
    
    # Try multiple binary patterns
    np.random.seed(42)
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Generate a random binary pattern with N/2 ones
        idx = np.random.choice(N, N//2, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        
        score = _compute_max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    if best_h is None:
        idx = np.arange(0, N, 2)
        best_h = np.zeros(N)
        best_h[idx] = 1.0
        best_score = _compute_max_overlap(best_h)
    
    # Local search on binary
    h = best_h.copy()
    score = best_score
    no_improve = 0
    
    while time.time() < deadline and no_improve < 3000:
        i = np.random.randint(N)
        old_val = h[i]
        h[i] = 1.0 - old_val
        
        new_score = _compute_max_overlap(h)
        
        if new_score < score - 1e-12:
            score = new_score
            no_improve = 0
        else:
            h[i] = old_val
            no_improve += 1
    
    # Continuous refinement
    h_cont = h.copy()
    best_cont_score = score
    best_cont_h = h.copy()
    
    T = 0.05
    for iteration in range(20000):
        if time.time() > deadline:
            break
        if T < 0.0001:
            break
        
        i = np.random.randint(N)
        delta = np.random.uniform(-0.05, 0.05)
        old_val = h_cont[i]
        h_cont[i] = np.clip(old_val + delta, 0.0, 1.0)
        
        current_sum = np.sum(h_cont)
        new_score = _compute_max_overlap(h_cont)
        sum_penalty = 5.0 * abs(current_sum - N/2)
        total_score = new_score + sum_penalty
        
        if total_score < best_cont_score - 1e-12:
            best_cont_score = total_score
            best_cont_h = h_cont.copy()
        elif np.random.random() < np.exp(-(total_score - best_cont_score) / T):
            pass  # accept but don't update best
        
        T *= 0.9995
    
    if best_cont_score < score:
        h_final = best_cont_h
    else:
        h_final = h
    
    # Project to exact sum
    current_sum = np.sum(h_final)
    if abs(current_sum - N/2) > 1e-10:
        diff = N/2 - current_sum
        h_final += diff / N
        h_final = np.clip(h_final, 0.0, 1.0)
        for _ in range(200):
            current_sum = np.sum(h_final)
            if abs(current_sum - N/2) < 1e-12:
                break
            diff = (N/2 - current_sum) / N
            h_final += diff
            h_final = np.clip(h_final, 0.0, 1.0)
    
    return h_final

_OPTIMIZED_H = _optimize()

def construct():
    return _OPTIMIZED_H.copy()
