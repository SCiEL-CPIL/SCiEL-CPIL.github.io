import numpy as np
import time
import random

N = 50

def find_sidon_set(n, max_size=10, num_trials=1000):
    """
    Finds a Sidon subset of {0, ..., n-1} of size max_size.
    A Sidon set is one where all pairwise sums are unique.
    """
    for size in range(max_size, 2, -1):
        for _ in range(num_trials):
            S = set(random.sample(range(n), size))
            S_list = list(S)
            sums = set()
            is_sidon = True
            for i in range(size):
                for j in range(i, size):
                    s = S_list[i] + S_list[j]
                    if s in sums:
                        is_sidon = False
                        break
                    sums.add(s)
                if not is_sidon:
                    break
            if is_sidon:
                return S_list
    # Fallback to a small known set or greedy
    # Greedy construction
    S = [0]
    for x in range(1, n):
        # Check if x can be added
        ok = True
        for y in S:
            if x + y in [a + b for a in S for b in S if a <= b]:
                 # This check is inefficient, need to store sums
                 pass
        # Simpler fallback: just return first 'size' elements of a known sequence or uniform
        # For robustness, if random fails, use a heuristic
        # Let's just try a few more random trials with smaller size
    for size in range(max_size, 1, -1):
        for _ in range(5000):
            S = set(random.sample(range(n), size))
            S_list = list(S)
            sums = set()
            is_sidon = True
            for i in range(size):
                for j in range(i, size):
                    s = S_list[i] + S_list[j]
                    if s in sums:
                        is_sidon = False
                        break
                    sums.add(s)
                if not is_sidon:
                    break
            if is_sidon:
                return S_list
    # Ultimate fallback: uniform
    return list(range(n))

def optimize_weights(indices, time_limit=100):
    """
    Optimizes weights on the given indices to maximize the ratio.
    """
    n = N
    h = np.zeros(n)
    M = len(indices)
    if M == 0:
        return h
    
    # Initialize uniform
    for idx in indices:
        h[idx] = 1.0 / M
        
    start_time = time.time()
    step_size = 0.01
    best_score = 0.0
    best_h = h.copy()
    
    def score(h):
        h = h / np.sum(h) # Ensure normalization
        g = np.convolve(h, h, mode='full')
        return np.sum(h**2) / np.max(g)
    
    best_score = score(h)
    
    while time.time() - start_time < time_limit:
        # Compute gradient
        g = np.convolve(h, h, mode='full')
        max_val = np.max(g)
        max_indices = np.where(g == max_val)[0]
        
        # Subgradient for one of the max indices
        m_star = max_indices[0]
        
        grad = np.zeros(n)
        for k in range(n):
            # Sum of h_j such that k + j == m_star
            j = m_star - k
            if 0 <= j < n:
                # Derivative of g_m_star w.r.t h_k
                # g_m = sum_{i+j=m} h_i h_j
                # d/dh_k (h_i h_j) is h_j if i=k, h_i if j=k
                # If i=k, j=m_star-k. Term is h_k h_j. Deriv h_j.
                # If j=k, i=m_star-k. Term is h_i h_k. Deriv h_i.
                # If i=j=k, term is h_k^2. Deriv 2h_k.
                
                if k == j:
                    grad[k] += 2 * h[k]
                else:
                    grad[k] += h[j]
                    # Wait, the sum in convolution is over all i.
                    # g_m = sum_i h_i h_{m-i}.
                    # d/dh_k g_m = h_{m-k} + h_{m-k} = 2 h_{m-k} if m-k != k?
                    # Let's check:
                    # If k != m-k, the term h_k h_{m-k} appears once as i=k and once as i=m-k?
                    # No, the sum is over i.
                    # Term i=k: h_k h_{m-k}. Deriv w.r.t h_k is h_{m-k}.
                    # Term i=m-k: h_{m-k} h_k. Deriv w.r.t h_k is h_{m-k}.
                    # So if k != m-k, total deriv is 2 h_{m-k}.
                    # If k == m-k, term is h_k^2. Deriv is 2 h_k.
                    # So in both cases, it's 2 h_{m-k}.
                    pass
        
        # Correct gradient calculation:
        grad = np.zeros(n)
        for k in range(n):
            j = m_star - k
            if 0 <= j < n:
                grad[k] = 2 * h[j]
        
        # Objective J = S2 / G
        # dJ/dh = (2h G - S2 dG/dh) / G^2
        S2 = np.sum(h**2)
        grad_J = (2 * h * max_val - S2 * grad) / (max_val ** 2)
        
        # Update
        h_new = h + step_size * grad_J
        
        # Project to simplex
        h_new = np.maximum(h_new, 0)
        sum_h = np.sum(h_new)
        if sum_h > 0:
            h_new = h_new / sum_h
        else:
            h_new = h.copy()
        
        h = h_new
        curr_score = score(h)
        
        if curr_score > best_score:
            best_score = curr_score
            best_h = h.copy()
            
        # Adaptive step size
        if curr_score < best_score * 0.99:
            step_size *= 0.5
        else:
            step_size *= 1.05
    
    return best_h

def construct():
    start_time = time.time()
    deadline = 110 # Leave 10s margin
    
    # Find a good Sidon set
    # Try sizes 8, 7, 6...
    indices = None
    for size in range(8, 3, -1):
        if time.time() - start_time > deadline - 5:
            break
        indices = find_sidon_set(N, max_size=size, num_trials=2000)
        if len(indices) == size:
            break
            
    if indices is None or len(indices) < 2:
        # Fallback to uniform
        return [1.0] * N
        
    # Optimize weights
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time < 1:
        # Not enough time, just return uniform on Sidon set
        h = np.zeros(N)
        for idx in indices:
            h[idx] = 1.0 / len(indices)
        return h.tolist()
        
    h = optimize_weights(indices, time_limit=remaining_time)
    
    # Ensure non-negative and normalized (though optimize does this)
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s > 0:
        h = h / s
    else:
        h = np.ones(N) / N
        
    return h.tolist()
