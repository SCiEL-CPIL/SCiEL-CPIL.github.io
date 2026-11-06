import time
import sys
from math import gcd

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    
    N = 256
    vertices = list(range(2, N + 1))
    n = len(vertices)
    
    # Precompute compatibility
    # compat[i][j] is True if lcm(vertices[i], vertices[j]) > N
    compat = [0] * n
    for i in range(n):
        a = vertices[i]
        for j in range(i + 1, n):
            b = vertices[j]
            g = gcd(a, b)
            if a * b // g > N:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
                
    weights = [1.0 / v for v in vertices]
    
    # Initial Greedy: Prioritize 128, then other large evens, then odds
    # We want to maximize sum.
    # Let's try a specific heuristic:
    # 1. Include 128.
    # 2. Include all evens > 128.
    # 3. Add odds if compatible.
    
    S = 0
    # Index of 128
    idx_128 = 128 - 2 # 126
    S |= (1 << idx_128)
    
    # Add other evens
    for v in range(130, N + 1, 2):
        idx = v - 2
        if (S & ~compat[idx]) == 0:
            S |= (1 << idx)
            
    # Add odds
    for v in range(3, N + 1, 2):
        idx = v - 2
        if (S & ~compat[idx]) == 0:
            S |= (1 << idx)
            
    def get_value(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            s += weights[idx]
            m ^= lsb
        return s
        
    best_mask = S
    best_val = get_value(S)
    
    # Local Search: Remove and Refill
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = best_mask
        while m:
            lsb = m & -m
            u_idx = (lsb.bit_length() - 1)
            m ^= lsb
            
            S_candidate = best_mask & ~lsb
            val_candidate = best_val - weights[u_idx]
            
            new_mask = S_candidate
            new_val = val_candidate
            
            # Greedy Refill: Add vertices in order of weight (descending)
            # Weights are 1/v, so small v first.
            # But we should maybe prioritize vertices that were just removed?
            # Or just iterate all.
            for v in range(n):
                if new_mask & (1 << v):
                    continue
                if (new_mask & ~compat[v]) == 0:
                    new_mask |= (1 << v)
                    new_val += weights[v]
                    
            if new_val > best_val:
                best_val = new_val
                best_mask = new_mask
                improved = True
                break
        
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = (lsb.bit_length() - 1)
        result.append(vertices[idx])
        m ^= lsb
        
    if not result:
        return [2]
        
    return result
