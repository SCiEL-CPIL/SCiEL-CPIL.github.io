"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit
    p = 1009
    # Precompute modular inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    best_len = 0
    best_start = 1
    
    # Two-pointer sliding window: both s and e only move forward
    # For a valid interval [s, e], no pair (x,y) with x*y % p == 1
    # When we increment s, we can potentially extend e further
    e = 0
    for s in range(1, p):
        if time.time() > deadline:
            break
        # Optimization: if max possible length from s is <= best_len, break
        if p - s <= best_len:
            break
        
        # Ensure e >= s
        if e < s:
            e = s - 1
        
        # Extend e as far as possible
        while e + 1 < p:
            e_next = e + 1
            # Check diagonal: e_next is self-inverse
            if e_next == 1 or e_next == p - 1:
                break
            # Check off-diagonal: inv(e_next) in [s, e_next-1]
            if s <= inv[e_next] <= e_next - 1:
                break
            e = e_next
        
        current_len = e - s + 1
        if current_len > best_len:
            best_len = current_len
            best_start = s
    
    if best_len == 0:
        # Fallback: return a valid single-element interval
        # Avoid self-inverse elements: 1 and 1008
        best_start = 2
        best_len = 1
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
