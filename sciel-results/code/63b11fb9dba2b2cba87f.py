import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    P = 27720  # LCM(1..12)
    
    best_L = 0
    best_r = 0
    
    # Precompute modular inverses or just use %
    
    def has_matching(r, L):
        # Left nodes: i in 1..12
        # Right nodes: j in 0..L-1
        # Edge i-j if (r + 1 + j) % i == 0
        
        # For each i, find valid j's
        # To speed up, order i by decreasing constraint (higher i first)
        # i=12 is most constrained
        
        assigned = [False] * L
        
        def dfs(i):
            if i == 0:
                return True
            # Find j in 0..L-1 such that not assigned[j] and (r+1+j) % i == 0
            # Optimization: 
            # j % i == (-r-1) % i
            rem = (-r - 1) % i
            # j can be rem, rem+i, rem+2i, ...
            for j in range(rem, L, i):
                if not assigned[j]:
                    assigned[j] = True
                    if dfs(i-1):
                        return True
                    assigned[j] = False
            return False
        
        return dfs(12)

    # Iterate through all residues
    for r in range(P):
        if time.time() > deadline:
            break
            
        # Binary search or linear search for L?
        # L is small. Linear is fine.
        # Min L is 12.
        # Max L is likely < 200.
        
        # We can check L from 12 upwards.
        # But we want to find the MAX f(r).
        # So for each r, we find the MIN L that works.
        # That min L is f(r).
        # We want to maximize f(r).
        
        # Is it possible that a larger L is "better" for some other reason?
        # No, f(r) is defined as the LEAST positive L.
        # So for a fixed r, f(r) is a single value.
        # We want to find r that maximizes this value.
        
        # So for each r, find min L.
        
        L = 12
        while L < 300: # Upper bound
            if has_matching(r, L):
                if L > best_L:
                    best_L = L
                    best_r = r
                break
            L += 1
            
    return best_r
