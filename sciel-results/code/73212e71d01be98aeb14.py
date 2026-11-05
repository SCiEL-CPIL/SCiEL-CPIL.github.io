"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from math import comb

def _popcount(x):
    """Fast population count for Python integers."""
    return bin(x).count('1')

def _full_verify(n, adj_bits):
    """
    Full verification for 6 and 7 subsets, sampled for 8+.
    """
    # Check 6-subsets
    for combo in combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        degs = [_popcount(adj_bits[v] & mask) for v in combo]
        if len(set(degs)) == 1:
            return False
            
    # Check 7-subsets
    for combo in combinations(range(n), 7):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        degs = [_popcount(adj_bits[v] & mask) for v in combo]
        if len(set(degs)) == 1:
            return False
            
    # Sample check for 8+
    for sz in range(8, n + 1):
        samples = min(100, comb(n, sz))
        for _ in range(samples):
            idx = random.sample(range(n), sz)
            mask = 0
            for v in idx:
                mask |= (1 << v)
            degs = [_popcount(adj_bits[v] & mask) for v in idx]
            if len(set(degs)) == 1:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = 5
    best_edges = []
    
    # We will try to find a valid graph for n=15, 16, 17, 18, 19, 20.
    # Start from lower n to ensure we have a baseline, then try to increase.
    # However, random search is more likely to succeed for smaller n.
    # Let's try to jump directly to higher n if possible, but fallback to lower.
    
    # Strategy:
    # 1. Try n=20 with limited restarts.
    # 2. If fail, try n=19, etc.
    # 3. For each n, use local search on random initializations.
    
    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline - 5:
            break
            
        found_valid = False
        
        # Number of restarts depends on n and remaining time
        # For n=20, C(20,6)=38760. Checking all is fast enough in C, but slow in Python loops.
        # We use bitmasks for speed.
        
        max_restarts = 20 if n <= 15 else 10
        
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize random graph
            adj_bits = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
            
            # Local search to minimize regular 6-subsets
            # We precompute masks for 6-subsets to speed up evaluation
            subsets_6 = []
            for combo in combinations(range(n), 6):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                subsets_6.append((combo, mask))
            
            def count_bad_6(adj):
                count = 0
                for combo, mask in subsets_6:
                    degs = [_popcount(adj[v] & mask) for v in combo]
                    if len(set(degs)) == 1:
                        count += 1
                return count
            
            current_bad_6 = count_bad_6(adj_bits)
            
            # If we start with 0 bad 6-subsets, check 7 and 8+
            if current_bad_6 == 0:
                if _full_verify(n, adj_bits):
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj_bits[u] & (1 << v):
                                edges.append([u, v])
                    best_n = n
                    best_edges = edges
                    found_valid = True
                    break
            
            # Local search loop
            stuck_counter = 0
            max_stuck = 50
            
            while time.time() < deadline - 5 and stuck_counter < max_stuck:
                # Pick random edge to flip
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                    
                # Flip
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
                
                new_bad_6 = count_bad_6(adj_bits)
                
                if new_bad_6 < current_bad_6:
                    current_bad_6 = new_bad_6
                    stuck_counter = 0
                    
                    # If we reach 0, check full validity
                    if current_bad_6 == 0:
                        if _full_verify(n, adj_bits):
                            edges = []
                            for u in range(n):
                                for v in range(u + 1, n):
                                    if adj_bits[u] & (1 << v):
                                        edges.append([u, v])
                            best_n = n
                            best_edges = edges
                            found_valid = True
                            break
                else:
                    # Revert
                    adj_bits[u] ^= (1 << v)
                    adj_bits[v] ^= (1 << u)
                    stuck_counter += 1
                    
            if found_valid:
                break
                
        if found_valid:
            # We found a valid graph for this n.
            # Should we try n+1? The loop iterates 20 down to 6.
            # If we found one for 18, we should have already tried 20 and 19.
            # So if we find one for 18, it means 20 and 19 failed.
            # We can break here because we are iterating downwards.
            break

    # Fallback if nothing found (should not happen for n=5)
    if best_n < 5:
        best_n = 5
        best_edges = []
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
