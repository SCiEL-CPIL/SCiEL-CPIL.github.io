"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _has_triangle_circulant_fast(n, S):
    """Check if circulant graph on Z_n with connection set S has a triangle.
    Triangle exists if there exist x,y in S such that x+y mod n is in S or -S.
    """
    S_set = frozenset(S)
    neg_S = frozenset({(-x) % n for x in S})
    for x in S:
        for y in S:
            z = (x + y) % n
            if z in S_set or z in neg_S:
                return True
    return False

def _max_independent_set_size(n, adj_mask):
    """
    Compute alpha of a graph given its adjacency bitmasks.
    Uses branch and bound with bitmasks.
    adj_mask: list of bitmasks where adj_mask[i] has bit j set iff i~j.
    """
    full_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    best = [0]
    
    def popcount(x):
        return bin(x).count('1')
    
    def dfs(candidates, current_size):
        # Upper bound
        if current_size + popcount(candidates) <= best[0]:
            return
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pick vertex with highest degree in candidates for better pruning
        best_v = -1
        max_deg = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = popcount(candidates & adj_mask[v])
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
        
        v = best_v
        # Include v
        dfs(candidates & comp_mask[v], current_size + 1)
        # Exclude v
        dfs(candidates & ~(1 << v), current_size)
    
    dfs(full_mask, 0)
    return best[0]

def _verify_circulant(n, S):
    """Verify circulant graph is triangle-free and alpha < 10."""
    if _has_triangle_circulant_fast(n, S):
        return False, 0
    
    # Build adjacency bitmasks
    adj_mask = [0] * n
    S_set = frozenset(S)
    for i in range(n):
        for s in S_set:
            j = (i + s) % n
            if j != i:
                adj_mask[i] |= (1 << j)
    
    # Run exact B&B
    alpha = _max_independent_set_size(n, adj_mask)
    if alpha >= 10:
        return False, alpha
    return True, alpha

def _build_edges(n, S):
    """Build edge list from circulant connection set."""
    edges = []
    S_set = set(S)
    for i in range(n):
        for s in S_set:
            j = (i + s) % n
            if j > i:
                edges.append([i, j])
    return edges

def _greedy_circulant(n, target_k):
    """
    Greedily build a symmetric connection set S for Z_n that avoids triangles.
    Returns (S, success) where S is a frozenset of shifts.
    """
    # Start with empty set
    S = frozenset()
    
    # Candidate shifts: 1..n//2
    candidates = list(range(1, n // 2 + 1))
    random.shuffle(candidates)
    
    for c in candidates:
        if len(S) >= target_k:
            break
        # Check if adding c (and -c) maintains triangle-free
        new_S = S | {c, (-c) % n}
        new_S = frozenset(new_S)
        if not _has_triangle_circulant_fast(n, new_S):
            S = new_S
    
    return S, len(S) >= target_k

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_n = 0
    best_edges = []
    
    # Try n from 40 down to 25
    for n in range(40, 24, -1):
        if time.time() > deadline - 15:
            break
            
        found = False
        half = n // 2
        available = list(range(1, half + 1))
        
        # Strategy 1: Greedy construction with random restarts
        for k in range(min(half, 12), 2, -1):
            if time.time() > deadline - 10:
                break
            
            # Try multiple greedy restarts for this k
            num_restarts = 20 if n <= 35 else 10
            
            for restart in range(num_restarts):
                if time.time() > deadline:
                    break
                
                S, success = _greedy_circulant(n, k)
                if not success:
                    continue
                
                # Verify alpha
                valid, alpha = _verify_circulant(n, S)
                if valid:
                    best_n = n
                    best_edges = _build_edges(n, S)
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
        
        # Strategy 2: Random sampling if greedy failed
        if time.time() > deadline - 10:
            continue
            
        for k in range(min(half, 12), 2, -1):
            if time.time() > deadline - 10:
                break
            
            num_attempts = 50 if n <= 35 else 25
            
            for attempt in range(num_attempts):
                if time.time() > deadline:
                    break
                
                if k > len(available):
                    continue
                    
                S_half = set(random.sample(available, k))
                S = set()
                for x in S_half:
                    S.add(x)
                    S.add((-x) % n)
                S = frozenset(S)
                
                if _has_triangle_circulant_fast(n, S):
                    continue
                
                valid, alpha = _verify_circulant(n, S)
                if valid:
                    best_n = n
                    best_edges = _build_edges(n, S)
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    # Fallback: Known small graphs
    if best_n < 19:
        n0 = 26
        best_edges = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
        best_n = n0
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
