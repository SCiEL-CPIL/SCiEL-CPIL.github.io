"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_range(n):
    return reduce(_lcm, range(1, n + 1))

LCM_PERIOD = _lcm_range(12)  # 27720

def _max_matching(adj, n_left):
    """Compute max bipartite matching size. adj[i] = list of right nodes for left node i (0-indexed)."""
    match_r = [-1] * len(adj[0]) if adj else []
    # Actually, let's use a standard approach with arrays
    # adj is a list of lists: adj[i] contains right-side indices
    
    # Use Hopcroft-Karp or simple DFS-based matching
    # For small sizes, DFS-based is fine
    
    match_r = {}
    def try_augment(u, visited):
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                if v not in match_r or try_augment(match_r[v], visited):
                    match_r[v] = u
                    return True
        return False
    
    result = 0
    for u in range(n_left):
        visited = set()
        if try_augment(u, visited):
            result += 1
    return result

def _compute_f(m, n=12, max_L=200):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_1..b_n with i|b_i."""
    for L in range(1, max_L + 1):
        # Build adjacency: for each i (1..n), which numbers in (m, m+L] are divisible by i
        # Numbers are m+1, m+2, ..., m+L; index j=0..L-1 corresponds to m+1+j
        adj = []
        for i in range(1, n + 1):
            candidates = []
            for j in range(L):
                if (m + 1 + j) % i == 0:
                    candidates.append(j)
            adj.append(candidates)
        
        # Check if perfect matching exists (size n)
        if not adj:
            continue
        # Need at least n candidates total, but more importantly, matching must cover all n left nodes
        # Quick check: each i must have at least one candidate
        if any(len(adj[i]) == 0 for i in range(n)):
            continue
        
        # Use bipartite matching
        # Left side: 0..n-1 (representing i=1..n)
        # Right side: 0..L-1 (representing positions)
        # We need matching of size n
        
        match_r = {}
        def try_augment(u, visited):
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    if v not in match_r or try_augment(match_r[v], visited):
                        match_r[v] = u
                        return True
            return False
        
        match_size = 0
        for u in range(n):
            visited = set()
            if try_augment(u, visited):
                match_size += 1
        
        if match_size == n:
            return L
    return max_L  # Should not reach here for reasonable inputs

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    # Precompute f(12, r) for all residues r in [0, LCM_PERIOD)
    best_f = 0
    best_residue = 0
    
    for r in range(LCM_PERIOD):
        if time.time() > deadline:
            break
        f_val = _compute_f(r, n=12, max_L=144)
        if f_val > best_f:
            best_f = f_val
            best_residue = r
    
    # Find m in [0, 1000000] with m % LCM_PERIOD == best_residue
    # m = best_residue, or best_residue + k * LCM_PERIOD
    if best_residue <= 1000000:
        m = best_residue
    else:
        # Find smallest m >= 0 with m % LCM_PERIOD == best_residue
        # m = best_residue (since best_residue < LCM_PERIOD, it's already in [0, LCM_PERIOD))
        m = best_residue
    
    # Ensure m is a Python int
    m = int(m)
    
    # Verify m is in range
    if m < 0 or m > 1000000:
        m = 0
    
    return m

# EVOLVE-BLOCK-END
