"""
Bootstrap portfolio: MODEL-AND-OPTIMIZE for large graphs with no regular induced subgraph on 6+ vertices.

Strategy:
1. Start from a known 20-vertex construction (or a random heuristic).
2. Verify validity quickly using bitmasks.
3. Attempt local search / greedy edge flips to improve or maintain validity.
4. Since we know 20-vertex graphs exist, aim to find one efficiently.
5. If time permits, try to push to 21 (harder, but possible if we get lucky or use smarter search).

Key insight: A regular induced subgraph on k vertices means every vertex in that subset has the same degree within the subset. For k=6, we need no 6-subset where all 6 vertices have equal induced degree.

We'll use a bitmask-based checker for speed.
"""
import numpy as np

import time
import random
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def has_regular_6plus_numpy(adj_matrix, n):
        """Check if any subset of size >= 6 induces a regular subgraph using NumPy."""
        # adj_matrix is an n x n numpy array (0/1), symmetric
        # For each subset mask, check if induced subgraph is regular
        
        # Precompute bit counts for all masks up to 2^n
        # For n=20, 2^20 = 1M masks
        # We'll iterate and use vectorized operations where possible
        
        # Strategy: For each mask with popcount >= 6, extract the induced subgraph
        # and check if all degrees are equal
        
        # To speed up, we can precompute for each vertex v and each mask,
        # the degree of v within the mask: deg[v][mask] = popcount(adj[v] & mask)
        # But that's n * 2^n which is too much memory for n=20.
        
        # Instead, iterate over masks and compute degrees on the fly using bit operations.
        
        # For n <= 22, 2^n is at most 4M which might be too slow in pure Python.
        # Let's use a different approach: check only subsets of size exactly 6 first,
        # then 7, 8, etc. But the problem requires checking ALL sizes >= 6.
        
        # Key observation: If a subset S of size k >= 6 is regular with degree d,
        # then we need to detect it. But checking all 2^n subsets is expensive.
        
        # Alternative: Use the fact that for a regular induced subgraph,
        # the number of edges in the subgraph must be k*d/2 where d is the degree.
        
        # Let's try a hybrid approach:
        # 1. Check all 6-subsets (C(n,6) subsets)
        # 2. Check all 7-subsets
        # 3. Continue until we find a violation or exhaust subsets
        
        # For n=20: C(20,6)=38760, C(20,7)=77520, C(20,8)=125970, ...
        # Total for sizes 6-20: sum_{k=6}^{20} C(20,k) = 2^20 - sum_{k=0}^{5} C(20,k)
        # sum_{k=0}^{5} C(20,k) = 1 + 20 + 190 + 1140 + 4845 + 15504 = 21700
        # So total = 1048576 - 21700 = 1026876 subsets to check.
        
        # This is feasible with optimized code.
        
        # Let's use a different strategy: iterate over all masks, but use
        # efficient bit manipulation.
        
        # Precompute popcount for all possible byte values
        popcount = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)
        
        def popcount_fast(x):
            """Fast popcount using table lookup."""
            if x == 0:
                return 0
            result = 0
            while x:
                result += popcount[x & 0xFF]
                x >>= 8
            return result
        
        # Convert adj_matrix to bitmask representation
        adj_masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj_matrix[i][j]:
                    mask |= (1 << j)
            adj_masks[i] = mask
        
        full_mask = (1 << n) - 1
        
        # Iterate over all masks with popcount >= 6
        for mask in range(full_mask + 1):
            pc = popcount_fast(mask)
            if pc < 6:
                continue
            
            # Check if induced subgraph on 'mask' is regular
            # For each vertex in the mask, compute its degree within the mask
            degs = []
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                # Degree of v within mask
                d = popcount_fast(adj_masks[v] & mask)
                degs.append(d)
                m ^= lsb
            
            # Check if all degrees are equal
            if len(set(degs)) == 1:
                return True
        
        return False
    
    def verify_edges(edges, n):
        """Verify that no subset of size >= 6 induces a regular subgraph."""
        if n < 6:
            return True
        
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=np.int8)
        for e in edges:
            u, v = e
            adj[u][v] = 1
            adj[v][u] = 1
        
        return not has_regular_6plus_numpy(adj, n)
    
    best_result = {'n': 5, 'edges': []}
    
    # Try n from 20 down to 6
    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline:
            break
        
        # Try multiple random graphs for this n
        max_attempts = 2000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random graph with varying probability
            p = 0.2 + 0.6 * random.random()
            
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        edges.append([i, j])
            
            if verify_edges(edges, n):
                best_result = {'n': n, 'edges': edges}
                break
        
        if best_result['n'] >= n:
            break
    
    return best_result
