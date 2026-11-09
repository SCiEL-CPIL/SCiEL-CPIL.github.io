<<<<<<< SEARCH
"""Deterministic algebraic construction + rapid vectorized verification for
graphs with no induced regular subgraph on >= 6 vertices.

Strategy:
  1. Generate candidate graphs via:
     - Paley-type graphs over small finite fields
     - Quadratic-form graphs over GF(q)
     - Random graphs with controlled density
     - Greedy constructive search with fast pruning
  2. For each candidate, verify NO subset of size >= 6 induces a regular graph.
     Verification uses NumPy bitmask operations for speed.
  3. Return the largest valid graph found within the time budget.
"""
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - start_time)

    # --- Verification: check if a graph has any induced regular subgraph of size >= 6 ---
    def verify_no_regular_subgraph(adj, n):
        """
        adj: n x n boolean numpy array (symmetric, zero diagonal)
        n: number of vertices
        Returns True if no subset of size >= 6 induces a regular graph.
        """
        # Precompute degrees within any subset using bitmasks
        # For each subset of size k >= 6, check if all induced degrees are equal
        
        # Efficient approach: iterate over all subsets of size 6 to n
        # For n up to ~20, C(20,6)=38760, C(20,7)=77520, ..., C(20,20)=1
        # Total subsets of size >= 6: sum_{k=6}^{20} C(20,k) = 2^20 - sum_{k=0}^{5} C(20,k)
        # = 1048576 - (1+20+190+1140+4845+15504) = 1048576 - 21600 = 1026976
        # That's about 1M subsets, each requiring O(k) work. Might be slow in Python.
        
        # Better: use the fact that for a subset S of size k, it's regular iff
        # all vertices in S have the same degree within S.
        # We can precompute for each vertex v and each subset S containing v,
        # the degree of v in S. But that's still expensive.
        
        # Alternative: for each subset of size exactly 6, check if regular.
        # If a subset of size > 6 is regular, then every 6-subset of it is also regular?
        # No! A 7-vertex regular graph (e.g., 3-regular on 7 vertices... wait, 7*3/2 is not integer)
        # Actually, a k-regular graph on m vertices requires m*k to be even.
        # Key insight: if S of size k >= 6 is regular, then |S| * d is even where d is the degree.
        # But more importantly, we only need to check subsets of size 6!
        # Why? Because if there's a regular induced subgraph on k >= 6 vertices,
        # then... actually no, a 7-vertex regular graph doesn't necessarily contain a 6-vertex regular induced subgraph.
        
        # Let me reconsider. We need to check ALL subsets of size >= 6.
        # For n <= 22, the total number of subsets of size >= 6 is:
        # sum_{k=6}^{22} C(22,k) = 2^22 - sum_{k=0}^{5} C(22,k)
        # C(22,0)=1, C(22,1)=22, C(22,2)=231, C(22,3)=1540, C(22,4)=7315, C(22,5)=26334
        # sum = 1+22+231+1540+7315+26334 = 35443
        # 2^22 = 4194304
        # So about 4.16M subsets. Too many to check individually in Python.
        
        # Optimized approach: for each subset, compute degrees vectorized.
        # Actually, let's use a different strategy:
        # For each subset S, the induced subgraph is regular iff the degree sequence is constant.
        # We can compute degrees for all vertices in S simultaneously using matrix operations.
        
        # But even better: use the following observation.
        # A graph is regular iff the adjacency matrix restricted to S has all row sums equal.
        
        # For efficiency, let's check subsets of size 6 first (most common violation),
        # then sizes 7, 8, etc., with early termination.
        
        # Precompute adjacency as integer bitmask for each vertex
        adj_masks = np.zeros(n, dtype=np.int64)
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i, j]:
                    mask |= (1 << j)
            adj_masks[i] = mask
        
        # Check all subsets of size 6
        # For a subset S = {v0, v1, ..., v5}, compute degree of each vi in S
        # deg(vi in S) = popcount(adj_masks[vi] & mask_S)
        
        # Vectorized check for all 6-subsets:
        # This is still C(n,6) subsets. For n=22, C(22,6) = 74613.
        # For each, we do 6 popcount operations. That's ~447K operations, feasible.
        
        for size in range(6, n + 1):
            if time_left() < 5.0:
                return False  # can't finish, assume invalid to be safe
            
            for subset in combinations(range(n), size):
                # Compute mask for this subset
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Compute degree of each vertex in subset
                degrees = []
                for v in subset:
                    # Count bits in adj_masks[v] & mask
                    m = adj_masks[v] & mask
                    # Popcount
                    deg = bin(m).count('1')
                    degrees.append(deg)
                
                # Check if all degrees are equal
                if len(set(degrees)) == 1:
                    return False  # Found a regular induced subgraph
        
        return True  # No regular induced subgraph of size >= 6
=======
"""Deterministic algebraic construction + rapid vectorized verification for
graphs with no induced regular subgraph on >= 6 vertices.

Strategy:
  1. Generate candidate graphs via:
     - Paley-type graphs over small finite fields
     - Random graphs with controlled density
     - Greedy constructive search with fast pruning
  2. For each candidate, verify NO subset of size >= 6 induces a regular graph.
     Verification uses NumPy bitmask operations for speed.
  3. Spectral-guided edge-flip local search to improve candidates.
  4. Return the largest valid graph found within the time budget.
"""
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - start_time)

    # --- Fast verification using bitmask popcount ---
    def verify_no_regular_subgraph(adj, n):
        """
        adj: n x n boolean numpy array (symmetric, zero diagonal)
        n: number of vertices
        Returns True if no subset of size >= 6 induces a regular graph.
        """
        # Precompute adjacency as integer bitmask for each vertex
        adj_masks = np.zeros(n, dtype=np.int64)
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i, j]:
                    mask |= (1 << j)
            adj_masks[i] = mask
        
        # Popcount lookup table for bytes
        popcount_table = np.zeros(256, dtype=np.int8)
        for i in range(256):
            popcount_table[i] = bin(i).count('1')
        
        def popcount(x):
            """Fast popcount for 64-bit integers using lookup table."""
            return int(popcount_table[x & 0xFF] + popcount_table[(x >> 8) & 0xFF] + 
                       popcount_table[(x >> 16) & 0xFF] + popcount_table[(x >> 24) & 0xFF] +
                       popcount_table[(x >> 32) & 0xFF] + popcount_table[(x >> 40) & 0xFF] +
                       popcount_table[(x >> 48) & 0xFF] + popcount_table[(x >> 56) & 0xFF])
        
        # Check all subsets of size 6 first (most common violation)
        for subset in combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Compute degree of each vertex in subset
            deg0 = popcount(int(adj_masks[subset[0]]) & mask)
            deg1 = popcount(int(adj_masks[subset[1]]) & mask)
            deg2 = popcount(int(adj_masks[subset[2]]) & mask)
            deg3 = popcount(int(adj_masks[subset[3]]) & mask)
            deg4 = popcount(int(adj_masks[subset[4]]) & mask)
            deg5 = popcount(int(adj_masks[subset[5]]) & mask)
            
            # Check if all degrees are equal
            if deg0 == deg1 == deg2 == deg3 == deg4 == deg5:
                return False  # Found a regular induced subgraph
        
        # Check subsets of size 7
        for subset in combinations(range(n), 7):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            deg0 = popcount(int(adj_masks[subset[0]]) & mask)
            deg1 = popcount(int(adj_masks[subset[1]]) & mask)
            deg2 = popcount(int(adj_masks[subset[2]]) & mask)
            deg3 = popcount(int(adj_masks[subset[3]]) & mask)
            deg4 = popcount(int(adj_masks[subset[4]]) & mask)
            deg5 = popcount(int(adj_masks[subset[5]]) & mask)
            deg6 = popcount(int(adj_masks[subset[6]]) & mask)
            
            if deg0 == deg1 == deg2 == deg3 == deg4 == deg5 == deg6:
                return False
        
        # Check subsets of size 8
        for subset in combinations(range(n), 8):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            deg0 = popcount(int(adj_masks[subset[0]]) & mask)
            deg1 = popcount(int(adj_masks[subset[1]]) & mask)
            deg2 = popcount(int(adj_masks[subset[2]]) & mask)
            deg3 = popcount(int(adj_masks[subset[3]]) & mask)
            deg4 = popcount(int(adj_masks[subset[4]]) & mask)
            deg5 = popcount(int(adj_masks[subset[5]]) & mask)
            deg6 = popcount(int(adj_masks[subset[6]]) & mask)
            deg7 = popcount(int(adj_masks[subset[7]]) & mask)
            
            if deg0 == deg1 == deg2 == deg3 == deg4 == deg5 == deg6 == deg7:
                return False
        
        # For sizes 9+, use a more efficient check
        for size in range(9, n + 1):
            if time_left() < 5.0:
                return False  # can't finish, assume invalid to be safe
            
            for subset in combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Compute degrees
                degrees = [popcount(int(adj_masks[v]) & mask) for v in subset]
                
                # Check if all degrees are equal
                if all(d == degrees[0] for d in degrees):
                    return False
        
        return True  # No regular induced subgraph of size >= 6
>>>>>>> REPLACE

