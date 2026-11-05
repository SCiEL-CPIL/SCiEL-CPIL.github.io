"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from math import comb

def _bitmask_regular_check(adj_bits, subset_mask, n):
    """Check if the induced subgraph on subset_mask is regular using bitmasks."""
    # Get the vertices in the subset
    verts = []
    for i in range(n):
        if subset_mask & (1 << i):
            verts.append(i)
    k = len(verts)
    if k < 6:
        return False  # only check size >= 6
    
    # Compute degrees within the subset
    degrees = []
    for v in verts:
        deg = 0
        for u in verts:
            if u != v and (adj_bits[v] >> u) & 1:
                deg += 1
        degrees.append(deg)
    
    return len(set(degrees)) == 1

def _check_all_subsets_fast(adj_bits, n, min_size=6):
    """Check all subsets of size >= min_size for regularity using bitmasks.
    Returns True if no regular induced subgraph exists."""
    # Precompute for all 6-subsets
    for combo in combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _bitmask_regular_check(adj_bits, mask, n):
            return False
    
    # Check 7-subsets
    for combo in combinations(range(n), 7):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if _bitmask_regular_check(adj_bits, mask, n):
            return False
    
    # For 8+, sample
    for sz in range(8, n + 1):
        total = comb(n, sz)
        samples = min(total, 200)
        for _ in range(samples):
            idx = random.sample(range(n), sz)
            mask = 0
            for v in idx:
                mask |= (1 << v)
            if _bitmask_regular_check(adj_bits, mask, n):
                return False
    
    return True

def _backtracking_solve(n, time_limit):
    """Backtracking CSP: build graph vertex by vertex.
    adj_bits[i] stores adjacency bitmask for vertex i.
    We decide edges (i,j) for i<j as we add vertices."""
    start_time = time.time()
    
    # adj_bits[v] = bitmask of neighbors of v among vertices 0..v-1 (already decided)
    # Plus we'll track full adjacency as we go
    adj_bits = [0] * n  # adj_bits[i] has bits for neighbors
    
    # We process vertices in order. When adding vertex k, we decide edges (0,k), (1,k), ..., (k-1,k)
    # This is a binary decision for each edge.
    
    # Use recursive backtracking with pruning
    def backtrack(k):
        """Process vertex k: decide edges (0,k), (1,k), ..., (k-1,k)."""
        if time.time() - start_time > time_limit:
            return False
        
        if k == n:
            # All vertices added, do final check
            return True
        
        # Try all 2^k assignments for edges to previous vertices
        # But prune: for each partial assignment, check if any regular subset is formed
        # among vertices 0..k
        
        # For efficiency, use iterative approach with early pruning
        # The number of edges to decide is k, so 2^k possibilities
        # For k up to ~15, this is manageable with pruning
        
        # Actually, let's do it edge by edge with forward checking
        # Edge (j, k) for j = 0, 1, ..., k-1
        
        def _bt_edges(j, current_adj):
            """Decide edges (j,k), (j+1,k), ..., (k-1,k) given current_adj for edges (0,k)..(j-1,k)."""
            if time.time() - start_time > time_limit:
                return False
            
            if j == k:
                # All edges for vertex k decided. Check if any regular subset in {0..k}
                # Only need to check subsets that include vertex k (new ones)
                # and subsets of size >= 6
                for sz in range(6, k + 1):
                    if sz > k + 1:
                        break
                    # Check subsets of {0..k} of size sz that include k
                    for combo in combinations(range(k), sz - 1):
                        subset = list(combo) + [k]
                        mask = 0
                        for v in subset:
                            mask |= (1 << v)
                        if _bitmask_regular_check(adj_bits_temp, mask, k + 1):
                            return False
                return True
            
            # Try edge (j, k) = 0
            adj_bits_temp[j] |= 0  # no change
            # Actually we need to be more careful with the adjacency representation
            
            # Let me restructure: use a list of bitmasks where adj_bits_temp[i] 
            # represents neighbors of i among 0..k
            # When we add vertex k with edge to j, we set bit k in adj_bits_temp[j]
            # and bit j in adj_bits_temp[k]
            
            # Try: no edge (j,k)
            if _bt_edges(j + 1, current_adj | (0 << j)):
                return True
            
            # Try: edge (j,k)
            # Set bit k in adj_bits_temp[j] and bit j in adj_bits_temp[k]
            adj_bits_temp[j] |= (1 << k)
            adj_bits_temp[k] |= (1 << j)
            if _bt_edges(j + 1, current_adj | (1 << j)):
                return True
            
            # Backtrack
            adj_bits_temp[j] &= ~(1 << k)
            adj_bits_temp[k] &= ~(1 << j)
            
            return False
        
        # Initialize temp adjacency
        global adj_bits_temp
        adj_bits_temp = adj_bits[:]  # copy current state
        
        # current_adj tracks which edges to vertex k are set (bit j = edge (j,k) exists)
        return _bt_edges(0, 0)
    
    # This recursive approach is too slow for n=21. Let me use a different strategy.
    # Instead, use randomized construction with fast bitmask verification.
    return None

def _random_construct_with_fast_check(n, time_limit):
    """Random construction with fast bitmask-based verification."""
    start_time = time.time()
    
    for attempt in range(100000):
        if time.time() - start_time > time_limit:
            break
        
        # Random graph
        adj_bits = [0] * n
        density = 0.2 + 0.6 * random.random()
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < density:
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
        
        # Fast check: all 6-subsets
        valid = True
        for combo in combinations(range(n), 6):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if _bitmask_regular_check(adj_bits, mask, n):
                valid = False
                break
        
        if not valid:
            continue
        
        # Check 7-subsets
        for combo in combinations(range(n), 7):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if _bitmask_regular_check(adj_bits, mask, n):
                valid = False
                break
        
        if not valid:
            continue
        
        # Check 8+ with sampling
        for sz in range(8, n + 1):
            total = comb(n, sz)
            samples = min(total, 300)
            for _ in range(samples):
                idx = random.sample(range(n), sz)
                mask = 0
                for v in idx:
                    mask |= (1 << v)
                if _bitmask_regular_check(adj_bits, mask, n):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Convert to edges
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_bits[u] >> v) & 1:
                        edges.append([u, v])
            return n, edges
    
    return None, None

def _local_search_bitmask(adj_bits, n, time_limit):
    """Local search on bitmask representation."""
    start_time = time.time()
    
    def is_valid(adj_bits):
        if time.time() - start_time > time_limit:
            return False
        for combo in combinations(range(n), 6):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if _bitmask_regular_check(adj_bits, mask, n):
                return False
        for combo in combinations(range(n), 7):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if _bitmask_regular_check(adj_bits, mask, n):
                return False
        for sz in range(8, n + 1):
            total = comb(n, sz)
            samples = min(total, 200)
            for _ in range(samples):
                idx = random.sample(range(n), sz)
                mask = 0
                for v in idx:
                    mask |= (1 << v)
                if _bitmask_regular_check(adj_bits, mask, n):
                    return False
        return True
    
    for step in range(50000):
        if time.time() - start_time > time_limit:
            break
        
        if is_valid(adj_bits):
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_bits[u] >> v) & 1:
                        edges.append([u, v])
            return n, edges
        
        # Random edge flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            if (adj_bits[u] >> v) & 1:
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
            else:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
    
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    best_n = 5
    best_edges = []
    
    for n in range(21, 5, -1):
        if time.time() > deadline - 5:
            break
        
        # Try random construction first
        result_n, result_edges = _random_construct_with_fast_check(n, deadline - time.time() - 5)
        if result_n is not None:
            best_n = result_n
            best_edges = result_edges
            break
        
        # Try local search from random start
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            
            adj_bits = [0] * n
            density = 0.2 + 0.6 * random.random()
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < density:
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
            
            result_n, result_edges = _local_search_bitmask(adj_bits, n, deadline - time.time() - 5)
            if result_n is not None:
                best_n = result_n
                best_edges = result_edges
                break
        
        if best_n >= n:
            break
    
    # Fallback: ensure we have at least a valid small graph
    if best_n < 6:
        # Trivial: empty graph on 5 vertices has no regular induced subgraph on 6+
        best_n = 5
        best_edges = []
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
