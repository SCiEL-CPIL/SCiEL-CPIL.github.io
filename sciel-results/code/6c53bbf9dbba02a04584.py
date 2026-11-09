"""
Combined approach: Polarity graph base with optimized exact independence check
and accelerated extension using structural guarantees and better branching.
"""

import time
import random
import sys
from functools import lru_cache

sys.setrecursionlimit(10000)

def _popcount(x):
    return bin(x).count('1')

def _has_c4(adj, adj_bits, n):
    """Check if graph contains a C4."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj_bits[u] & adj_bits[v]
            if common & (common - 1):
                return True
    return False

def _can_add_edge(adj, adj_bits, u, v, n):
    """Check if adding edge (u, v) would create a C4."""
    for x in adj[v]:
        if x == u:
            continue
        if adj_bits[u] & adj_bits[x]:
            return False
    for y in adj[u]:
        if y == v:
            continue
        if adj_bits[v] & adj_bits[y]:
            return False
    return True

def _add_edge(adj, adj_bits, u, v):
    """Add edge u-v, update structures."""
    adj[u].add(v)
    adj[v].add(u)
    adj_bits[u] |= (1 << v)
    adj_bits[v] |= (1 << u)

def _remove_edge(adj, adj_bits, u, v):
    """Remove edge u-v, update structures."""
    adj[u].discard(v)
    adj[v].discard(u)
    adj_bits[u] &= ~(1 << v)
    adj_bits[v] &= ~(1 << u)

def _has_independent_set_k(adj_bits, n, k):
    """
    Fast exact check for independent set of size k using bitset-based
    branch and bound with aggressive pruning.
    """
    if k <= 0:
        return True
    if n < k:
        return False
    if k == 1:
        return n > 0

    full_mask = (1 << n) - 1
    
    # Precompute non-adjacency masks (including self-exclusion)
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = full_mask & ~(adj_bits[v] | (1 << v))
    
    # Greedy coloring upper bound on alpha(G[rem])
    # This is the chromatic number of the complement, which gives an upper bound
    # on the number of independent sets needed to cover V, but more importantly
    # for our purposes, it's used as a pruning heuristic
    def greedy_coloring_upper_bound(rem):
        uncolored = rem
        color_class = []
        num_colors = 0
        
        while uncolored:
            # Pick vertex with max degree in G (min degree in G')
            best_v = -1
            best_deg = -1
            temp = uncolored
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = _popcount(adj_bits[v] & uncolored)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                break
                
            v_bit = 1 << best_v
            assigned_color = -1
            for c in range(num_colors):
                if non_adj[best_v] & color_class[c]:
                    continue
                assigned_color = c
                break
            
            if assigned_color == -1:
                assigned_color = num_colors
                color_class.append(0)
                num_colors += 1
            
            color_class[assigned_color] |= v_bit
            uncolored ^= v_bit
        
        return num_colors

    cache = {}
    
    def search(rem, size):
        if size >= k:
            return True
        if rem == 0:
            return False
        
        if size + _popcount(rem) < k:
            return False
        
        key = (rem, size)
        if key in cache:
            return cache[key]
        
        # Upper bound pruning
        ub = greedy_coloring_upper_bound(rem)
        if size + ub < k:
            cache[key] = False
            return False
        
        # Choose branching vertex: highest degree in G[rem]
        best_v = -1
        best_deg = -1
        temp = rem
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = _popcount(adj_bits[v] & rem)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            cache[key] = False
            return False
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        new_rem_include = rem & non_adj[best_v]
        if search(new_rem_include, size + 1):
            cache[key] = True
            return True
            
        # Branch 2: Exclude best_v
        new_rem_exclude = rem & ~v_bit
        if search(new_rem_exclude, size):
            cache[key] = True
            return True
            
        cache[key] = False
        return False
        
    return search(full_mask, 0)

def _polarity_graph_pg25():
    """
    Construct the polarity graph of PG(2,5).
    Points are vectors in GF(5)^3, up to scale.
    There are (5^3 - 1)/(5 - 1) = 31 points.
    Two points are adjacent if their dot product is 0.
    This graph is C4-free.
    """
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    nx = 1
                    ny = (y * inv_x) % 5
                    nz = (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx = 0
                    ny = 1
                    nz = (z * inv_y) % 5
                else:
                    inv_z = pow(z, 3, 5)
                    nx = 0
                    ny = 0
                    nz = 1
                points.append((nx, ny, nz))
    
    unique_points = list(set(points))
    unique_points.sort()
    n = len(unique_points)
    
    adj = [set() for _ in range(n)]
    adj_bits = [0] * n
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = unique_points[i]
            p2 = unique_points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                _add_edge(adj, adj_bits, i, j)
                
    return n, adj, adj_bits

def _try_add_vertex_structural(adj, adj_bits, n, v_new, rng, deadline):
    """
    Try to add a new vertex by finding a valid neighborhood.
    Uses structural properties to guide the search.
    """
    # Strategy: Try different neighborhood sizes and compositions
    # The new vertex must connect to a set S such that:
    # 1. No C4 is created (structural constraint)
    # 2. alpha < 11 is maintained
    
    # Try greedy construction with different random orders
    for attempt in range(100):
        if time.time() > deadline:
            return False, None, None
        
        vertices = list(range(n))
        rng.shuffle(vertices)
        
        S = set()
        S_adj_union = 0
        
        # Greedily build neighborhood
        for u in vertices:
            if time.time() > deadline:
                return False, None, None
            
            # Check if adding u to S would create a C4
            # This happens if u shares a neighbor with some existing vertex in S
            # that is also adjacent to the new vertex
            valid = True
            for w in S:
                if adj_bits[u] & adj_bits[w]:
                    valid = False
                    break
            
            if valid:
                S.add(u)
                S_adj_union |= adj_bits[u]
        
        if not S:
            continue
        
        # Create copy and add new vertex
        adj_copy = [set(s) for s in adj]
        adj_bits_copy = list(adj_bits)
        
        adj_copy.append(set(S))
        adj_bits_copy.append(0)
        
        new_bits = 0
        for u in S:
            adj_copy[u].add(v_new)
            adj_bits_copy[u] |= (1 << v_new)
            new_bits |= (1 << u)
        adj_bits_copy[v_new] = new_bits
        
        # Check if this maintains alpha < 11
        if not _has_independent_set_k(adj_bits_copy, n + 1, 11):
            return True, adj_copy, adj_bits_copy
    
    return False, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # 1. Start with Polarity Graph of PG(2,5)
    n0, adj, adj_bits = _polarity_graph_pg25()
    n = n0
    best_n = n
    best_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                best_edges.append([u, v])
                
    # Verify base graph
    if _has_c4(adj, adj_bits, n):
        pass
    if _has_independent_set_k(adj_bits, n, 11):
        pass
        
    rng = random.Random(42)
    max_n = 43
    
    # 2. Extend by adding vertices one by one
    for target_n in range(n0 + 1, max_n + 1):
        if time.time() > deadline - 5:
            break
            
        success = False
        # Try many random configurations for the new vertex's neighborhood
        for r in range(200):
            if time.time() > deadline - 5:
                break
                
            ok, adj_new, adj_bits_new = _try_add_vertex_structural(
                adj, adj_bits, n, n, rng, deadline - 5
            )
            
            if ok:
                adj = adj_new
                adj_bits = adj_bits_new
                n += 1
                success = True
                break
        
        if not success:
            break
        
        best_n = n
        best_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    best_edges.append([u, v])

    # 3. Fallback: try direct randomized greedy for larger n if we're stuck
    if best_n < 39 and time.time() < deadline - 10:
        for target_n in range(min(43, best_n + 10), best_n, -1):
            if time.time() > deadline - 5:
                break
            found = False
            for r in range(80):
                if time.time() > deadline - 5:
                    break
                adj_t = [set() for _ in range(target_n)]
                adj_bits_t = [0] * target_n
                pairs = [(u, v) for u in range(target_n) for v in range(u + 1, target_n)]
                rng.shuffle(pairs)
                for u, v in pairs:
                    if _can_add_edge(adj_t, adj_bits_t, u, v, target_n):
                        _add_edge(adj_t, adj_bits_t, u, v)
                if not _has_c4(adj_t, adj_bits_t, target_n):
                    if not _has_independent_set_k(adj_bits_t, target_n, 11):
                        best_n = target_n
                        best_edges = []
                        for u in range(target_n):
                            for v in adj_t[u]:
                                if u < v:
                                    best_edges.append([u, v])
                        found = True
                        break
            if found:
                break

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Constructed graph with n={res['n']}")
