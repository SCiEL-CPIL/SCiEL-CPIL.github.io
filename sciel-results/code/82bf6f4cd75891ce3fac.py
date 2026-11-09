"""
Program to construct a C4-free graph with no independent set of size 11, maximizing n.
Uses a polarity graph base (PG(2,5)) and optimized extension with fast independence check.
"""

import time
import random
import sys
from functools import lru_cache

sys.setrecursionlimit(10000)

def _popcount(x):
    return bin(x).count('1')

def _has_c4(adj, adj_bits, n):
    """
    Check if graph contains a C4.
    A C4 exists if there are two vertices u, v that have >= 2 common neighbors.
    """
    for u in range(n):
        for v in range(u + 1, n):
            common = adj_bits[u] & adj_bits[v]
            if common & (common - 1):
                return True
    return False

def _can_add_edge(adj, adj_bits, u, v, n):
    """
    Check if adding edge (u, v) would create a C4.
    """
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
    Fast exact check for independent set of size k.
    Uses bitset-based branch and bound with aggressive pruning.
    """
    if k <= 0:
        return True
    if n < k:
        return False
    
    # Precompute: for each vertex, its non-neighbors mask (including itself)
    # non_neighbors[v] = vertices that are NOT adjacent to v (v can be in independent set with them)
    # Actually for independent set: if we pick v, we can only pick vertices NOT adjacent to v
    # non_adj_mask[v] = ~(adj_bits[v] | (1 << v)) & ((1 << n) - 1)
    
    full_mask = (1 << n) - 1
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = full_mask & ~(adj_bits[v] | (1 << v))
    
    # Memoization cache: (rem, size_needed) -> bool
    # But rem can be large, so we use a dict with limited size
    cache = {}
    
    def search(rem, size):
        if size >= k:
            return True
        if rem == 0:
            return False
        
        # Pruning: not enough vertices
        if size + _popcount(rem) < k:
            return False
            
        # Check cache
        key = (rem, size)
        if key in cache:
            return cache[key]
            
        # Additional pruning: if the maximum independent set of the remaining graph
        # is less than what we need, prune.
        # Quick upper bound: greedy coloring of complement? Too expensive.
        # Simpler: if rem has fewer than (k - size) vertices, already handled above.
        
        # Pick branching vertex: highest degree in induced subgraph on rem
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
        
        # Branch 1: Include best_v -> remove v and all its neighbors
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
        if time.time() > deadline - 3:
            break
            
        success = False
        # Try many random configurations for the new vertex's neighborhood
        for r in range(200):
            if time.time() > deadline - 3:
                break
                
            adj_copy = [set(s) for s in adj]
            adj_bits_copy = list(adj_bits)
            
            v_new = n
            
            # Greedy: pick random order, add vertex if no conflict with S
            vertices = list(range(n))
            rng.shuffle(vertices)
            
            S = set()
            # Use bitset for conflict check: S_bits tracks which vertices are in S
            S_bits = 0
            for u in vertices:
                # u conflicts with w in S if adj_bits[u] & adj_bits[w] != 0
                # Equivalent: (adj_bits[u] & S_adj_union) == 0 where S_adj_union = OR of adj_bits[w] for w in S
                # But that's expensive. Instead, check each w in S.
                # Optimize: precompute union of adj_bits for vertices in S
                # Actually, u can be added iff for all w in S: (adj_bits[u] & adj_bits[w]) == 0
                # This is equivalent to: adj_bits[u] & (OR of adj_bits[w] for w in S) == 0
                # Let's maintain S_adj_union as we build S
                # But we don't have it in the loop. Let's just check directly for small S.
                can_add = True
                for w in S:
                    if adj_bits_copy[u] & adj_bits_copy[w]:
                        can_add = False
                        break
                if can_add:
                    S.add(u)
                    S_bits |= (1 << u)
            
            # Add v_new to the graph
            adj_copy.append(set(S))
            adj_bits_copy.append(0)
            new_bits = 0
            for u in S:
                adj_copy[u].add(v_new)
                adj_bits_copy[u] |= (1 << v_new)
                new_bits |= (1 << u)
            adj_bits_copy[v_new] = new_bits
            
            # Check alpha
            if not _has_independent_set_k(adj_bits_copy, n + 1, 11):
                adj = adj_copy
                adj_bits = adj_bits_copy
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
    if best_n < 35 and time.time() < deadline - 10:
        for target_n in range(min(43, best_n + 10), best_n, -1):
            if time.time() > deadline - 3:
                break
            found = False
            for r in range(50):
                if time.time() > deadline - 3:
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
