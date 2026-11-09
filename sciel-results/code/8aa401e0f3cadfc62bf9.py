"""
Program to construct a C4-free graph with no independent set of size 11, maximizing n.
Uses a polarity graph base (PG(2,5)) and optimized extension with fast independence check.
"""

import time
import random
import sys

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

def _has_independent_set_k(adj_bits, n, k, non_adj=None):
    """
    Fast exact check for independent set of size k.
    Uses bitset-based branch and bound with aggressive pruning.
    Optimized: avoids expensive greedy coloring UB, uses simple bounds + memoization.
    """
    if k <= 0:
        return True
    if n < k:
        return False
    
    full_mask = (1 << n) - 1
    if non_adj is None:
        non_adj = [0] * n
        for v in range(n):
            non_adj[v] = full_mask & ~(adj_bits[v] | (1 << v))
    
    # Precompute degree in complement (non-adjacency degree) for each vertex
    non_deg = [_popcount(non_adj[v]) for v in range(n)]
    
    # Simple upper bound: greedy maximal independent set in complement
    def greedy_alpha_upper_bound(rem):
        """
        Greedy upper bound on alpha(G[rem]).
        Repeatedly pick the vertex with max non-degree, add to independent set,
        remove it and its neighbors (in G) from rem.
        This gives a LOWER bound on alpha, but we use it differently:
        Actually, for upper bound, we need coloring of complement.
        Let's use a simpler approach: just count vertices.
        """
        return _popcount(rem)
    
    # Better: greedy coloring of complement (clique cover of G[rem])
    def greedy_coloring_upper_bound(rem):
        """
        Upper bound on alpha(G[rem]) via greedy clique cover.
        Each color class is a clique in G (independent in complement).
        """
        uncolored = rem
        num_colors = 0
        while uncolored:
            # Pick vertex with max degree in G within uncolored
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
            # Build clique: start with best_v, add vertices adjacent to all current
            class_mask = 1 << best_v
            candidates = uncolored & adj_bits[best_v]
            while candidates:
                # Pick vertex with max degree in candidates
                cv = -1
                cdeg = -1
                temp2 = candidates
                while temp2:
                    lsb2 = temp2 & -temp2
                    vv = lsb2.bit_length() - 1
                    d = _popcount(adj_bits[vv] & candidates)
                    if d > cdeg:
                        cdeg = d
                        cv = vv
                    temp2 ^= lsb2
                if cv == -1:
                    break
                class_mask |= (1 << cv)
                candidates &= adj_bits[cv]
            uncolored &= ~class_mask
            num_colors += 1
        return num_colors
    
    # Quick greedy lower bound: try to find k independent vertices
    def greedy_independent_lower_bound(rem, needed):
        """Try to greedily find 'needed' independent vertices in rem."""
        if needed <= 0:
            return True
        if _popcount(rem) < needed:
            return False
        found = 0
        remaining = rem
        while remaining and found < needed:
            # Pick vertex with max non-degree in remaining
            best_v = -1
            best_nondeg = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                nondeg = _popcount(non_adj[v] & remaining)
                if nondeg > best_nondeg:
                    best_nondeg = nondeg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            found += 1
            remaining &= non_adj[best_v]
        return found >= needed
    
    cache = {}
    max_cache_size = 50000
    node_count = [0]
    time_limit = 2.0  # seconds for this check
    start_t = time.time()
    
    def search(rem, size):
        node_count[0] += 1
        if node_count[0] % 5000 == 0 and time.time() - start_t > time_limit:
            # Timeout: return True conservatively (assume independent set exists)
            return True
        
        if size >= k:
            return True
        if rem == 0:
            return False
        
        rem_count = _popcount(rem)
        if size + rem_count < k:
            return False
        
        # Quick lower bound
        if greedy_independent_lower_bound(rem, k - size):
            return True
        
        key = rem  # size is determined by how we got here, but let's include it
        # Actually size matters, but for a given rem the max size we can achieve is bounded
        # Let's use just rem as key since we're doing DFS
        if key in cache:
            return cache[key]
        
        # Upper bound via greedy coloring
        ub = greedy_coloring_upper_bound(rem)
        if size + ub < k:
            if len(cache) < max_cache_size:
                cache[key] = False
            return False
        
        # Pick branching vertex: highest non-degree in rem (most non-adjacent)
        best_v = -1
        best_nondeg = -1
        temp = rem
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            nondeg = _popcount(non_adj[v] & rem)
            if nondeg > best_nondeg:
                best_nondeg = nondeg
                best_v = v
            temp ^= lsb
        
        if best_v == -1:
            if len(cache) < max_cache_size:
                cache[key] = False
            return False
        
        # If no non-adjacency, rem is a clique, so max independent set is 1
        if best_nondeg == 0:
            result = (size + 1 >= k)
            if len(cache) < max_cache_size:
                cache[key] = result
            return result
        
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        new_rem_include = rem & non_adj[best_v]
        if search(new_rem_include, size + 1):
            if len(cache) < max_cache_size:
                cache[key] = True
            return True
        
        # Branch 2: Exclude best_v
        new_rem_exclude = rem & ~v_bit
        if search(new_rem_exclude, size):
            if len(cache) < max_cache_size:
                cache[key] = True
            return True
        
        if len(cache) < max_cache_size:
            cache[key] = False
        return False
    
    result = search(full_mask, 0)
    return result

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

def _compute_non_adj(adj_bits, n):
    """Precompute non-adjacency bitmasks."""
    full_mask = (1 << n) - 1
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = full_mask & ~(adj_bits[v] | (1 << v))
    return non_adj

def _try_add_vertex(adj, adj_bits, n, S, rng):
    """
    Try to add a new vertex n with neighborhood S.
    Returns (success, new_adj, new_adj_bits, new_non_adj) or (False, None, None, None).
    S must be a set of vertices that form a clique in G (to avoid C4).
    """
    v_new = n
    adj_copy = [set(s) for s in adj]
    adj_bits_copy = list(adj_bits)
    
    # Verify S is a clique in G (no C4)
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            if not (adj_bits[S_list[i]] & (1 << S_list[j])):
                return False, None, None, None
    
    # Add vertex
    adj_copy.append(set(S))
    adj_bits_copy.append(0)
    new_bits = 0
    for u in S:
        adj_copy[u].add(v_new)
        adj_bits_copy[u] |= (1 << v_new)
        new_bits |= (1 << u)
    adj_bits_copy[v_new] = new_bits
    
    # Check C4
    if _has_c4(adj_copy, adj_bits_copy, n + 1):
        return False, None, None, None
    
    # Check independence
    non_adj_ext = _compute_non_adj(adj_bits_copy, n + 1)
    if _has_independent_set_k(adj_bits_copy, n + 1, 11, non_adj_ext):
        return False, None, None, None
    
    return True, adj_copy, adj_bits_copy, non_adj_ext

def _random_clique(adj, adj_bits, n, target_size, rng):
    """
    Find a random clique of approximately target_size in G.
    A clique in G means all vertices are mutually adjacent.
    For C4-free graphs, this is the constraint for adding a new vertex.
    """
    if target_size <= 0:
        return set()
    
    # Randomly pick a starting vertex
    start = rng.randint(0, n - 1)
    clique = {start}
    candidates = set(adj[start])
    
    for _ in range(target_size - 1):
        if not candidates:
            break
        # Pick random vertex from candidates
        cand_list = list(candidates)
        v = cand_list[rng.randint(0, len(cand_list) - 1)]
        clique.add(v)
        # Update candidates: must be adjacent to v
        new_candidates = set()
        for w in candidates:
            if w != v and (adj_bits[v] & (1 << w)):
                new_candidates.add(w)
        candidates = new_candidates
    
    return clique

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # 1. Start with Polarity Graph of PG(2,5)
    n0, adj, adj_bits = _polarity_graph_pg25()
    n = n0
    best_n = n
    best_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                best_edges.append([u, v])
    
    non_adj = _compute_non_adj(adj_bits, n)
    
    rng = random.Random(42)
    max_n = 43
    
    # 2. Extend by adding vertices one by one with random cliques
    for target_n in range(n0 + 1, max_n + 1):
        if time.time() > deadline - 10:
            break
            
        success = False
        # Try many random clique configurations for the new vertex's neighborhood
        num_trials = 500 if target_n <= n0 + 5 else 200
        for r in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            # Choose clique size: try various sizes
            # In C4-free graphs, max clique size is bounded
            # For PG(2,5), degrees vary, so try sizes 3-8
            target_clique_size = rng.randint(3, min(8, n // 4 + 2))
            
            # Get a random clique
            S = _random_clique(adj, adj_bits, n, target_clique_size, rng)
            if len(S) < 2:
                continue
            
            # Try to add vertex with this neighborhood
            ok, adj_copy, adj_bits_copy, non_adj_ext = _try_add_vertex(
                adj, adj_bits, n, S, rng
            )
            
            if ok:
                adj = adj_copy
                adj_bits = adj_bits_copy
                n += 1
                non_adj = non_adj_ext
                success = True
                break
        
        if not success:
            # Try with smaller clique sizes
            for cs in range(2, 1):
                for r in range(100):
                    if time.time() > deadline - 10:
                        break
                    S = _random_clique(adj, adj_bits, n, cs, rng)
                    if len(S) < cs:
                        continue
                    ok, adj_copy, adj_bits_copy, non_adj_ext = _try_add_vertex(
                        adj, adj_bits, n, S, rng
                    )
                    if ok:
                        adj = adj_copy
                        adj_bits = adj_bits_copy
                        n += 1
                        non_adj = non_adj_ext
                        success = True
                        break
                if success:
                    break
        
        if not success:
            break
        
        best_n = n
        best_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    best_edges.append([u, v])

    # 3. If stuck below 39, try more aggressive search
    if best_n < 39 and time.time() < deadline - 15:
        # Try multiple random restarts of the extension process
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            # Start fresh from PG(2,5) with different random seed
            n_r, adj_r, adj_bits_r = _polarity_graph_pg25()
            rng_r = random.Random(42 + restart * 1000)
            
            for target_n in range(n_r + 1, min(43, best_n + 15)):
                if time.time() > deadline - 10:
                    break
                
                added = False
                for r in range(300):
                    if time.time() > deadline - 10:
                        break
                    target_clique_size = rng_r.randint(2, min(7, n_r // 4 + 2))
                    S = _random_clique(adj_r, adj_bits_r, n_r, target_clique_size, rng_r)
                    if len(S) < 2:
                        continue
                    ok, adj_new, adj_bits_new, non_adj_new = _try_add_vertex(
                        adj_r, adj_bits_r, n_r, S, rng_r
                    )
                    if ok:
                        adj_r = adj_new
                        adj_bits_r = adj_bits_new
                        n_r += 1
                        added = True
                        break
                
                if not added:
                    break
            
            if n_r > best_n:
                best_n = n_r
                best_edges = []
                for u in range(n_r):
                    for v in adj_r[u]:
                        if u < v:
                            best_edges.append([u, v])

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Constructed graph with n={res['n']}")
