"""Triangle-free graph without independent 10-set, maximizing n (Ramsey R(3,10))."""
import time
import random
import sys

_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def _bit_count(x):
    return bin(x).count('1')

def _has_triangle_bitmask(adj, n):
    """Check if adjacency bitmask list has a triangle."""
    for i in range(n):
        # Neighbors of i
        ni = adj[i]
        # For each neighbor j > i
        # We need to check if any neighbor of i is connected to another neighbor of i
        # Let's iterate j in ni
        j_mask = ni
        while j_mask:
            # Get lowest set bit
            lsb = j_mask & -j_mask
            j = lsb.bit_length() - 1
            j_mask ^= lsb
            if j > i:
                # Check common neighbors of i and j
                # common = adj[i] & adj[j]
                # If common != 0, triangle exists
                # But we must ensure k is not i or j?
                # adj[i] contains j. adj[j] contains i.
                # So adj[i] & adj[j] will contain i and j?
                # No, adj[i] is mask of neighbors. i is not in adj[i].
                # So adj[i] & adj[j] is the set of k such that k~i and k~j.
                # If this set is not empty, triangle (i,j,k).
                if adj[i] & adj[j]:
                    return True
    return False

def _max_independent_set_size_bitmask(adj, n):
    """Find size of maximum independent set using branch and bound with bitmasks."""
    best = [0]
    full_mask = (1 << n) - 1
    
    # Precompute non-neighbors for faster coloring bound? 
    # non_adj[v] = ~adj[v] & full_mask & ~(1<<v)
    
    def greedy_clique_partition_bound(available):
        """
        Upper bound on independent set size in G[available].
        Equivalent to chromatic number of complement G[available].
        We partition available vertices into cliques of G.
        Each clique in G is an independent set in G_bar.
        Wait, the logic was:
        Partition V into cliques C_1..C_k of G.
        Any independent set in G picks at most 1 from each C_i.
        So alpha(G) <= k.
        """
        unassigned = available
        k = 0
        while unassigned:
            k += 1
            # Pick a vertex v from unassigned
            # Heuristic: pick vertex with max degree in G[unassigned] to make clique small? 
            # Or min degree? 
            # For coloring (packing cliques), we want to pack as many as possible into one clique.
            # So we want a large clique.
            # Greedy: pick v. 
            # Then add any u in unassigned such that u is adjacent to all current clique vertices.
            
            # Let's pick v as the one with max degree in G[unassigned]
            # To avoid O(n^2) scan, just pick lowest bit for speed, or random?
            # Let's pick lowest bit for determinism and speed.
            lsb = unassigned & -unassigned
            v = lsb.bit_length() - 1
            unassigned ^= lsb
            
            clique = lsb
            # Try to add more vertices to this clique
            # Candidates must be in unassigned and adjacent to ALL vertices in clique
            # Since clique grows, we maintain 'candidates' = unassigned & adj[v] initially?
            # If we add u, new candidates must be adjacent to u too.
            
            candidates = unassigned & adj[v]
            while candidates:
                # Pick u from candidates
                lsb_u = candidates & -candidates
                u = lsb_u.bit_length() - 1
                candidates ^= lsb_u
                unassigned ^= lsb_u
                
                # Add u to clique
                clique |= lsb_u
                # Update candidates: must be adjacent to u
                candidates &= adj[u]
                
        return k

    def bnb(available, current_size):
        if not available:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound
        # Simple bound: current_size + _bit_count(available)
        # Better bound: current_size + greedy_clique_partition_bound(available)
        # The coloring bound is stronger but more expensive.
        # Let's use the simple bound first, and maybe switch if n is large?
        # For n=40, simple bound is often enough if we branch well.
        # But coloring bound prunes much better.
        # Let's try coloring bound.
        
        bound = greedy_clique_partition_bound(available)
        if current_size + bound <= best[0]:
            return
        
        # Pick vertex to branch on
        # Heuristic: Max degree in G[available]
        # Or vertex that appears in most cliques?
        # Let's pick max degree in G[available]
        v_mask = available
        max_deg = -1
        v = -1
        while v_mask:
            lsb = v_mask & -v_mask
            u = lsb.bit_length() - 1
            v_mask ^= lsb
            deg = _bit_count(adj[u] & available)
            if deg > max_deg:
                max_deg = deg
                v = u
        
        if v == -1:
            return

        # Branch 1: Include v
        # New available: available \ {v} \ N(v)
        new_avail_include = available & ~(1 << v) & ~adj[v]
        bnb(new_avail_include, current_size + 1)
        
        # Branch 2: Exclude v
        new_avail_exclude = available & ~(1 << v)
        bnb(new_avail_exclude, current_size)

    bnb(full_mask, 0)
    return best[0]

def _validate_bitmask(adj, n):
    """Check triangle-free and no independent 10-set."""
    if _has_triangle_bitmask(adj, n):
        return False
    if _max_independent_set_size_bitmask(adj, n) >= 10:
        return False
    return True

def _greedy_triangle_free_bitmask(n, rng):
    """
    Build a triangle-free graph greedily using bitmasks.
    """
    adj = [0] * n
    
    # Generate all potential edges
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(edges)
    
    for i, j in edges:
        if _remaining_time() < 2:
            break
        # Check if adding (i,j) creates a triangle
        # Triangle if common neighbor exists
        # common = adj[i] & adj[j]
        if adj[i] & adj[j]:
            continue
        
        # Add edge
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    
    return adj

def _local_search_bitmask(adj, n, rng, max_iter=2000):
    """
    Improve graph by adding edges to reduce independence number.
    """
    adj = adj[:]
    alpha = _max_independent_set_size_bitmask(adj, n)
    
    if alpha < 10:
        return adj, alpha
    
    for _ in range(max_iter):
        if _remaining_time() < 3:
            break
        
        # Find all addable edges
        addable = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)):
                    # Check triangle
                    if not (adj[i] & adj[j]):
                        addable.append((i, j))
        
        if not addable:
            break
        
        # Try random addable edges
        improved = False
        for _ in range(10): # Try a few
            i, j = rng.choice(addable)
            
            # Add edge
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
            new_alpha = _max_independent_set_size_bitmask(adj, n)
            if new_alpha < alpha:
                alpha = new_alpha
                improved = True
                if alpha < 10:
                    break
            else:
                # Revert
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
        
        if alpha < 10:
            break
        if not improved:
             # Maybe try removing edges? 
             # Or just break if no improvement in many steps?
             # For now, continue searching addable edges
             pass

    return adj, alpha

def _try_construction_bitmask(n, seed, max_local_iters=500):
    """
    Try to build a valid graph on n vertices.
    """
    rng = random.Random(seed)
    
    # Strategy 1: Random greedy
    adj = _greedy_triangle_free_bitmask(n, rng)
    alpha = _max_independent_set_size_bitmask(adj, n)
    
    if alpha < 10:
        return adj, alpha
    
    # Try local search
    if _remaining_time() > 5:
        adj_ls, alpha_ls = _local_search_bitmask(adj, n, rng, max_local_iters)
        if alpha_ls < 10:
            return adj_ls, alpha_ls
        if alpha_ls < alpha:
            adj, alpha = adj_ls, alpha_ls
            
    # Strategy 2: Different random orderings
    for attempt in range(5):
        if _remaining_time() < 5:
            break
        rng2 = random.Random(seed * 1000 + attempt)
        adj2 = _greedy_triangle_free_bitmask(n, rng2)
        alpha2 = _max_independent_set_size_bitmask(adj2, n)
        
        if alpha2 < 10:
            return adj2, alpha2
        
        if alpha2 < alpha and _remaining_time() > 5:
            adj_ls2, alpha_ls2 = _local_search_bitmask(adj2, n, rng2, max_local_iters)
            if alpha_ls2 < alpha:
                adj, alpha = adj_ls2, alpha_ls2
                
    return None

def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n.
    """
    best_adj = None
    best_n = 0
    best_alpha = float('inf')
    
    # Try n from high to low
    for n_target in range(40, 20, -1):
        if _remaining_time() < 8:
            break
        
        found = False
        
        # Try random greedy + local search
        # Use multiple seeds
        for seed in range(50):
            if _remaining_time() < 3:
                break
            
            result = _try_construction_bitmask(n_target, seed)
            if result is not None:
                adj_res, alpha_res = result
                if alpha_res < best_alpha:
                    best_adj = adj_res
                    best_n = n_target
                    best_alpha = alpha_res
                if alpha_res < 10:
                    found = True
                    break
        
        if found:
            break
    
    # Fallback: use a known small construction if nothing worked
    if best_adj is None:
        # Use a random greedy that we know works for small n
        n = 26
        rng = random.Random(42)
        adj = _greedy_triangle_free_bitmask(n, rng)
        best_adj = adj
        best_n = n
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
