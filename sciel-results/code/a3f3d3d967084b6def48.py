"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys
from collections import defaultdict

# Deadline guard
DEADLINE = 115.0  # seconds, leaving safety margin before 120s

def _is_c6_free(adj_matrix, n=40):
    """Check if graph with adjacency matrix contains no C6 (6-cycle).
    A C6 is a cycle of length exactly 6: v0-v1-v2-v3-v4-v5-v0
    where all vertices are distinct and edges exist.
    """
    # For each pair of non-adjacent vertices, check if there's a path of length 5 between them
    # that doesn't repeat vertices (which would form a C6)
    # More efficient: for each edge (u,v), check if there's a path of length 4 from u to v
    # through distinct vertices (excluding u and v at the endpoints being counted)
    
    # Actually, let's use a different approach:
    # A C6 exists if there are vertices v0,v1,v2,v3,v4,v5 all distinct with edges
    # (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v5),(v5,v0)
    
    # Efficient check: for each pair (a,b) of non-adjacent vertices, 
    # count paths of length 5 from a to b. If any such path exists with all intermediate
    # vertices distinct from a and b and from each other, we have a C6.
    
    # Better approach: use the fact that C6-free means no two non-adjacent vertices 
    # have a common neighbor at distance 2 that can form the cycle... 
    
    # Let me use a direct but optimized approach:
    # For each unordered pair {u,v} that is NOT an edge, check if there exists
    # a path u-w1-w2-w3-w4-v where all wi are distinct and not u,v.
    
    # Even better: check all 6-tuples? Too slow.
    
    # Use: for each edge (a,b), for each neighbor c of a (c != b), for each neighbor d of b (d != a),
    # if c != d and c != a, d != b, and c is not adjacent to b (wait, that's not right either)
    
    # Let me think more carefully. A C6: v0-v1-v2-v3-v4-v5-v0.
    # Pick an edge v0-v1. Then v5 is a neighbor of v0 (v5 != v1), v2 is a neighbor of v1 (v2 != v0).
    # Then v4 is a neighbor of v5 (v4 != v0), v3 is a neighbor of v2 (v3 != v1).
    # And v3-v4 must be an edge, v3 != v4, and all six vertices distinct.
    
    # For each edge (v0,v1):
    #   for each v5 in N(v0) \ {v1}:
    #     for each v2 in N(v1) \ {v0}:
    #       if v5 == v2: continue
    #       for each v4 in N(v5) \ {v0}:
    #         if v4 == v2: continue
    #         for each v3 in N(v2) \ {v1}:
    #           if v3 == v4: continue
    #           if v3 == v0 or v3 == v1 or v3 == v5: continue
    #           if v4 == v0 or v4 == v1 or v4 == v5: continue
    #           if adj[v3][v4]: return False  # Found C6
    
    # This is O(m * d^4) which for d~5, m~105 is about 105*625 = 65k, manageable.
    
    adj = adj_matrix
    for v0 in range(n):
        for v1 in range(v0+1, n):
            if not adj[v0][v1]:
                continue
            for v5 in range(n):
                if v5 == v0 or v5 == v1:
                    continue
                if not adj[v0][v5]:
                    continue
                for v2 in range(n):
                    if v2 == v0 or v2 == v1 or v2 == v5:
                        continue
                    if not adj[v1][v2]:
                        continue
                    for v4 in range(n):
                        if v4 == v0 or v4 == v5 or v4 == v2:
                            continue
                        if not adj[v5][v4]:
                            continue
                        for v3 in range(n):
                            if v3 == v0 or v3 == v1 or v3 == v2 or v3 == v5 or v3 == v4:
                                continue
                            if not adj[v2][v3]:
                                continue
                            if adj[v3][v4]:
                                return False
    return True

def _count_edges(adj_matrix, n=40):
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i][j]:
                count += 1
    return count

def _add_edge(adj_matrix, u, v, n=40):
    adj_matrix[u][v] = True
    adj_matrix[v][u] = True

def _remove_edge(adj_matrix, u, v, n=40):
    adj_matrix[u][v] = False
    adj_matrix[v][u] = False

def _get_edges(adj_matrix, n=40):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i][j]:
                edges.append([i, j])
    return edges

def _random_greedy(n=40, seed=None, max_time=110.0):
    """Randomized greedy construction: start empty, add edges in random order if C6-free."""
    if seed is not None:
        random.seed(seed)
    
    start_time = time.time()
    adj = [[False]*n for _ in range(n)]
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() - start_time > max_time:
            break
        _add_edge(adj, u, v, n)
        if not _is_c6_free(adj, n):
            _remove_edge(adj, u, v, n)
    
    return adj, _count_edges(adj, n)

def _local_search(adj, n=40, max_time=105.0):
    """Try to add more edges via local search."""
    start_time = time.time()
    current_edges = _count_edges(adj, n)
    
    while time.time() - start_time < max_time:
        improved = False
        # Try adding each non-edge
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() - start_time > max_time:
                return adj, current_edges
            if adj[u][v]:
                continue
            _add_edge(adj, u, v, n)
            if _is_c6_free(adj, n):
                current_edges += 1
                improved = True
            else:
                _remove_edge(adj, u, v, n)
        
        if not improved:
            break
    
    return adj, current_edges

def _swap_local_search(adj, n=40, max_time=105.0):
    """Try edge swaps: remove one edge, add another, to increase edge count while staying C6-free."""
    start_time = time.time()
    current_edges = _count_edges(adj, n)
    best_adj = [row[:] for row in adj]
    best_count = current_edges
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Get current edges
        cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
        
        random.shuffle(cur_edges)
        random.shuffle(non_edges)
        
        for rem_u, rem_v in cur_edges:
            if time.time() - start_time > max_time:
                break
            _remove_edge(adj, rem_u, rem_v, n)
            
            for add_u, add_v in non_edges:
                if time.time() - start_time > max_time:
                    break
                if adj[add_u][add_v]:
                    continue
                _add_edge(adj, add_u, add_v, n)
                if _is_c6_free(adj, n):
                    # Check if this improves connectivity or structure
                    new_count = _count_edges(adj, n)
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = [row[:] for row in adj]
                        improved = True
                    else:
                        # Keep the swap even if same count, might help future moves
                        pass
                else:
                    _remove_edge(adj, add_u, add_v, n)
            
            # Restore removed edge
            _add_edge(adj, rem_u, rem_v, n)
        
        if not improved and time.time() - start_time < max_time:
            # Try random restarts within local search
            break
    
    if best_count > current_edges:
        return best_adj, best_count
    return adj, current_edges

def _known_good_structure(n=40):
    """Try to build a good initial structure based on known extremal graph constructions."""
    adj = [[False]*n for _ in range(n)]
    
    # Start with 8 copies of K5 (100 edges, C6-free)
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a+1, 5):
                adj[base+a][base+b] = True
                adj[base+b][base+a] = True
    
    return adj

def _try_complete_graphs_union(n=40, sizes=None):
    """Try unions of complete graphs of various sizes summing to n."""
    if sizes is None:
        # Try various partitions of 40 into parts <= 5 (since K6 contains C6)
        # Actually K6 does contain C6, so max part size is 5
        # Partitions of 40 into parts of size at most 5:
        # 8x5=40, 7x5+5=40 (same), etc.
        # The best is likely 8x5 giving 8*C(5,2)=8*10=100 edges
        sizes = [5,5,5,5,5,5,5,5]
    
    adj = [[False]*n for _ in range(n)]
    idx = 0
    for size in sizes:
        for i in range(size):
            for j in range(i+1, size):
                adj[idx+i][idx+j] = True
                adj[idx+j][idx+i] = True
        idx += size
    
    return adj, _count_edges(adj, n)

def construct():
    """Construct a C6-free graph on 40 vertices maximizing edges."""
    start_time = time.time()
    best_edges = 0
    best_adj = None
    
    n = 40
    
    # Phase 1: Get a good starting point
    # Start with 8xK5 structure
    adj = _known_good_structure(n)
    best_adj = [row[:] for row in adj]
    best_edges = _count_edges(adj, n)
    
    # Phase 2: Randomized greedy from random orderings
    for attempt in range(20):
        if time.time() - start_time > DEADLINE - 20:
            break
        adj_g, count_g = _random_greedy(n, seed=42 + attempt, max_time=min(3.0, DEADLINE - (time.time() - start_time)))
        if count_g > best_edges:
            best_edges = count_g
            best_adj = [row[:] for row in adj_g]
    
    # Phase 3: Local search on best structure
    if time.time() - start_time < DEADLINE - 15:
        adj_ls, count_ls = _local_search(best_adj, n, max_time=min(10.0, DEADLINE - (time.time() - start_time)))
        if count_ls > best_edges:
            best_edges = count_ls
            best_adj = adj_ls
    
    # Phase 4: More aggressive local search with multiple restarts
    for attempt in range(5):
        if time.time() - start_time > DEADLINE - 10:
            break
        # Perturb the best solution
        adj_pert = [row[:] for row in best_adj]
        # Remove a few random edges and re-add
        cur_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj_pert[i][j]]
        random.shuffle(cur_edges_list)
        for _ in range(min(3, len(cur_edges_list))):
            u, v = cur_edges_list.pop()
            _remove_edge(adj_pert, u, v, n)
        
        adj_ls, count_ls = _local_search(adj_pert, n, max_time=min(8.0, DEADLINE - (time.time() - start_time)))
        if count_ls > best_edges:
            best_edges = count_ls
            best_adj = adj_ls
    
    # Phase 5: Swap-based local search
    if time.time() - start_time < DEADLINE - 5:
        adj_sw, count_sw = _swap_local_search(best_adj, n, max_time=min(10.0, DEADLINE - (time.time() - start_time)))
        if count_sw > best_edges:
            best_edges = count_sw
            best_adj = adj_sw
    
    # Final verification
    if best_adj is None:
        # Fallback: 8xK5
        adj = _known_good_structure(n)
        return _get_edges(adj, n)
    
    # Verify C6-free
    if not _is_c6_free(best_adj, n):
        # Shouldn't happen, but fallback
        adj = _known_good_structure(n)
        return _get_edges(adj, n)
    
    return _get_edges(best_adj, n)

# EVOLVE-BLOCK-END
