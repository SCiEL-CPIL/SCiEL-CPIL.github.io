import time
import random
import itertools

def _has_independent_set_10(adj_mask, n):
    """
    Checks if there is an independent set of size 10.
    adj_mask[i] is a bitmask of neighbors of i.
    Uses branch and bound with the standard alpha-detection strategy:
    always include the first available vertex (no exclude branch needed).
    """
    
    def dfs(available_mask, size):
        if size >= 10:
            return True
        if available_mask == 0:
            return False
        
        # If not enough vertices remain, prune
        if bin(available_mask).count('1') < 10 - size:
            return False
            
        # Pick the lowest available vertex
        v_bit = available_mask & -available_mask
        v = v_bit.bit_length() - 1
        
        # Include v: remove v and all its neighbors from available
        new_avail = available_mask & ~(adj_mask[v] | v_bit)
        if dfs(new_avail, size + 1):
            return True
            
        # Exclude v: just remove v from available
        new_avail = available_mask & ~v_bit
        if dfs(new_avail, size):
            return True
            
        return False

    return dfs((1 << n) - 1, 0)

def _turan_graph(n, r):
    """
    Generate a Turán graph T(n, r): partition n vertices into r parts
    as evenly as possible, connect all vertices between different parts.
    This is a complete r-partite graph, which is triangle-free when r=2 (bipartite)
    but for r>=3 it contains triangles. So use r=2 for triangle-free.
    """
    adj = [0] * n
    # For triangle-free, we need bipartite. Use Turán T(n, 2)
    mid = n // 2
    for i in range(mid):
        for j in range(mid, n):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    return adj

def _random_maximal_triangle_free(n, seed=None):
    """
    Generate a random maximal triangle-free graph by adding edges in random order.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    adj = [0] * n
    edges_to_add = []
    for i in range(n):
        for j in range(i + 1, n):
            edges_to_add.append((i, j))
    rng.shuffle(edges_to_add)
    
    for u, v in edges_to_add:
        common = adj[u] & adj[v]
        if common == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    return adj

def _local_search_reduce_alpha(adj, n, max_iters=500):
    """
    Try to reduce alpha by adding edges (keeping triangle-free) or removing edges
    and re-adding. Returns improved adj if alpha decreased.
    """
    current_alpha_has_10 = _has_independent_set_10(adj, n)
    if not current_alpha_has_10:
        return adj, True
    
    best_adj = adj[:]
    best_has_10 = True
    
    for _ in range(max_iters):
        # Try adding a random non-edge that doesn't create a triangle
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if best_adj[i] & (1 << j):
            continue  # already an edge
        
        # Check if adding (i,j) creates a triangle
        common = best_adj[i] & best_adj[j]
        if common != 0:
            continue
        
        # Try adding the edge
        new_adj = best_adj[:]
        new_adj[i] |= (1 << j)
        new_adj[j] |= (1 << i)
        
        if not _has_independent_set_10(new_adj, n):
            return new_adj, True
        # Keep it if it seems to help (hard to measure without full check)
        # Just accept it and continue searching
    
    return best_adj, not _has_independent_set_10(best_adj, n)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s margin before 150s limit
    
    best_n = 26
    best_edges = [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    
    # Try from largest to smallest
    for target_n in [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27]:
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        max_attempts = 50  # More attempts per n
        
        while attempts < max_attempts and time.time() < deadline and not found:
            attempts += 1
            
            # Strategy 1: Random maximal triangle-free
            adj = _random_maximal_triangle_free(target_n, seed=attempts * 1000 + target_n)
            
            if not _has_independent_set_10(adj, target_n):
                best_n = target_n
                best_edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i] & (1 << j):
                            best_edges.append([i, j])
                found = True
                break
            
            # Strategy 2: Start from bipartite, add random edges
            if attempts % 3 == 1:
                adj = _turan_graph(target_n, 2)
                # Now try to add more edges while keeping triangle-free
                extra_edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if not (adj[i] & (1 << j)):
                            extra_edges.append((i, j))
                random.shuffle(extra_edges)
                for u, v in extra_edges:
                    common = adj[u] & adj[v]
                    if common == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                
                if not _has_independent_set_10(adj, target_n):
                    best_n = target_n
                    best_edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if adj[i] & (1 << j):
                                best_edges.append([i, j])
                    found = True
                    break
            
            # Strategy 3: Random with local search
            if attempts % 3 == 2 and time.time() < deadline - 20:
                adj = _random_maximal_triangle_free(target_n, seed=attempts * 2000 + target_n + 777)
                adj, success = _local_search_reduce_alpha(adj, target_n, max_iters=100)
                if success:
                    best_n = target_n
                    best_edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if adj[i] & (1 << j):
                                best_edges.append([i, j])
                    found = True
                    break
        
        if found:
            break
    
    return {'n': best_n, 'edges': best_edges}
