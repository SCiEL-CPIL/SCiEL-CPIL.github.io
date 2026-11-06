"""Triangle-free 4-chromatic graphs on 16 vertices via local search with swaps."""
# EVOLVE-BLOCK-START
import random
import time

def _build_grotzsch_edges():
    """Return the edge list of the Grötzsch graph (Mycielski of C5) on vertices 0..10."""
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[10, 5 + u] for u in range(5)]
    return edges

def _make_bitmask_adj(edge_list, n=16):
    """Convert edge list to list of integer bitmasks representing adjacency."""
    adj = [0] * n
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_valid_triangle_free(adj):
    """Verify that adjacency bitmasks define a simple triangle-free graph."""
    n = len(adj)
    for u in range(n):
        if adj[u] & (1 << u):
            return False
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            mask ^= lb
            if adj[v] & adj[u] & ~(1 << u) & ~(1 << v):
                return False
    return True

def _is_3_colorable(adj, n=16):
    """
    Check if the graph is 3-colorable using backtracking.
    Returns True if 3-colorable, False otherwise.
    """
    colors = [-1] * n
    
    def dfs(u):
        if u == n:
            return True
        # Try colors 0, 1, 2
        for c in range(3):
            # Check if color c is available for u
            ok = True
            mask = adj[u]
            while mask:
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                mask ^= lb
                if colors[v] == c:
                    ok = False
                    break
            if ok:
                colors[u] = c
                if dfs(u + 1):
                    return True
                colors[u] = -1
        return False
    
    # Order vertices by degree (descending) to prune faster
    # But for n=16, simple order is often fine. Let's stick to 0..n-1 for simplicity and speed of implementation.
    return dfs(0)

def _local_search(start_adj, deadline, seed):
    """
    Perform local search to maximize edges while maintaining triangle-freeness and 4-chromaticity.
    """
    n = 16
    rng = random.Random(seed)
    adj = list(start_adj)
    current_edges = sum(bin(a).count('1') for a in adj) // 2
    
    # Precompute all potential edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_adj = list(adj)
    best_edges = current_edges
    
    # Initial state is 4-chromatic (Grötzsch core), so we know it's valid.
    # We will perform moves.
    
    # To speed up 3-coloring check, we can cache or just run it.
    # n=16 is small.
    
    while time.time() < deadline:
        move_type = rng.choice(['add', 'swap', 'remove'])
        
        if move_type == 'add':
            # Pick a random non-edge
            u, v = rng.choice(all_edges)
            if adj[u] & (1 << v):
                continue
            # Check triangle-freeness
            if adj[u] & adj[v]:
                continue
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            new_edges = current_edges + 1
            # Adding an edge to a 4-chromatic graph keeps it 4-chromatic (or higher).
            # So we don't need to check 3-colorability!
            # This is a huge speedup.
            current_edges = new_edges
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = list(adj)
                
        elif move_type == 'remove':
            # Pick a random edge
            # To pick a random edge, pick a random u, then a random neighbor v
            u = rng.randint(0, n - 1)
            mask = adj[u]
            if not mask:
                continue
            # Count bits
            count = bin(mask).count('1')
            k = rng.randint(0, count - 1)
            # Find k-th set bit
            temp_mask = mask
            v = -1
            for _ in range(k + 1):
                lb = temp_mask & (-temp_mask)
                temp_mask ^= lb
                v = lb.bit_length() - 1
            
            # Remove edge (u, v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            current_edges -= 1
            # Must check if graph is still 4-chromatic
            if not _is_3_colorable(adj, n):
                # It is 4-chromatic, so valid
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = list(adj)
            else:
                # It became 3-colorable, revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                
        elif move_type == 'swap':
            # Remove one edge, add another
            # Pick edge to remove
            u1 = rng.randint(0, n - 1)
            mask1 = adj[u1]
            if not mask1:
                continue
            count1 = bin(mask1).count('1')
            k1 = rng.randint(0, count1 - 1)
            temp_mask = mask1
            v1 = -1
            for _ in range(k1 + 1):
                lb = temp_mask & (-temp_mask)
                temp_mask ^= lb
                v1 = lb.bit_length() - 1
            
            # Pick edge to add
            u2, v2 = rng.choice(all_edges)
            if adj[u2] & (1 << v2):
                continue
            
            # Tentatively remove (u1, v1)
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            
            # Check if we can add (u2, v2) without triangle
            # Note: u2, v2 might be connected to u1, v1
            if adj[u2] & adj[v2]:
                # Revert
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                continue
            
            # Tentatively add (u2, v2)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            # Check 3-colorability
            if not _is_3_colorable(adj, n):
                # Valid
                # Edge count stays same, but structure changed
                # We don't update best_edges because count is same, but maybe this opens up more adds later?
                # For now, we accept the move to explore state space.
                pass
            else:
                # Revert
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                
        # Occasionally, try to greedily add any possible edges
        if rng.random() < 0.1:
            improved = True
            while improved:
                improved = False
                for u, v in all_edges:
                    if not (adj[u] & (1 << v)) and not (adj[u] & adj[v]):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_edges += 1
                        improved = True
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = list(adj)

    # Convert best_adj to edge list
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result_edges.append([u, v])
    return result_edges, best_edges

def construct():
    """
    Construct a triangle-free graph on 16 vertices that is not 3-colorable,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 100.0
    
    grotzsch = _build_grotzsch_edges()
    best_edges = list(grotzsch)
    best_count = len(grotzsch)
    
    seed = 42
    while time.time() < deadline - 10.0:
        # Start from Grötzsch graph
        adj = _make_bitmask_adj(grotzsch, 16)
        edges, count = _local_search(adj, deadline, seed)
        if count > best_count:
            best_count = count
            best_edges = edges
        seed += 1
        if best_count >= 50: # Heuristic upper bound guess, stop early if we hit a very high number
            break

    # Final validation
    if not _is_valid_triangle_free(_make_bitmask_adj(best_edges)) or _is_3_colorable(_make_bitmask_adj(best_edges)):
        best_edges = list(grotzsch)
        
    return best_edges
# EVOLVE-BLOCK-END
