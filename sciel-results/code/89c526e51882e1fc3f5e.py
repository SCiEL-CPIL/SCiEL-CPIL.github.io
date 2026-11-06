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

def _is_3_colorable(adj):
    """
    Check if the graph defined by adj bitmasks is 3-colorable.
    Returns True if 3-colorable, False otherwise.
    Uses vertex ordering by degree for faster pruning.
    """
    n = len(adj)
    # Calculate degrees
    degrees = [0] * n
    for i in range(n):
        degrees[i] = bin(adj[i]).count('1')
    
    # Order vertices by descending degree
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    colors = [-1] * n
    
    def backtrack(v_idx):
        if v_idx == n:
            return True
        
        v = order[v_idx]
        mask = 0
        neighbors = adj[v]
        while neighbors:
            lb = neighbors & (-neighbors)
            u = lb.bit_length() - 1
            neighbors ^= lb
            if colors[u] != -1:
                mask |= (1 << colors[u])
        
        available = ~mask & 0b111
        if available == 0:
            return False
        
        for c in range(3):
            if available & (1 << c):
                colors[v] = c
                if backtrack(v_idx + 1):
                    return True
                colors[v] = -1
        return False
    
    return backtrack(0)

def _local_search(seed, deadline):
    n = 16
    rng = random.Random(seed)
    
    # Initialize with Grotzsch
    grotzsch = _build_grotzsch_edges()
    adj = _make_bitmask_adj(grotzsch, n)
    edge_set = set((min(u, v), max(u, v)) for u, v in grotzsch)
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    non_edges = [e for e in all_edges if e not in edge_set]
    
    best_count = len(edge_set)
    best_adj = adj[:]
    
    failed_swaps = 0
    
    while time.time() < deadline:
        # 1. Try to add a random non-edge
        added = False
        if non_edges:
            idx = rng.randrange(len(non_edges))
            u, v = non_edges[idx]
            # Check triangle-free
            if adj[u] & adj[v] == 0:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                non_edges.pop(idx)
                added = True
                if len(edge_set) > best_count:
                    best_count = len(edge_set)
                    best_adj = adj[:]
                failed_swaps = 0 # Reset counter on success
        
        # 2. If no edge added, try a swap (remove one, add one)
        if not added:
            failed_swaps += 1
            
            # If stuck, try a kick (perturbation)
            if failed_swaps > 100:
                # Save state
                saved_adj = adj[:]
                saved_edge_set = set(edge_set)
                saved_non_edges = non_edges[:]
                
                # Remove 3 random edges
                kick_edges = rng.sample(list(edge_set), k=3) if len(edge_set) >= 3 else list(edge_set)
                for e_out in kick_edges:
                    u_out, v_out = e_out
                    adj[u_out] &= ~(1 << v_out)
                    adj[v_out] &= ~(1 << u_out)
                    edge_set.remove(e_out)
                    # Update non_edges? It's messy. Let's just rebuild or ignore non_edges list for kick phase
                    # Actually, we can just try to add edges from all possible pairs
                    # But we need to be efficient.
                    # Let's just try to add any valid non-edge from the full set
                    pass
                
                # Try to greedily add edges
                for _ in range(50): # Limit kick recovery time
                    if time.time() > deadline:
                        break
                    # Pick a random pair
                    while True:
                        u = rng.randrange(n)
                        v = rng.randrange(n)
                        if u == v: continue
                        e_in = (min(u, v), max(u, v))
                        if e_in not in edge_set:
                            break
                    if adj[u] & adj[v] == 0:
                        # Check 3-colorability
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edge_set.add(e_in)
                        if not _is_3_colorable(adj):
                            # Success, keep it
                            break
                        else:
                            # Revert
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            edge_set.remove(e_in)
                
                # If kick resulted in better graph, keep it. Else revert.
                if len(edge_set) > best_count:
                    best_count = len(edge_set)
                    best_adj = adj[:]
                    # Rebuild non_edges for future?
                    # We can just clear non_edges and let it refill? 
                    # No, non_edges is used for random selection.
                    # Let's rebuild non_edges
                    non_edges = [e for e in all_edges if e not in edge_set]
                    failed_swaps = 0
                else:
                    # Revert kick
                    adj = saved_adj
                    edge_set = saved_edge_set
                    non_edges = saved_non_edges
                    failed_swaps = 0 # Reset to try normal swaps again
            
            # Normal swap logic
            else:
                if not edge_set:
                    break
                e_out = rng.choice(list(edge_set))
                u_out, v_out = e_out
                
                # Remove it
                adj[u_out] &= ~(1 << v_out)
                adj[v_out] &= ~(1 << u_out)
                edge_set.remove(e_out)
                
                swap_success = False
                # Try a few random swaps
                for _ in range(20): # Reduced from 50 to allow more kicks if needed? Or keep 50?
                    if time.time() > deadline:
                        break
                    while True:
                        u = rng.randrange(n)
                        v = rng.randrange(n)
                        if u == v:
                            continue
                        e_in = (min(u, v), max(u, v))
                        if e_in not in edge_set:
                            break
                    
                    if adj[u] & adj[v] == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edge_set.add(e_in)
                        
                        if not _is_3_colorable(adj):
                            swap_success = True
                            if len(edge_set) > best_count:
                                best_count = len(edge_set)
                                best_adj = adj[:]
                            break
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            edge_set.remove(e_in)
                
                if not swap_success:
                    adj[u_out] |= (1 << v_out)
                    adj[v_out] |= (1 << u_out)
                    edge_set.add(e_out)
                    # Update non_edges? 
                    # e_out is back in edge_set, so it should not be in non_edges.
                    # It was popped? No, we didn't pop it from non_edges when we removed it initially?
                    # Wait, in the "add" phase, we pop from non_edges.
                    # In the "swap" phase, we remove from edge_set, but don't add to non_edges.
                    # So non_edges is stale.
                    # This is a bug in the original logic if non_edges is used for selection.
                    # The original code uses `non_edges` for the "add" phase.
                    # If we don't update `non_edges` after swaps, we might pick an edge that is already in `edge_set`?
                    # No, we check `if e_in not in edge_set`.
                    # But `non_edges` might contain edges that ARE in `edge_set`?
                    # No, initially `non_edges` is complement.
                    # If we add an edge, we pop it from `non_edges`.
                    # If we remove an edge in swap, we don't add it to `non_edges`.
                    # So `non_edges` shrinks over time?
                    # If we revert a swap, the edge is back in `edge_set`, but it's NOT in `non_edges`.
                    # So we will never try to re-add it via the "add" phase.
                    # This is acceptable, as we can still find it via "swap" phase?
                    # But "swap" phase picks random `u, v` from all pairs, not from `non_edges`.
                    # So it's fine. `non_edges` is just a list of candidates for the "add" phase.
                    # Stale `non_edges` (missing some valid non-edges) just means we miss some "add" opportunities.
                    # But "swap" covers the space.
        
    # Convert best_adj to edge list
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result_edges.append([u, v])
    
    return result_edges, len(result_edges)

def construct():
    """
    Construct a triangle-free graph on 16 vertices that is not 3-colorable,
    maximizing the number of edges. Uses local search with swaps.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # Run multiple instances of local search with different seeds
    seed = 0
    while time.time() < deadline - 10.0:
        edges, count = _local_search(seed, deadline)
        if count > best_count:
            best_count = count
            best_edges = edges
        seed += 1
        if best_count >= 55:
            break

    # Final validation
    if best_edges:
        if not _is_valid_triangle_free(_make_bitmask_adj(best_edges)):
            grotzsch = _build_grotzsch_edges()
            best_edges = list(grotzsch)
    else:
        best_edges = _build_grotzsch_edges()
    
    return best_edges
# EVOLVE-BLOCK-END
