"""Triangle-free 4-chromatic graphs on 16 vertices via cost-based greedy and local search."""
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

def _greedy_complete_cost(adj, edge_set, core_edges, n, rng, deadline):
    """
    Greedily add edges using a cost function to minimize future blocking.
    """
    while time.time() < deadline:
        # Find all addable edges
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in edge_set:
                    continue
                # Check triangle
                if adj[u] & adj[v]:
                    continue
                # Calculate cost: sum of degrees
                cost = bin(adj[u]).count('1') + bin(adj[v]).count('1')
                candidates.append((cost, u, v))
        
        if not candidates:
            break
        
        # Sort by cost
        candidates.sort(key=lambda x: x[0])
        
        # Pick from top K to add randomness
        k = min(5, len(candidates))
        # Use rng to pick from the best k
        # Or just pick the best one deterministically? 
        # Randomization helps escape local optima.
        # Let's pick randomly from the top 10% or top 5.
        top_candidates = candidates[:k]
        _, u, v = rng.choice(top_candidates)
        
        # Add edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
        
    return adj, edge_set

def _local_search(adj, edge_set, core_edges, n, rng, deadline):
    """
    Perform local search by removing non-core edges and re-greedy filling.
    """
    best_count = len(edge_set)
    best_adj = adj[:]
    best_edge_set = set(edge_set)
    
    # Get list of non-core edges
    non_core_edges = [e for e in edge_set if e not in core_edges]
    
    while time.time() < deadline:
        if not non_core_edges:
            break
        
        # Pick a random non-core edge to remove
        e_out = rng.choice(non_core_edges)
        u, v = e_out
        
        # Remove it
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edge_set.remove(e_out)
        
        # Re-greedy fill
        adj_new, edge_set_new = _greedy_complete_cost(
            adj[:], edge_set, core_edges, n, rng, min(time.time() + 0.1, deadline)
        )
        
        if len(edge_set_new) > best_count:
            best_count = len(edge_set_new)
            best_adj = adj_new[:]
            best_edge_set = set(edge_set_new)
            # Update non_core_edges list
            non_core_edges = [e for e in best_edge_set if e not in core_edges]
        else:
            # Revert? 
            # We modified adj and edge_set in place in _greedy_complete_cost? 
            # No, we passed copies? 
            # _greedy_complete_cost takes adj and edge_set. If we pass adj[:] and edge_set, 
            # it modifies the copy. 
            # But we need to revert the removal of e_out to try the next swap.
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add(e_out)
            
    return best_adj, best_edge_set, best_count

def construct():
    """
    Construct a triangle-free graph on 16 vertices that is not 3-colorable,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    grotzsch_edges = _build_grotzsch_edges()
    core_edges = set((min(u, v), max(u, v)) for u, v in grotzsch_edges)
    
    best_edges = list(grotzsch_edges)
    best_count = len(grotzsch_edges)
    
    seed = 42
    while time.time() < deadline - 10.0:
        # Initialize
        adj = _make_bitmask_adj(grotzsch_edges, n)
        edge_set = set(core_edges)
        
        # Greedy phase
        adj, edge_set = _greedy_complete_cost(adj, edge_set, core_edges, n, random.Random(seed), deadline)
        
        count = len(edge_set)
        if count > best_count:
            best_count = count
            best_edges = [[u, v] for u, v in edge_set]
        
        # Local search phase
        if time.time() < deadline - 10.0:
            adj_ls, edge_set_ls, count_ls = _local_search(adj, edge_set, core_edges, n, random.Random(seed), deadline)
            if count_ls > best_count:
                best_count = count_ls
                best_edges = [[u, v] for u, v in edge_set_ls]
        
        seed += 1
        # If we reach a high count, we can stop early?
        # Max possible is 64. 40+ is good.
        if best_count >= 50:
            break

    # Final validation
    if best_edges:
        if not _is_valid_triangle_free(_make_bitmask_adj(best_edges)):
            best_edges = list(grotzsch_edges)
    else:
        best_edges = list(grotzsch_edges)
    
    return best_edges
# EVOLVE-BLOCK-END
