import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Build a C4-free graph on n vertices with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 110  # Safety margin
    
    best_result = None
    best_n = 0
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple random restarts with local search
        for attempt in range(80):
            if time.time() > deadline:
                break
            
            result = _local_search_c4_free(n, required_min_deg, deadline, attempt)
            if result is not None:
                if n > best_n:
                    best_n = n
                    best_result = result
                    # If we found a solution for the current n, we can return it 
                    # unless we want to try higher n (but we are iterating down).
                    # So returning immediately is optimal if we are maximizing n.
                    return result
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a valid small graph
    return {'n': 1, 'edges': []}


def _check_c4_add(adj_sets, n, u, v):
    """
    Check if adding edge (u,v) creates a C4.
    adj_sets: list of sets of neighbors.
    Returns True if C4 would be created.
    Uses O(1) set intersection (C-implemented) instead of Python loops.
    """
    neighbors_u = adj_sets[u]
    neighbors_v = adj_sets[v]
    
    for x in neighbors_u:
        if x == v:
            continue
        # If any neighbor of x (other than u) is also a neighbor of v, C4 exists
        if adj_sets[x] & neighbors_v:
            return True
    return False


def _polarity_seed(n, rng):
    """
    Build a C4-free graph using a polarity-based construction.
    This gives a much better starting point than empty graph + random greedy.
    """
    adj_sets = [set() for _ in range(n)]
    
    # Partition vertices into two groups
    mid = n // 2
    group_a = list(range(mid))
    group_b = list(range(mid, n))
    rng.shuffle(group_a)
    rng.shuffle(group_b)
    
    # Connect each vertex in A to a random subset of B
    # To avoid C4, we use a "polarity" structure:
    # Each vertex in A connects to exactly the same subset of B would create C4s.
    # Instead, use a random bipartite-like structure with controlled density.
    
    # Strategy: create a bipartite graph between A and B where each vertex
    # has degree ~ (n-39)/2, then add intra-group edges carefully.
    
    target_deg_b = max(1, (n - 39) // 2)
    
    for a in group_a:
        # Connect to target_deg_b random vertices in B
        b_candidates = list(group_b)
        rng.shuffle(b_candidates)
        for b in b_candidates[:target_deg_b]:
            adj_sets[a].add(b)
            adj_sets[b].add(a)
    
    # Add some intra-A and intra-B edges to boost degrees
    # Intra-A edges: random matching-like
    a_remaining = list(group_a)
    rng.shuffle(a_remaining)
    for i in range(0, len(a_remaining) - 1, 2):
        u, v = a_remaining[i], a_remaining[i+1]
        if not _check_c4_add(adj_sets, n, u, v):
            adj_sets[u].add(v)
            adj_sets[v].add(u)
    
    b_remaining = list(group_b)
    rng.shuffle(b_remaining)
    for i in range(0, len(b_remaining) - 1, 2):
        u, v = b_remaining[i], b_remaining[i+1]
        if not _check_c4_add(adj_sets, n, u, v):
            adj_sets[u].add(v)
            adj_sets[v].add(u)
    
    return adj_sets


def _local_search_c4_free(n, min_deg_req, deadline, seed_offset):
    """Construct a C4-free graph on n vertices with min degree >= min_deg_req using local search."""
    rng = random.Random(seed_offset * 7919 + 31)
    
    # Use polarity seed for better initialization
    adj_sets = _polarity_seed(n, rng)
    
    # Phase 1: Greedy fill - try to add remaining edges in random order
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_edges)
    
    for (u, v) in all_edges:
        if time.time() > deadline:
            break
        if u not in adj_sets[v]:
            if not _check_c4_add(adj_sets, n, u, v):
                adj_sets[u].add(v)
                adj_sets[v].add(u)
            
    # Phase 2: Local search to improve minimum degree
    max_iters = 2000
    
    for _iter in range(max_iters):
        if time.time() > deadline:
            break
        
        # Calculate degrees
        degrees = [len(s) for s in adj_sets]
        min_deg = min(degrees)
        
        if min_deg >= min_deg_req:
            # Convert to edge list
            edges = []
            for u in range(n):
                for v in adj_sets[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        # Find a vertex with minimum degree (randomly among them)
        min_verts = [i for i, d in enumerate(degrees) if d == min_deg]
        v = rng.choice(min_verts)
        
        # Try to add an edge from v to a non-neighbor
        candidates = [u for u in range(n) if u != v and u not in adj_sets[v]]
        rng.shuffle(candidates)
        
        added = False
        for u in candidates:
            if time.time() > deadline:
                break
            if not _check_c4_add(adj_sets, n, u, v):
                adj_sets[u].add(v)
                adj_sets[v].add(u)
                added = True
                break
        
        if not added:
            # Cannot add edge to v directly.
            # Try a swap: Remove a random edge (a, b), then try to add edges to v.
            # Only accept if min degree increases.
            
            all_current_edges = []
            for u in range(n):
                for w in adj_sets[u]:
                    if u < w:
                        all_current_edges.append((u, w))
                        
            if not all_current_edges:
                break
                
            a, b = rng.choice(all_current_edges)
            
            # Remove edge (a, b)
            adj_sets[a].remove(b)
            adj_sets[b].remove(a)
            
            # Try to add edges to v
            added_edges = []
            candidates = [u for u in range(n) if u != v and u not in adj_sets[v]]
            rng.shuffle(candidates)
            
            for u in candidates:
                if time.time() > deadline:
                    break
                if not _check_c4_add(adj_sets, n, u, v):
                    adj_sets[u].add(v)
                    adj_sets[v].add(u)
                    added_edges.append(u)
                    # Limit additions to keep search space small
                    if len(added_edges) >= 3:
                        break
                        
            new_degrees = [len(s) for s in adj_sets]
            new_min = min(new_degrees)
            
            if new_min >= min_deg:
                # Accept (allow equal to explore)
                pass
            else:
                # Revert
                adj_sets[a].add(b)
                adj_sets[b].add(a)
                for u in added_edges:
                    adj_sets[u].remove(v)
                    adj_sets[v].remove(u)
                
                # Try a different vertex next iteration
                continue

    # Final check
    degrees = [len(s) for s in adj_sets]
    if min(degrees) >= min_deg_req:
        edges = []
        for u in range(n):
            for v in adj_sets[u]:
                if u < v:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    return None

# EVOLVE-BLOCK-END
