import time
import random

def construct():
    """
    Build a C4-free graph on n vertices with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 110  # Safety margin
    
    # We want to maximize n. Try 47 first, then 46, etc.
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple random restarts
        # If we find a solution for n, we return it immediately as we are iterating down.
        # However, 47 might be hard. 46 is likely easier.
        # We should spend most time on 47, then 46.
        
        # Allocate time based on n
        # If n=47, spend 60s. If n=46, spend 40s. If n=45, spend 10s.
        # This is a heuristic.
        
        attempts = 20 if n > 45 else 10
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            result = _solve_c4_free(n, required_min_deg, deadline, attempt)
            if result is not None:
                return result
                
    # Fallback
    return {'n': 1, 'edges': []}

def _solve_c4_free(n, min_deg_req, deadline, seed_offset):
    """
    Solve for a specific n using bitmask local search.
    """
    rng = random.Random(seed_offset * 7919 + 31)
    
    # Initialize with an empty graph
    # adj_mask[i] is a bitmask of neighbors of i
    adj_mask = [0] * n
    
    # Helper to check if adding edge (u, v) creates a C4
    # u, v must not be adjacent already
    def can_add_edge(u, v):
        if (adj_mask[u] >> v) & 1:
            return False # Already adjacent
        
        # Check if there is a path of length 3: u-a-b-v
        # This exists if there is an edge (a,b) where a in N(u) and b in N(v)
        # Note: a cannot be v, b cannot be u.
        # Since u,v not adjacent, v not in N(u) and u not in N(v).
        
        # We need to check if the set of neighbors of N(u) intersects with N(v).
        # Let S_u = union of N(a) for all a in N(u).
        # If S_u & N(v) != 0, then C4 exists.
        
        # Optimization: 
        # Iterate over neighbors of u.
        # For each neighbor a of u, check if any neighbor of a is in N(v).
        
        # Using bitmasks:
        # common = 0
        # mask_u = adj_mask[u]
        # # Iterate through set bits in mask_u
        # m = mask_u
        # while m:
        #     lsb = m & -m
        #     a = lsb.bit_length() - 1
        #     # Check if adj_mask[a] intersects adj_mask[v]
        #     if adj_mask[a] & adj_mask[v]:
        #         return False
        #     m -= lsb
        # return True
        
        # Faster:
        # Precompute "2-step neighborhood" bitmask? 
        # N2(u) = union of N(a) for a in N(u).
        # If we maintain N2, updating it is expensive (O(deg^2)).
        # Given n=47, deg is small, so O(deg) check is fine.
        
        m = adj_mask[u]
        mask_v = adj_mask[v]
        while m:
            lsb = m & -m
            a = lsb.bit_length() - 1
            if adj_mask[a] & mask_v:
                return False
            m -= lsb
        return True

    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    def get_degree(u):
        return bin(adj_mask[u]).count('1')

    def get_min_degree():
        min_d = n
        for u in range(n):
            d = get_degree(u)
            if d < min_d:
                min_d = d
        return min_d

    # Phase 1: Greedy initialization
    # Try to add edges in random order
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if can_add_edge(u, v):
            add_edge(u, v)
            
    # Phase 2: Local Search
    # We want min degree >= min_deg_req
    # Loop until deadline or success
    
    max_iters = 50000
    for _iter in range(max_iters):
        if time.time() > deadline:
            break
            
        # Calculate degrees
        degrees = [get_degree(u) for u in range(n)]
        min_deg = min(degrees)
        
        if min_deg >= min_deg_req:
            # Success
            edges = []
            for u in range(n):
                m = adj_mask[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    m -= lsb
            return {'n': n, 'edges': edges}
        
        # Find vertex with min degree
        min_verts = [i for i, d in enumerate(degrees) if d == min_deg]
        v = rng.choice(min_verts)
        
        # Try to add an edge to v
        candidates = [u for u in range(n) if u != v and not ((adj_mask[v] >> u) & 1)]
        rng.shuffle(candidates)
        
        added = False
        for u in candidates:
            if can_add_edge(u, v):
                add_edge(u, v)
                added = True
                break
        
        if not added:
            # Stuck. Try to remove an edge to free up space.
            # Strategy: Remove an edge (a, b) that is "blocking" v.
            # An edge (a, b) blocks v if a is in N(v) or b is in N(v)?
            # No, (a, b) creates a C4 with (u, v) if a in N(u) and b in N(v) (or vice versa).
            # Since we failed to add (u, v) for all u, it means for every u not in N(v),
            # there exists a in N(u) and b in N(v) such that (a, b) is an edge.
            
            # Simple heuristic: Remove a random edge incident to a high-degree vertex.
            # Or remove an edge (a, b) where a in N(v) and b is a neighbor of some u.
            
            # Let's try removing a random edge from the whole graph.
            # To make it smarter, remove edges that are part of many "potential C4s" with v.
            
            # Identify edges (a, b) such that a in N(v) and b is connected to some non-neighbor u of v.
            # This is complex. Let's just try a few random edge removals.
            
            removed = False
            for _try in range(10):
                if time.time() > deadline:
                    break
                    
                # Pick a random existing edge
                # We need a list of edges or pick random u, v and check if edge exists
                u_rand = rng.randint(0, n - 1)
                v_rand = rng.randint(0, n - 1)
                if u_rand == v_rand:
                    continue
                if (adj_mask[u_rand] >> v_rand) & 1:
                    # Remove edge
                    remove_edge(u_rand, v_rand)
                    removed = True
                    break
            
            if not removed:
                # Graph is empty? Unlikely.
                break
            
            # After removal, try to add edges to v again
            # (Loop will restart, check min degree again)
            
    return None
