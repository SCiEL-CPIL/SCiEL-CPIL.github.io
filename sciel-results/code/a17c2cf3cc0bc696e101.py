"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    n = 40
    
    def time_ok():
        return time.time() < deadline
    
    # Bitset-based adjacency
    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def has_c6(adj):
        """Check if graph has any C6 using bitset operations."""
        # For each vertex u, compute N(u) as a bitmask
        # A C6: u-v-w-x-y-z-u where all distinct
        # Strategy: for each pair (u, x) at distance 3, check if there are two
        # distinct paths of length 3 between them.
        
        # More efficient: for each vertex u
        #   For each neighbor v of u
        #     For each neighbor w of v (w != u)
        #       Compute the set of vertices reachable from w in 2 steps (excluding u, v)
        #       If any such vertex y is a neighbor of u and y != v, and the path is valid...
        
        # Let me use a direct but optimized approach:
        # For each vertex u (0..n-1):
        #   N_u = neighbors of u
        #   For each pair (v, z) in N_u with v < z:
        #     # We want to check if there's a path v-w-x-y-z of length 4
        #     # where w, x, y are all distinct and not in {u, v, z}
        #     # and all 6 vertices u,v,w,x,y,z are distinct
        #     
        #     # N(v) \ {u}: neighbors of v except u
        #     # N(z) \ {u}: neighbors of z except u
        #     # We need w in N(v)\{u}, y in N(z)\{u}, w != y, w != z, y != v
        #     # And x in N(w)\{u,v} intersect N(y)\{u,z}
        #     # With x not in {u, v, w, y, z}
        
        for u in range(n):
            N_u = adj[u]
            if N_u == 0:
                continue
            # Get list of neighbors of u
            neighbors_u = []
            mask = N_u
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                neighbors_u.append(v)
                mask ^= bit
            
            for i in range(len(neighbors_u)):
                for j in range(i + 1, len(neighbors_u)):
                    v = neighbors_u[i]
                    z = neighbors_u[j]
                    # N(v) excluding u
                    Nv_excl_u = adj[v] & ~(1 << u)
                    # N(z) excluding u
                    Nz_excl_u = adj[z] & ~(1 << u)
                    
                    # For each w in Nv_excl_u (w != z to ensure distinctness from z)
                    mask_w = Nv_excl_u & ~(1 << z)
                    while mask_w:
                        bit_w = mask_w & (-mask_w)
                        w = bit_w.bit_length() - 1
                        mask_w ^= bit_w
                        
                        if w == u or w == v or w == z:
                            continue
                        
                        # N(w) excluding u, v
                        Nw_excl = adj[w] & ~(1 << u) & ~(1 << v)
                        
                        # For each y in Nz_excl_u (y != w to ensure distinctness)
                        mask_y = Nz_excl_u & ~(1 << w)
                        while mask_y:
                            bit_y = mask_y & (-mask_y)
                            y = bit_y.bit_length() - 1
                            mask_y ^= bit_y
                            
                            if y == u or y == v or y == z or y == w:
                                continue
                            
                            # N(y) excluding u, z
                            Ny_excl = adj[y] & ~(1 << u) & ~(1 << z)
                            
                            # Check if Nw_excl and Ny_excl intersect at some x
                            # x must not be in {u, v, w, y, z}
                            # Nw_excl already excludes u, v
                            # Ny_excl already excludes u, z
                            # We need to also exclude w, y, z from the intersection
                            common = Nw_excl & Ny_excl
                            # Exclude w, y, z
                            common &= ~(1 << w) & ~(1 << y) & ~(1 << z)
                            
                            if common:
                                return True
        
        return False
    
    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) would create a C6."""
        # A new C6 involving edge (u,v) must use this edge.
        # So we need a path of length 5 from u to v not using edge (u,v).
        # i.e., u-a-b-c-d-v where all 6 vertices distinct and a!=v, d!=u, etc.
        
        # Path: u-a-b-c-d-v
        # a in N(u), d in N(v)
        # a != v, d != u
        # b in N(a)\{u}, c in N(d)\{v}
        # b != d, c != a, b != v, c != u
        # b in N(c)\{a} (edge b-c exists)
        # All 6 distinct: u,a,b,c,d,v
        
        Nu = adj[u] & ~(1 << v)  # neighbors of u except v
        Nv = adj[v] & ~(1 << u)  # neighbors of v except u
        
        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a
            
            if a == v:
                continue
            
            Na_excl_u = adj[a] & ~(1 << u)
            
            mask_d = Nv
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d
                
                if d == u or d == a:
                    continue
                
                Nd_excl_v = adj[d] & ~(1 << v)
                
                # b in Na_excl_u, c in Nd_excl_v
                # b != d, c != a, b != v, c != u
                # edge b-c exists
                # b != c
                
                # For each b in Na_excl_u (b != d, b != v):
                mask_b = Na_excl_u & ~(1 << d) & ~(1 << v)
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    
                    if b == u or b == a or b == d or b == v:
                        continue
                    
                    # c in Nd_excl_u (c != a, c != u, c != b)
                    # c must be neighbor of b
                    Nc_candidates = Nd_excl_v & adj[b]
                    Nc_candidates &= ~(1 << a) & ~(1 << u) & ~(1 << b)
                    
                    if Nc_candidates:
                        return True
        
        return False
    
    # Start with 8 disjoint K5s
    edges = []
    for k in range(8):
        edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
    
    adj = make_adj(edges)
    edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    
    # Greedy edge addition
    # Try to add edges in order of some heuristic
    all_possible_edges = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set]
    
    # Sort by some heuristic: prefer edges between vertices in different K5s
    def edge_heuristic(e):
        u, v = e
        # Prefer edges connecting different groups
        group_u = u // 5
        group_v = v // 5
        if group_u != group_v:
            return 0
        else:
            return 1
    
    all_possible_edges.sort(key=edge_heuristic)
    
    # First pass: add all edges that don't create C6
    added_any = True
    while added_any and time_ok():
        added_any = False
        remaining = []
        for u, v in all_possible_edges:
            if (u, v) in edge_set:
                continue
            if would_create_c6(adj, u, v):
                remaining.append((u, v))
            else:
                # Add the edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                edges.append([u, v])
                added_any = True
        all_possible_edges = remaining
    
    # Local search: try to add more edges by removing some
    # or try different orderings
    best_edges = edges[:]
    best_count = len(edges)
    
    # Try multiple restarts with randomization
    for restart in range(50):
        if not time_ok():
            break
        
        # Start from current best or a random subset
        if restart == 0:
            current_edges = best_edges[:]
        else:
            # Start from a random subset of best edges
            current_edges = [e for e in best_edges if random.random() < 0.9]
        
        current_edge_set = set((min(u,v), max(u,v)) for u, v in current_edges)
        current_adj = make_adj(current_edges)
        
        # Greedy addition
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in current_edge_set]
        random.shuffle(remaining)
        
        added_any = True
        while added_any and time_ok():
            added_any = False
            new_remaining = []
            for u, v in remaining:
                if (u, v) in current_edge_set:
                    continue
                if would_create_c6(current_adj, u, v):
                    new_remaining.append((u, v))
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edge_set.add((u, v))
                    current_edges.append([u, v])
                    added_any = True
            remaining = new_remaining
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
    
    return best_edges
# EVOLVE-BLOCK-END
