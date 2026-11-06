import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_graph = None
    best_n = 0
    
    # Helper to check if adding edge (u, v) creates a C5
    # G is represented as a list of sets or bitmasks
    # Using bitmasks for speed
    def creates_c5(adj, u, v):
        # Check if there is a path u-x-y-v
        # x in N(u), y in N(v), (x,y) in E
        # Also x != v, y != u, x != y
        nu = adj[u]
        nv = adj[v]
        
        # Iterate x in N(u)
        # For each x, check if N(x) intersects N(v) \ {u}
        # If yes, path u-x-y-v exists.
        # Note: if x==v, we have edge u-v which is being added, so x!=v is implicit if we check before adding.
        # But we are checking *before* adding. So u and v are not adjacent.
        
        # Bitmask intersection
        # N(x) & N(v) should have bits other than u.
        # Actually, if N(x) & N(v) is non-empty, let y be in the intersection.
        # If y == u, then x is adjacent to u and v. Path u-x-u? No, simple path.
        # We need distinct vertices u, x, y, v.
        # So we need y != u.
        
        x_mask = nu
        while x_mask:
            x_bit = x_mask & -x_mask
            x = x_bit.bit_length() - 1
            x_mask ^= x_bit
            
            # Check if N(x) has a neighbor in N(v) other than u
            # N(x) & N(v)
            common = adj[x] & nv
            # Remove u from common if present
            common &= ~(1 << u)
            if common:
                return True
        
        return False

    def find_c5(adj, n):
        # Returns a list of 5 vertices forming a C5, or None
        # Simple DFS
        # This might be slow if called many times.
        # Optimization: Only call when needed.
        
        # Try to find a cycle of length 5
        # Start from 0, try to build a path of length 5 that closes.
        # Path: v0-v1-v2-v3-v4-v0
        # v0 is fixed. v1, v2, v3, v4 are distinct and not v0.
        
        for v0 in range(n):
            # v1 in N(v0)
            v1_mask = adj[v0]
            while v1_mask:
                v1_bit = v1_mask & -v1_mask
                v1 = v1_bit.bit_length() - 1
                v1_mask ^= v1_bit
                
                # v2 in N(v1) \ {v0}
                v2_mask = adj[v1] & ~(1 << v0)
                while v2_mask:
                    v2_bit = v2_mask & -v2_mask
                    v2 = v2_bit.bit_length() - 1
                    v2_mask ^= v2_bit
                    
                    # v3 in N(v2) \ {v0, v1}
                    v3_mask = adj[v2] & ~(1 << v0) & ~(1 << v1)
                    while v3_mask:
                        v3_bit = v3_mask & -v3_mask
                        v3 = v3_bit.bit_length() - 1
                        v3_mask ^= v3_bit
                        
                        # v4 in N(v3) \ {v0, v1, v2}
                        # And v4 must be in N(v0)
                        v4_mask = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & adj[v0]
                        if v4_mask:
                            v4_bit = v4_mask & -v4_mask
                            v4 = v4_bit.bit_length() - 1
                            return [v0, v1, v2, v3, v4]
        return None

    def find_independent_set_9(adj, n):
        # Find an independent set of size 9 in G
        # Equivalent to finding a clique of size 9 in complement
        # Complement adjacency: ~adj[i] & ((1<<n)-1) & ~(1<<i)
        
        comp_adj = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp_adj[i] = all_mask ^ adj[i] ^ (1 << i)
        
        # DFS for clique of size 9
        # Use bitsets
        # State: current set of vertices (mask), current size, last vertex
        # To speed up, order vertices by degree in complement?
        
        # Simple recursive search
        # We want to find ANY independent set of size 9.
        
        def dfs(mask, size, last):
            if size == 9:
                return True
            # Remaining vertices must be able to form a clique of size 9-size
            # Candidates are neighbors of 'last' in complement, within 'mask'
            # Actually, we can just pick any vertex from mask that is connected to all previously picked?
            # No, independent set in G means all pairwise non-adjacent in G.
            # So in complement, they must be pairwise adjacent.
            # So we are looking for a clique in comp_adj.
            
            # Candidates: vertices in mask that are adjacent to 'last' in comp_adj?
            # No, if we build the set incrementally, all new vertices must be adjacent to ALL previous vertices.
            # So maintain a bitmask of valid candidates.
            
            # Let's restructure:
            # candidates: bitmask of vertices that are adjacent to all vertices in the current clique.
            
        # Better approach:
        # candidates mask
        # start with all vertices
        # pick a vertex, intersect candidates with its neighbors in comp.
        
        def find_clique(candidates, depth):
            if depth == 9:
                return True
            # If not enough candidates, fail
            if candidates.bit_count() < depth:
                return False
            
            # Pick a vertex from candidates
            # Heuristic: pick one with high degree in subgraph? Or just first one.
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            
            # Next candidates: neighbors of v in comp, within current candidates
            new_candidates = candidates & comp_adj[v] & ~v_bit
            
            if find_clique(new_candidates, depth + 1):
                return True
            
            # Remove v and try next
            candidates ^= v_bit
            return find_clique(candidates, depth)
            
        return find_clique((1<<n)-1, 0)

    def is_valid(adj, n):
        if find_c5(adj, n):
            return False
        if find_independent_set_9(adj, n):
            return False
        return True

    def to_edges(adj, n):
        edges = []
        for u in range(n):
            v_mask = adj[u]
            while v_mask:
                v_bit = v_mask & -v_mask
                v = v_bit.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                v_mask ^= v_bit
        return edges

    # Main search loop
    for n in range(35, 28, -1):
        if time.time() > deadline - 5.0:
            break
            
        found = False
        # Try multiple random restarts
        for restart in range(100):
            if time.time() > deadline - 2.0:
                break
                
            # Initialize random graph
            # Start with p=0.5
            adj = [0] * n
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
            # Phase 1: Remove C5s
            # Limit iterations to avoid infinite loops if stuck
            max_c5_removals = 1000
            for _ in range(max_c5_removals):
                if time.time() > deadline - 2.0:
                    break
                c5 = find_c5(adj, n)
                if not c5:
                    break
                # Remove a random edge from c5
                # Edges are (c5[0],c5[1]), (c5[1],c5[2]), ..., (c5[4],c5[0])
                e_idx = random.randint(0, 4)
                u = c5[e_idx]
                v = c5[(e_idx+1)%5]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
            if time.time() > deadline - 2.0:
                continue

            # Check if C5 free
            if find_c5(adj, n):
                continue
                
            # Phase 2: Reduce Alpha
            # Try to add edges to break independent sets of size 9
            max_adds = 100
            for _ in range(max_adds):
                if time.time() > deadline - 2.0:
                    break
                
                # Check if alpha < 9
                if not find_independent_set_9(adj, n):
                    break # Success!
                
                # Find an independent set of size 9 to get candidate pairs
                # The find_independent_set_9 above just returns bool.
                # We need the set to pick a pair.
                # Let's modify find_independent_set_9 to return the set or a pair.
                
                # Actually, we can just pick a random pair of non-adjacent vertices
                # and check if adding it is safe and if it helps.
                # But we need to ensure we are breaking the specific independent set.
                
                # Let's get the independent set.
                # Re-implementing to return the set is better.
                
                # For now, let's just try to add random safe edges.
                # If we add enough safe edges, alpha will drop.
                
                added = False
                # Get list of non-edges
                non_edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if not (adj[u] & (1 << v)):
                            non_edges.append((u, v))
                
                random.shuffle(non_edges)
                
                for u, v in non_edges:
                    if creates_c5(adj, u, v):
                        continue
                    # Safe to add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
                
                if not added:
                    break # Stuck
                    
            # Final check
            if not find_c5(adj, n) and not find_independent_set_9(adj, n):
                edges = to_edges(adj, n)
                return {'n': n, 'edges': edges}
                
    # Fallback: Return the seed graph (n=28)
    return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}

