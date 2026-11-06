import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def is_valid(n, adj):
        """Check K4-free and no independent 6-set."""
        # K4 check
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u) & ~(1 << v)
                    c = common
                    while c:
                        lsb = c & -c
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        if adj[x] & (common & ~(1 << x)):
                            return False
        # Independent 6-set check
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    continue
                for w in range(v + 1, n):
                    if (adj[u] >> w) & 1 or (adj[v] >> w) & 1:
                        continue
                    for x in range(w + 1, n):
                        if (adj[u] >> x) & 1 or (adj[v] >> x) & 1 or (adj[w] >> x) & 1:
                            continue
                        for y in range(x + 1, n):
                            if (adj[u] >> y) & 1 or (adj[v] >> y) & 1 or (adj[w] >> y) & 1 or (adj[x] >> y) & 1:
                                continue
                            for z in range(y + 1, n):
                                if (adj[u] >> z) & 1 or (adj[v] >> z) & 1 or (adj[w] >> z) & 1 or (adj[x] >> z) & 1 or (adj[y] >> z) & 1:
                                    continue
                                return False
        return True

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def make_paley_graph(p):
        """Create a Paley graph of prime order p (p ≡ 1 mod 4)."""
        adj = [0] * p
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def backtrack_construct(n, deadline, seed_graph=None):
        """
        Constraint-based backtracking: build graph vertex by vertex.
        For vertex i, its adjacency to vertices 0..i-1 must not create a K4
        or an independent 6-set. We enumerate valid masks.
        
        Returns adj list or None if no valid graph found within time.
        """
        if seed_graph is not None:
            adj = list(seed_graph)
        else:
            adj = [0] * n
        
        # Precompute for each vertex i: which subsets of {0..i-1} are valid
        # A mask m for vertex i is valid if:
        # 1. No K4: for any two vertices a,b in m that are adjacent, 
        #    they must not share a common neighbor in m that is also adjacent to both
        #    Actually simpler: the induced subgraph on m must be K4-free
        # 2. No independent 6-set: the set of vertices NOT in m (among 0..i-1) 
        #    must not contain an independent set of size 6 in the current graph
        #    But we also need to check that adding i doesn't create an I6 with some 5 from non-m
        
        # This is complex. Let's use a simpler approach:
        # Build incrementally, and at each step, try all possible adjacency patterns
        # and check validity incrementally.
        
        # For efficiency, we'll use randomized restarts with different initial patterns
        # and greedy construction with backtracking on conflicts.
        
        # Actually, let's try a different approach: start from a known good small graph
        # and grow it by adding vertices one at a time, checking constraints.
        
        def can_add_vertex(adj, n, new_vertex, mask):
            """
            Check if adding new_vertex with adjacency mask to previous vertices
            creates a K4 or independent 6-set.
            mask: bits 0..new_vertex-1 indicate which previous vertices are adjacent
            """
            # K4 check: for any edge (a,b) in the previous graph where both a,b are in mask,
            # and a and b share a common neighbor c in mask, then {a,b,c,new_vertex} is K4
            # Wait, that's not quite right. K4 means 4 mutually adjacent vertices.
            # If new_vertex is adjacent to a and b, and a,b,c are mutually adjacent, 
            # and new_vertex is adjacent to c, then {a,b,c,new_vertex} is K4.
            
            # So: for any triangle (a,b,c) in the previous graph where all three are in mask,
            # adding new_vertex adjacent to all three creates a K4.
            
            # Check: does mask contain a triangle?
            # For each pair (a,b) in mask that are adjacent, check if they have a common neighbor in mask
            m = mask
            prev_adj = [adj[i] & ((1 << new_vertex) - 1) for i in range(new_vertex)]
            
            # Check for triangle in mask
            a_mask = m
            while a_mask:
                la = a_mask & -a_mask
                a = la.bit_length() - 1
                a_mask ^= la
                b_mask = m & (a_mask | la)  # vertices after a in mask
                while b_mask:
                    lb = b_mask & -b_mask
                    b = lb.bit_length() - 1
                    b_mask ^= lb
                    # Check if a and b are adjacent
                    if not ((adj[a] >> b) & 1):
                        continue
                    # Check if they share a common neighbor in mask
                    common = adj[a] & adj[b] & m
                    if common:
                        return False  # Would create K4
            
            # Independent 6-set check:
            # The new vertex is NOT adjacent to vertices in non_mask = ~mask & ((1<<new_vertex)-1)
            # An independent 6-set including new_vertex would be: new_vertex + 5 vertices from non_mask
            # that are mutually non-adjacent.
            # So check: does non_mask contain an independent set of size 5?
            
            non_mask = ((1 << new_vertex) - 1) & ~mask
            # Count independent 5-sets in non_mask
            def count_indep_5(mask_sub, adj_list, n_sub):
                """Count independent sets of size 5 in induced subgraph, capped at 1."""
                vertices = []
                m = mask_sub
                while m:
                    lsb = m & -m
                    vertices.append(lsb.bit_length() - 1)
                    m ^= lsb
                
                nv = len(vertices)
                if nv < 5:
                    return 0
                
                for i in range(nv):
                    for j in range(i + 1, nv):
                        if (adj_list[vertices[i]] >> vertices[j]) & 1:
                            continue
                        for k in range(j + 1, nv):
                            if (adj_list[vertices[i]] >> vertices[k]) & 1 or (adj_list[vertices[j]] >> vertices[k]) & 1:
                                continue
                            for l in range(k + 1, nv):
                                if (adj_list[vertices[i]] >> vertices[l]) & 1 or (adj_list[vertices[j]] >> vertices[l]) & 1 or (adj_list[vertices[k]] >> vertices[l]) & 1:
                                    continue
                                for m2 in range(l + 1, nv):
                                    if (adj_list[vertices[i]] >> vertices[m2]) & 1 or (adj_list[vertices[j]] >> vertices[m2]) & 1 or (adj_list[vertices[k]] >> vertices[m2]) & 1 or (adj_list[vertices[l]] >> vertices[m2]) & 1:
                                        continue
                                    return 1
                return 0
            
            if count_indep_5(non_mask, adj, new_vertex) >= 1:
                return False  # Would create independent 6-set
            
            return True
        
        # Start with a small valid graph and grow
        # Use Paley 17 as base if available
        if seed_graph is None:
            # Start with a small random graph
            for i in range(min(10, n)):
                for j in range(i + 1, min(10, n)):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        
        current_n = 10  # We've built 10 vertices
        
        while current_n < n:
            if time.time() > deadline:
                return None
            
            v = current_n
            found = False
            # Try random masks, biased toward balanced density
            attempts = 0
            max_attempts = 500
            while attempts < max_attempts:
                attempts += 1
                # Generate a random mask with density ~0.5
                mask = 0
                for i in range(v):
                    if random.random() < 0.5:
                        mask |= (1 << i)
                
                if can_add_vertex(adj, v, v, mask):
                    # Accept this mask
                    for i in range(v):
                        if (mask >> i) & 1:
                            adj[v] |= (1 << i)
                            adj[i] |= (1 << v)
                    found = True
                    break
            
            if not found:
                # Backtrack: try to modify earlier decisions
                # For simplicity, return None and restart
                return None
            
            current_n += 1
        
        # Final validation
        if is_valid(n, adj):
            return adj
        return None

    best_graph = None
    best_n = 0

    # Strategy 1: Try to find valid graphs for n=36, 35 using backtracking
    for target_n in [36, 35]:
        if time.time() > deadline:
            break
        
        found = False
        max_restarts = 15
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Use different seeds for variety
            if r == 0:
                # Start from Paley 17
                paley17 = make_paley_graph(17)
                adj = paley17 + [0] * (target_n - 17)
                # Fill in the remaining vertices randomly
                for i in range(17, target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                # Now try to grow from 17
                # Actually, let's just use backtracking from scratch with a seed
                result = backtrack_construct(target_n, deadline, seed_graph=adj)
            else:
                # Random start
                adj = [0] * target_n
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                result = backtrack_construct(target_n, deadline, seed_graph=adj)
            
            if result is not None:
                best_graph = result
                best_n = target_n
                found = True
                break
        
        if found:
            break

    # Strategy 2: Fallback to smaller n
    if best_n < 35:
        for target_n in range(34, 10, -1):
            if time.time() > deadline:
                break
            
            found = False
            max_restarts = 20
            for r in range(max_restarts):
                if time.time() > deadline:
                    break
                
                adj = [0] * target_n
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                
                result = backtrack_construct(target_n, deadline, seed_graph=adj)
                
                if result is not None:
                    best_graph = result
                    best_n = target_n
                    found = True
                    break
            
            if found:
                break

    if best_graph is None:
        # Fallback to Paley 17
        adj = make_paley_graph(17)
        return {'n': 17, 'edges': get_edges(adj, 17)}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
