import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n = 16
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    # Start with a dense random graph
    random.seed(42)
    edges = set()
    for e in all_edges:
        if random.random() < 0.85:
            edges.add(e)

    # Adjacency as bitmasks
    adj = [0] * n

    def rebuild_adj():
        for i in range(n):
            adj[i] = 0
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    rebuild_adj()

    def find_q3_global():
        """Find any Q3 in the graph. Returns (A, B) or None."""
        # Iterate through all 8-subsets
        for combo in itertools.combinations(range(n), 8):
            vlist = list(combo)
            # Precompute mask for the set
            mask_set = 0
            for v in vlist:
                mask_set |= 1 << v
            
            # Check all bipartitions of 4 and 4
            # We can iterate i, j, k, l for A, and B is the rest
            for i in range(8):
                for j in range(i + 1, 8):
                    for k in range(j + 1, 8):
                        for l in range(k + 1, 8):
                            A = [vlist[i], vlist[j], vlist[k], vlist[l]]
                            B = [x for x in vlist if x not in A]
                            
                            mask_A = 0
                            for a in A: mask_A |= 1 << a
                            mask_B = 0
                            for b in B: mask_B |= 1 << b
                            
                            # Check if A and B form a Q3
                            # Condition: Each vertex in A has exactly 3 neighbors in B
                            # And each vertex in B has exactly 3 neighbors in A
                            valid = True
                            for a in A:
                                if (adj[a] & mask_B).bit_count() != 3:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            for b in B:
                                if (adj[b] & mask_A).bit_count() != 3:
                                    valid = False
                                    break
                            if valid:
                                return A, B
        return None

    def check_add_edge(u, v):
        """Check if adding (u,v) creates a Q3. Assumes (u,v) is NOT in edges yet."""
        # u and v must be in opposite parts of the Q3
        # Let A contain u, B contain v
        # A = {u, a1, a2, a3}, B = {v, b1, b2, b3}
        # Remaining vertices: V_rem = V \ {u, v}
        
        # Precompute masks for u and v neighbors in V_rem?
        # We iterate over A' (3 from V_rem) and B' (3 from V_rem \ A')
        
        # Optimization: 
        # u must be connected to all b in B'
        # v must be connected to all a in A'
        
        candidates_A = []
        # We need to choose 3 vertices for A' from V_rem
        # But we can prune: a in A' must be connected to v
        # b in B' must be connected to u
        
        # Let's just iterate. 14 vertices remaining.
        # C(14, 3) = 364 for A'
        # For each A', B' is chosen from remaining 11. C(11, 3) = 165.
        
        # To speed up, precompute list of vertices connected to v (excluding u, and not in A' yet)
        # and connected to u.
        
        # Actually, simple iteration is fine if we use bitmasks.
        
        # Get list of vertices not u or v
        others = [x for x in range(n) if x != u and x != v]
        
        for A_prime in itertools.combinations(others, 3):
            # Check if v is connected to all in A_prime
            mask_A_prime = 0
            for a in A_prime:
                mask_A_prime |= 1 << a
            if (adj[v] & mask_A_prime).bit_count() != 3:
                continue
            
            # Now choose B_prime from others \ A_prime
            remaining = [x for x in others if x not in A_prime]
            
            # b in B_prime must be connected to u
            # We can filter remaining first?
            # candidates for B_prime must be neighbors of u
            b_candidates = [b for b in remaining if (adj[u] >> b) & 1]
            
            if len(b_candidates) < 3:
                continue
                
            for B_prime in itertools.combinations(b_candidates, 3):
                mask_B_prime = 0
                for b in B_prime:
                    mask_B_prime |= 1 << b
                
                # Check degrees in the subgraph A_prime U B_prime
                # a in A_prime must have exactly 2 neighbors in B_prime
                # b in B_prime must have exactly 2 neighbors in A_prime
                
                valid = True
                for a in A_prime:
                    if (adj[a] & mask_B_prime).bit_count() != 2:
                        valid = False
                        break
                if not valid:
                    continue
                for b in B_prime:
                    if (adj[b] & mask_A_prime).bit_count() != 2:
                        valid = False
                        break
                if valid:
                    return True
        return False

    # Phase 1: Remove edges to make Q3-free
    while time.time() < deadline - 20:
        found = find_q3_global()
        if not found:
            break
        A, B = found
        # Remove a random edge from the bipartite part A x B
        # Note: All edges between A and B exist? No, only 12 exist.
        # We should remove one of the existing edges.
        existing = []
        for a in A:
            for b in B:
                if (adj[a] >> b) & 1:
                    existing.append((a, b))
        if existing:
            e_to_remove = random.choice(existing)
            edges.remove(e_to_remove)
            adj[e_to_remove[0]] &= ~(1 << e_to_remove[1])
            adj[e_to_remove[1]] &= ~(1 << e_to_remove[0])
        else:
            break

    # Phase 2: Try to add edges
    while time.time() < deadline - 5:
        added = False
        non_edges = [e for e in all_edges if e not in edges]
        random.shuffle(non_edges)
        for (u, v) in non_edges:
            # Check if adding (u,v) creates a Q3
            if not check_add_edge(u, v):
                edges.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added = True
        if not added:
            break

    return [list(e) for e in edges]

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
