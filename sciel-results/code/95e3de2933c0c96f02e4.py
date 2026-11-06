import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n = 16
    # Total possible edges: 16*15/2 = 120
    # We will process edges in a specific order to allow efficient backtracking.
    # Order: By vertex index, then neighbor index.
    # Edge (u, v) with u < v.
    # This order ensures that when we decide on edge (u, v), 
    # all edges incident to vertices < u are already decided.
    # Edges incident to u with vertices < u are decided.
    # Edges incident to u with vertices > u are NOT decided yet.
    # Edges incident to v with vertices < v are NOT decided yet (except those with u).
    
    # Actually, a better order for backtracking is often "lexicographic" or 
    # grouping by the smaller index.
    # Let's stick to standard lexicographic: (0,1), (0,2), ..., (0,15), (1,2), ...
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
            
    m = len(all_edges)
    
    # Precompute which edges are incident to each vertex?
    # Or just maintain a bitmask of the graph.
    # With 120 edges, a Python integer bitmask is perfect.
    
    # Bit i corresponds to all_edges[i]
    
    def get_adj_mask(edge_mask):
        """
        Returns a list of 16 integers, where adj_mask[u] is a bitmask of neighbors of u.
        This is expensive if called every step. 
        Instead, we maintain adj_mask incrementally.
        """
        pass

    # State:
    # edge_mask: int representing current set of edges.
    # adj_masks: list of 16 ints. adj_masks[u] has bit v set if (u,v) is an edge.
    
    # We need a fast check: does the current graph contain Q3?
    # Since we are adding edges one by one (backtracking), we only need to check 
    # if the *newly added* edge completes a Q3.
    
    # A Q3 is an 8-vertex subgraph. 
    # If we add edge (u, v), any new Q3 must contain (u, v).
    # How many Q3s contain a specific edge?
    # Q3 has 12 edges. By symmetry, each edge is in (8! / (12 * 8! / 12)) ... no.
    # Total Q3 embeddings in K16 is large.
    
    # Alternative: Check if the induced subgraph on any 8 vertices is Q3?
    # Too slow.
    
    # Let's use the property: Q3 is bipartite.
    # Actually, let's use the "forbidden configuration" check specifically for the new edge.
    # If we add (u, v), a Q3 exists if there is a mapping of Q3 vertices to graph vertices 
    # such that the image of one of the Q3 edges is (u, v).
    
    # Let's precompute the "shells" around an edge.
    # In Q3, an edge connects a vertex of degree 3 to another of degree 3.
    # Let the edge be (0, 1) in Q3 (binary 000, 001).
    # The other vertices are 010, 011, 100, 101, 110, 111.
    # Neighbors of 0: 1, 2, 4
    # Neighbors of 1: 0, 3, 5
    # 
    # If we add edge (u, v) to the graph G, we need to check if there exist 
    # vertices a, b, c, d, e, f such that:
    # (u, v) corresponds to (0, 1).
    # Then u must be connected to a, b (corresponding to 2, 4)
    # v must be connected to c, d (corresponding to 3, 5)
    # a is connected to c (2-3 edge? 010-011 yes)
    # b is connected to d (100-101 yes)
    # c is connected to e (3-7? 011-111 yes)
    # d is connected to f (5-7? 101-111 yes)
    # a is connected to f? (2-6? 010-110 yes)
    # b is connected to e? (4-6? 100-110 yes)
    # e is connected to f? (7-6? 111-110 yes)
    # Wait, let's list edges of Q3 relative to edge (0,1):
    # Vertices: 0, 1, 2, 3, 4, 5, 6, 7
    # Edges:
    # (0,1) - The new edge
    # (0,2), (0,4) - u's other neighbors
    # (1,3), (1,5) - v's other neighbors
    # (2,3), (4,5) - Cross links
    # (2,6), (4,6) - Links to 6
    # (3,7), (5,7) - Links to 7
    # (6,7) - Link between 6 and 7
    
    # So if we add (u, v), we look for:
    # a, b in N(u) \ {v}
    # c, d in N(v) \ {u}
    # e, f in V \ {u, v, a, b, c, d}
    # Such that:
    # (a, c) in E
    # (b, d) in E
    # (a, e) in E
    # (b, e) in E
    # (c, f) in E
    # (d, f) in E
    # (e, f) in E
    # And NO other edges among these 8 vertices that are not in Q3?
    # Actually, we just need to find *a* Q3 subgraph. 
    # If the graph contains *any* edges among these 8 vertices that are NOT in the Q3 structure, 
    # it doesn't invalidate the Q3 subgraph (subgraph, not induced).
    # So we just need to satisfy the existence of these 12 edges.
    
    # Optimization: 
    # Precompute for each pair (u, v) the list of potential "witnesses" (a, b, c, d, e, f)?
    # That's too many.
    
    # Let's use a bitmask approach for the check.
    # It is faster to just check the specific pattern.
    
    def check_q3_with_edge(edge_mask, adj_masks, u, v):
        # u, v are the endpoints of the newly added edge.
        # Candidates for a, b: neighbors of u (excluding v)
        # Candidates for c, d: neighbors of v (excluding u)
        
        # Get neighbor lists from masks
        # adj_masks[u] is a bitmask. 
        # We need to iterate over set bits.
        
        def bits(mask):
            while mask:
                lsb = mask & -mask
                yield lsb.bit_length() - 1
                mask ^= lsb

        nu = list(bits(adj_masks[u] & ~(1 << v)))
        nv = list(bits(adj_masks[v] & ~(1 << u)))
        
        if len(nu) < 2 or len(nv) < 2:
            return False
            
        # Iterate over pairs (a, b) from nu and (c, d) from nv
        # To avoid duplicates and ensure a!=b, c!=d
        for i_a in range(len(nu)):
            a = nu[i_a]
            for i_b in range(i_a + 1, len(nu)):
                b = nu[i_b]
                # Check (a, c) and (b, d) requirements? 
                # We need to pick c, d from nv.
                for i_c in range(len(nv)):
                    c = nv[i_c]
                    for i_d in range(i_c + 1, len(nv)):
                        d = nv[i_d]
                        
                        # Check edges (a, c) and (b, d)
                        if not (adj_masks[a] >> c) & 1: continue
                        if not (adj_masks[b] >> d) & 1: continue
                        
                        # Now we need e, f.
                        # e must be neighbor of a, b.
                        # f must be neighbor of c, d, e.
                        # And (e, f) must exist.
                        
                        # Candidates for e: N(a) & N(b) \ {u, v, c, d}
                        # Note: a, b are neighbors of u. c, d are neighbors of v.
                        # a, b are not neighbors of v (in Q3, 2 is not neighbor of 1).
                        # c, d are not neighbors of u.
                        
                        # Let's find e candidates.
                        # e must be distinct from u, v, a, b, c, d.
                        
                        common_ab = adj_masks[a] & adj_masks[b]
                        # Remove u, v, a, b, c, d
                        forbidden_mask = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        e_candidates_mask = common_ab & ~forbidden_mask
                        
                        if not e_candidates_mask:
                            continue
                            
                        for e in bits(e_candidates_mask):
                            # Now find f.
                            # f must be neighbor of c, d, e.
                            # And f distinct from u, v, a, b, c, d, e.
                            common_cde = adj_masks[c] & adj_masks[d] & adj_masks[e]
                            forbidden_mask_f = forbidden_mask | (1 << e)
                            f_candidates_mask = common_cde & ~forbidden_mask_f
                            
                            if not f_candidates_mask:
                                continue
                                
                            for f in bits(f_candidates_mask):
                                # Check if (e, f) is an edge. 
                                # It is, because f is in N(e).
                                # We found a Q3!
                                return True
        return False

    # Backtracking Search
    # We want to maximize edges. 
    # We can do a DFS. 
    # Branching factor 2 (add/remove). 
    # Depth 120. 
    # This is too deep for naive DFS without strong pruning.
    
    # Pruning: 
    # 1. If current edges + remaining edges <= best_found, prune.
    # 2. If adding an edge creates Q3, we MUST remove it (branch is dead end for 'add').
    
    # To make this efficient, we should order edges smartly.
    # If we process edges in a way that creates Q3s early, we prune early.
    # However, we want to KEEP edges. 
    # So we should try to ADD edges first.
    
    # Let's try a Greedy Heuristic first to get a good lower bound (best_found).
    # Then use Backtracking to improve.
    
    # Greedy:
    # Start with empty graph.
    # Try to add edges in random order? Or specific order?
    # Let's use the random restart idea from the seed but faster.
    
    import random
    random.seed(42)
    
    best_edge_count = 0
    best_edge_mask = 0
    best_adj_masks = [0]*n
    
    # Quick Greedy to find a good start
    # Try several random orderings
    for _ in range(5):
        if time.time() > deadline - 20: break
        
        order = list(range(m))
        random.shuffle(order)
        
        current_mask = 0
        current_adj = [0]*n
        count = 0
        
        for idx in order:
            u, v = all_edges[idx]
            # Try adding
            # Check if adding (u,v) creates Q3
            # We need to simulate adding it to check.
            # Since we are in a loop, we can just check.
            
            # Update adj temporarily? No, just pass to check function.
            # But check function needs adj_masks.
            # Let's create a helper that checks if edge (u,v) can be added to (mask, adj).
            
            # To avoid overhead, we'll just do it inline or with a fast check.
            
            # Optimization: 
            # If deg(u) < 2 or deg(v) < 2, cannot form Q3 (needs 3 neighbors each in Q3, 
            # but here u,v are part of edge, so they need 2 OTHER neighbors).
            # Actually, in Q3, every vertex has degree 3.
            # So if we add (u,v), u must have at least 2 other neighbors in the final Q3.
            # So if current deg(u) < 2, we can't form a Q3 *using u as a Q3 vertex*?
            # No, u might become a Q3 vertex later. 
            # But if we add (u,v) NOW, and deg(u) is 0, then u has no other neighbors 
            # to form the rest of the Q3 *yet*. 
            # However, we are adding edges one by one. 
            # If we add (u,v) now, and later add (u,a), (u,b), etc., 
            # the Q3 might form. 
            # So we cannot prune based on current degree alone unless we know 
            # we will never add more edges to u.
            
            # So we must check.
            
            # Simulate add
            new_mask = current_mask | (1 << idx)
            new_adj = current_adj[:] # Copy is expensive. 
            # Let's modify in place and revert if needed? 
            # Or just pass the mask and compute adj on the fly? 
            # Computing adj from mask is O(m). 
            # Maintaining adj is O(1) per edge.
            
            # Let's maintain adj.
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
            if not check_q3_with_edge(new_mask, current_adj, u, v):
                # Keep it
                current_mask = new_mask
                count += 1
            else:
                # Revert
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
        
        if count > best_edge_count:
            best_edge_count = count
            best_edge_mask = current_mask
            best_adj_masks = current_adj[:]
            
    # Now, Backtracking to improve.
    # We will use the best found so far as the lower bound.
    # We will try to add MORE edges.
    
    # Strategy: 
    # Identify edges NOT in best_edge_mask.
    # Try to add them one by one, using backtracking if a conflict arises.
    
    # Actually, a full backtracking search over 120 edges is hard.
    # Let's do a "Repair" phase.
    # Start with best_edge_mask.
    # For each missing edge, try to add it.
    # If it creates a Q3, try to remove one edge from the Q3 to allow this addition?
    # This is complex.
    
    # Simpler: 
    # Use the backtracking search but with a strong heuristic: 
    # Process edges in the order of the best solution found, then the rest.
    # No, that's biased.
    
    # Let's try a different structural approach:
    # **Integer Linear Programming (ILP)**? 
    # Standard library doesn't have ILP. 
    # But we can use a simple branch and bound.
    
    # Given the 120s limit, and the complexity of Q3 check, 
    # the Greedy approach above likely finds a good solution quickly.
    # The "remove random edge" approach in the seed is slow because it searches for Q3s 
    # in the whole graph repeatedly.
    
    # My `check_q3_with_edge` is local to the new edge, which is much faster.
    # Let's refine the Greedy to be more robust.
    
    # 1. Start with empty graph.
    # 2. Sort edges by a heuristic? 
    #    Maybe edges that are "harder" to include first?
    #    Or just random restarts with the fast check.
    
    # Let's run more restarts with the fast check.
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        order = list(range(m))
        random.shuffle(order)
        
        current_mask = 0
        current_adj = [0]*n
        count = 0
        
        for idx in order:
            u, v = all_edges[idx]
            
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
            if not check_q3_with_edge(current_mask, current_adj, u, v):
                current_mask |= (1 << idx)
                count += 1
            else:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
        if count > best_edge_count:
            best_edge_count = count
            best_edge_mask = current_mask
            best_adj_masks = current_adj[:]
            
    # Construct result
    result = []
    for i in range(m):
        if (best_edge_mask >> i) & 1:
            result.append(list(all_edges[i]))
            
    return result

# EVOLVE-BLOCK-END
