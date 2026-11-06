"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    deadline = time.time() + 110  # Safety margin before 120s kill
    d = 7
    nverts = 1 << d
    total_edges = d * nverts // 2  # 448
    
    # Precompute all edges and their "co-edge" relationships for fast C4 checking
    # An edge (u, v) is part of a C4 if there exists another edge (c, e) 
    # such that u-c, v-e, and c-e are also in the set.
    # Actually, for a 2-face in Q_d, the edges are (u,v), (u',v'), (u,u'), (v,v').
    # If we add (u,v), it completes a C4 if for any dimension j != i (where i is dim of u,v),
    # the other 3 edges of the face containing (u,v) in direction j are present.
    
    # Let's precompute for each edge index, the list of "blocking" edge indices.
    # Edge is defined by (u, v) with u < v.
    # Let's map edge to index.
    
    all_edges = []
    edge_map = {}
    for x in range(nverts):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_map[(x, y)] = idx
    
    num_edges = len(all_edges)
    
    # Precompute conflicts: for each edge, which other edges form a C4 with it?
    # If we have edges e1, e2, e3, e4 forming a cycle, adding e4 is blocked if e1, e2, e3 are present.
    # For a given edge e = (u, v) in dimension i:
    # It belongs to (d-1) squares.
    # In square j (j != i), the other 3 edges are:
    # (u, u^bit_j), (v, v^bit_j), (u^bit_j, v^bit_j)
    
    # We want to check if adding edge `e` creates a C4.
    # This happens if ANY of the (d-1) squares containing `e` has its other 3 edges present.
    
    # Let's precompute for each edge `e`, the list of "witness" edges.
    # If any witness edge is present, and the other 2 are present, it's a C4? No.
    # The condition is: all 3 other edges are present.
    
    # So for each edge `e`, we store a list of tuples of 3 edge indices.
    # If all 3 indices are in the current set, `e` cannot be added (or must be removed).
    
    blockers = [[] for _ in range(num_edges)]
    for idx, (u, v) in enumerate(all_edges):
        # Determine dimension i
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            bit_j = 1 << j
            u2 = u ^ bit_j
            v2 = v ^ bit_j
            
            # The 3 other edges in this square:
            # 1: (u, u2)
            # 2: (v, v2)
            # 3: (u2, v2)
            
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            
            idx1 = edge_map.get(e1)
            idx2 = edge_map.get(e2)
            idx3 = edge_map.get(e3)
            
            if idx1 is not None and idx2 is not None and idx3 is not None:
                blockers[idx].append((idx1, idx2, idx3))
                
    best_set = set()
    best_count = 0
    
    # Strategy: Randomized Local Search
    # 1. Start with a random subset of edges (e.g., 60% density)
    # 2. Remove all edges that form C4s (repair phase)
    # 3. Try to add back removed edges or swap (improve phase)
    
    def repair_and_optimize(initial_set, max_iters=500):
        # initial_set is a set of edge indices
        current = set(initial_set)
        
        # Phase 1: Repair - Remove edges that complete C4s
        # We can do this iteratively until no more C4s are formed by removal?
        # Actually, if we start with a random set, we just need to ensure the final set is C4-free.
        # A simple way: iterate through edges, if an edge is in `current` and forms a C4 with other edges in `current`, remove it.
        # To make it deterministic and fast, we just check if adding it would be bad, but here we are removing.
        # If an edge is present, and its 3 blockers are present, we have a C4.
        # We must remove at least one edge from the C4.
        
        # Let's just clear the set and greedily add? No, that's the parent strategy.
        # Let's try: Start with random set. 
        # Remove any edge that is part of a C4.
        # Repeat until stable? Or just one pass?
        
        # Better: 
        # 1. Start with random set S.
        # 2. While there is a C4 in S, remove a random edge from it.
        # This is slow.
        
        # Alternative: 
        # 1. Start with random set S.
        # 2. Create a valid set V by iterating through S in random order, adding edge if it doesn't form a C4 with V.
        # This is essentially randomized greedy, which is what the inspiration suggested (score 1.078).
        # The prompt asks for a DIFFERENT strategy.
        
        # Let's try Hill Climbing on the "removed" set or the "included" set.
        # Start with a dense random set.
        # Remove all edges that are "critical" for C4s?
        
        # Let's try this:
        # 1. Generate random permutation of all edges.
        # 2. Greedy add (like parent) but with random order. This is what got 1.078.
        # The directive says "not refining the current program... rethinking the approach".
        # The current program is DETERMINISTIC greedy.
        # The inspiration used RANDOMIZED greedy and got 1.078.
        # I should try something else, like Local Search / Simulated Annealing or a different heuristic.
        
        # Let's try: Start with ALL edges. Remove edges to break all C4s.
        # This is a hitting set problem / vertex cover on the hypergraph of C4s.
        # Greedy hitting set: Pick an edge that is in the most remaining C4s, remove it, update C4s.
        
        # Let's implement a greedy "remove" strategy.
        # 1. List all C4s in Q7.
        # 2. While there are C4s:
        #    a. Find C4 with most edges present? No, all edges are present initially.
        #    b. Just pick a random C4, remove a random edge from it.
        #    c. This might be slow.
        
        # Let's try: 
        # Start with all edges.
        # Count how many C4s each edge is part of.
        # Remove the edge with the highest count (or random among top).
        # Update counts for all C4s containing that edge.
        
        # How many C4s in Q7?
        # A C4 is determined by 2 dimensions i, j and a base vertex u.
        # The edges are (u, u^i), (u, u^j), (u^i, u^i^j), (u^j, u^i^j).
        # Number of C4s = C(7, 2) * 2^(7-2) = 21 * 32 = 672.
        # Each C4 has 4 edges.
        # Total edge-C4 incidences = 672 * 4 = 2688.
        # Each edge is in (7-1) = 6 C4s? 
        # Edge (u, v) in dim i. It is in squares for each j != i. So 6 C4s.
        # 448 edges * 6 = 2688. Matches.
        
        # So we have a small hypergraph.
        
        c4s = []
        c4_edges = []
        for i in range(d):
            for j in range(i+1, d):
                bit_i = 1 << i
                bit_j = 1 << j
                for u in range(nverts):
                    # Only consider u such that u < v for the first edge? 
                    # The C4 is a set of 4 edges.
                    # Edges:
                    # e1: (u, u^bit_i)
                    # e2: (u, u^bit_j)
                    # e3: (u^bit_i, u^bit_i^bit_j)
                    # e4: (u^bit_j, u^bit_i^bit_j)
                    
                    # We need unique representation of the C4.
                    # The set of 4 vertices is {u, u^i, u^j, u^ij}.
                    # This set is unique for the C4.
                    # We can generate it for u where u has 0 in bits i and j?
                    # Or just generate all and use a set of frozensets of vertices?
                    # Or set of sorted edge indices.
                    
                    v1 = u
                    v2 = u ^ bit_i
                    v3 = u ^ bit_j
                    v4 = u ^ bit_i ^ bit_j
                    
                    # Edges
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v4), max(v2, v4))
                    e4 = (min(v3, v4), max(v3, v4))
                    
                    idx1 = edge_map[e1]
                    idx2 = edge_map[e2]
                    idx3 = edge_map[e3]
                    idx4 = edge_map[e4]
                    
                    c4_tuple = tuple(sorted([idx1, idx2, idx3, idx4]))
                    if c4_tuple not in c4_edges:
                        c4_edges.append(c4_tuple)
                        c4s.append(c4_tuple)
                        
        # c4s is a list of tuples of 4 edge indices.
        
        # Greedy Removal Strategy:
        # 1. Start with all edges present.
        # 2. Maintain a list of "active" C4s (those with all 4 edges present).
        # 3. While active C4s exist:
        #    a. Pick a C4 (e.g., random or one with high edge-degree).
        #    b. Remove one edge from it.
        #    c. Update active C4s: any C4 containing that edge is no longer active (unless it was already inactive, but we only track active).
        #       Actually, we need to know which C4s become inactive.
        #       We can maintain a count of "missing edges" for each C4?
        #       Or just check.
        
        # Since we start with all edges, all 672 C4s are active.
        # When we remove an edge, all C4s containing it become inactive (have a missing edge).
        # But wait, if we remove an edge, we might break multiple C4s.
        # We want to maximize remaining edges.
        # This is equivalent to finding a minimum edge cover of the C4 hypergraph? No.
        # We want to remove the MINIMUM number of edges to break ALL C4s.
        # This is the "Hypergraph Edge Cover" problem (minimum hitting set).
        # Greedy heuristic:
        # 1. Count how many active C4s each edge is in.
        # 2. Remove the edge with the highest count.
        # 3. Decrement count for all C4s containing that edge. If count becomes 0, it's "broken" (no longer a threat).
        # 4. Repeat until no active C4s.
        
        # Let's implement this.
        
        def greedy_removal():
            # active_c4s: set of indices of C4s that are still "alive" (all 4 edges present)
            # edge_c4_count: how many active C4s each edge is part of
            # c4_edge_list: for each C4, list of edges
            
            active_c4s = set(range(len(c4s)))
            edge_c4_count = [0] * num_edges
            # For each C4, which edges does it contain?
            # We already have c4s as list of 4 edge indices.
            
            # Initialize counts
            for c4_idx, c4 in enumerate(c4s):
                if c4_idx in active_c4s:
                    for e in c4:
                        edge_c4_count[e] += 1
            
            removed = set()
            
            while active_c4s:
                # Find edge with max count
                # If counts are 0 for all, something is wrong.
                max_count = max(edge_c4_count)
                if max_count == 0:
                    break # Should not happen if active_c4s is not empty
                
                # Pick random edge among those with max count? Or just the first?
                # Randomization helps escape local optima.
                candidates = [e for e in range(num_edges) if edge_c4_count[e] == max_count]
                e_to_remove = random.choice(candidates)
                
                removed.add(e_to_remove)
                
                # Update active C4s
                # We need to know which C4s contain e_to_remove.
                # We can precompute edge_to_c4s: list of C4 indices for each edge.
                
                # Let's precompute this outside the loop for speed.
                # But for now, let's just iterate all active C4s? 672 is small.
                new_active = set()
                for c4_idx in active_c4s:
                    c4 = c4s[c4_idx]
                    if e_to_remove in c4:
                        # This C4 is broken
                        # Decrement counts for other edges in this C4
                        for e in c4:
                            if e != e_to_remove:
                                edge_c4_count[e] -= 1
                        # c4_idx is removed from active_c4s
                    else:
                        new_active.add(c4_idx)
                
                active_c4s = new_active
                
            # The remaining edges are all_edges - removed
            return list(set(range(num_edges)) - removed)
        
        # Run greedy removal multiple times with different random seeds?
        # The greedy choice is deterministic if we pick the first max. 
        # Randomizing the choice among max-count edges adds diversity.
        
        best = 0
        best_edges = []
        
        num_restarts = 0
        while time.time() < deadline:
            res = greedy_removal()
            count = len(res)
            if count > best:
                best = count
                best_edges = res
                num_restarts += 1
                # If we found a very good one, maybe try more?
                # Just keep going.
        
        # Convert indices back to [u, v] format
        out = []
        for idx in best_edges:
            u, v = all_edges[idx]
            out.append([u, v])
            
        return out

# EVOLVE-BLOCK-END
