import random
import time

def construct():
    d = 7
    n = 2 ** d
    
    # 1. Precompute edges
    edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                edge_index[e] = len(edges)
                edges.append(e)
                
    E = len(edges) # 448
    
    # 2. Precompute blocking triples for each edge
    # For each edge e, find all 4-cycles (squares) containing e.
    # A square is defined by two vertices u, v that are distance 2 apart, and the two middle vertices.
    # Or simpler: For edge (u, v), the other dimension j defines the square.
    # Square vertices: u, v, u^j, v^j.
    # Edges in square: (u,v), (u, u^j), (v, v^j), (u^j, v^j).
    # If we pick (u,v), the other 3 are the blocking triple.
    
    blocking_triples = [[] for _ in range(E)]
    
    for idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            bit_j = 1 << j
            u_j = u ^ bit_j
            v_j = v ^ bit_j
            
            # The other 3 edges
            e1 = (min(u, u_j), max(u, u_j))
            e2 = (min(v, v_j), max(v, v_j))
            e3 = (min(u_j, v_j), max(u_j, v_j))
            
            # Get their indices
            idx1 = edge_index[e1]
            idx2 = edge_index[e2]
            idx3 = edge_index[e3]
            
            blocking_triples[idx].append((idx1, idx2, idx3))
            
    # 3. Helper to check if an edge can be added
    def can_add(edge_idx, chosen_set):
        for triple in blocking_triples[edge_idx]:
            if triple[0] in chosen_set and triple[1] in chosen_set and triple[2] in chosen_set:
                return False
        return True

    def add_edge(edge_idx, chosen_set):
        chosen_set.add(edge_idx)
        
    def remove_edge(edge_idx, chosen_set):
        chosen_set.discard(edge_idx)

    best_count = 0
    best_set = set()
    
    deadline = time.time() + 110
    
    # Local Search Improvement
    def local_search(chosen_set):
        count = len(chosen_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to swap one edge for another
            # We iterate over edges not in chosen
            # For each such edge, if it is blocked by exactly 1 edge in chosen, we can swap.
            # This might open up space for more edges.
            
            # To be efficient, we just do a few passes
            for _ in range(10): # Max 10 passes of swaps
                if time.time() >= deadline:
                    break
                swap_made = False
                # Shuffle order of non-chosen edges to avoid bias
                non_chosen = [i for i in range(E) if i not in chosen_set]
                random.shuffle(non_chosen)
                
                for e_idx in non_chosen:
                    if time.time() >= deadline:
                        break
                    if e_idx in chosen_set:
                        continue
                    
                    # Find which edges in chosen block e_idx
                    # e_idx is blocked if any of its blocking triples is fully in chosen.
                    # We want to find a single edge b in chosen such that removing b allows e_idx to be added.
                    # This implies that the blocking triple for e_idx contains b, and the other 2 are in chosen.
                    # And there are no OTHER blocking triples for e_idx that are fully in chosen (after removing b).
                    
                    blockers = []
                    for triple in blocking_triples[e_idx]:
                        # Check how many of the triple are in chosen
                        in_chosen = [t for t in triple if t in chosen_set]
                        if len(in_chosen) == 3:
                            # This triple is currently fully present, so e_idx is blocked by this triple.
                            # To unblock, we must remove at least one edge from this triple.
                            # If we remove one, we need to ensure no other triple is still fully present.
                            pass
                        elif len(in_chosen) == 2:
                            # If we remove the missing one? No, we are removing from chosen.
                            # If 2 are in chosen, and 1 is not, removing one of the 2 won't help unless the 3rd was the one we removed?
                            # Wait. The condition for "blocked" is that ALL 3 are in chosen.
                            # So if 2 are in chosen, it's not blocked by this triple.
                            # We only care about triples where all 3 are in chosen.
                            pass
                            
                    # Let's re-evaluate.
                    # e_idx is blocked if there exists a triple T in blocking_triples[e_idx] such that T subset of chosen_set.
                    # To unblock, we need to remove at least one edge from EVERY blocking triple.
                    # If there is only ONE blocking triple, and we remove one edge from it, it's unblocked.
                    
                    blocking_triples_present = []
                    for triple in blocking_triples[e_idx]:
                        if triple[0] in chosen_set and triple[1] in chosen_set and triple[2] in chosen_set:
                            blocking_triples_present.append(triple)
                            
                    if not blocking_triples_present:
                        # Should not happen if we maintain validity, but if it does, just add
                        add_edge(e_idx, chosen_set)
                        improved = True
                        swap_made = True
                        break
                        
                    if len(blocking_triples_present) == 1:
                        # Only one triple is blocking.
                        # We can remove any edge from this triple and add e_idx.
                        # This is a swap. Count stays same.
                        # But it might allow adding other edges.
                        # Let's try removing the first edge in the triple and adding e_idx.
                        # To be smarter, we could try all 3 options and see which leads to more additions.
                        # For speed, just pick one.
                        t = blocking_triples_present[0]
                        # Try removing t[0]
                        remove_edge(t[0], chosen_set)
                        add_edge(e_idx, chosen_set)
                        improved = True
                        swap_made = True
                        break
                    elif len(blocking_triples_present) == 2:
                        # We need to remove an edge that is in the intersection of these two triples?
                        # Or remove two edges?
                        # If we remove one edge, it must break BOTH triples.
                        # So the edge must be in T1 and T2.
                        t1 = set(blocking_triples_present[0])
                        t2 = set(blocking_triples_present[1])
                        common = t1.intersection(t2)
                        if common:
                            # Remove one common edge
                            rem_edge = list(common)[0]
                            remove_edge(rem_edge, chosen_set)
                            add_edge(e_idx, chosen_set)
                            improved = True
                            swap_made = True
                            break
                        else:
                            # No common edge. We would need to remove 2 edges to add 1. Net loss.
                            # Skip.
                            pass
                            
            if not swap_made:
                break
                
        return len(chosen_set)

    # Main Loop
    while time.time() < deadline:
        # Randomized Greedy
        chosen = set()
        edge_order = list(range(E))
        random.shuffle(edge_order)
        
        for e_idx in edge_order:
            if can_add(e_idx, chosen):
                chosen.add(e_idx)
                
        count = len(chosen)
        
        # Local Search
        if time.time() < deadline - 5: # Reserve time
            count = local_search(chosen)
            
        if count > best_count:
            best_count = count
            best_set = set(chosen)
            
    # Convert back to list of edges
    return [list(edges[i]) for i in best_set]
