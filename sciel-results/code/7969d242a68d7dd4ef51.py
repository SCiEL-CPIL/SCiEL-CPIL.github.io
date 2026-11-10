"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 115  # 115s safety margin before 120s limit
    n = 16
    
    # Generate all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    
    # Helper to check if edge (u,v) creates a Q3
    def creates_q3(u, v, edge_set):
        """
        Check if adding edge (u,v) to edge_set creates a subgraph isomorphic to Q3.
        Q3 vertices: 0..7. Edges connect i,j if i^j is power of 2.
        Structure relative to edge (0,1):
        0-1
        0-2, 1-3, 2-3 (face 0-1-3-2) -> Wait, 2 is 010, 3 is 011. 0(000)-2(010), 1(001)-3(011), 2-3.
        0-4, 1-5, 4-5 (face 0-1-5-4) -> 4 is 100, 5 is 101.
        2-6, 4-6, 3-7, 5-7, 6-7 (back edges)
        Actually, let's stick to the neighbor iteration logic which is robust.
        """
        # Candidates for a2: neighbors of u (excluding v)
        neigh_u = set()
        for w in range(n):
            if w == v:
                continue
            if (min(u, w), max(u, w)) in edge_set:
                neigh_u.add(w)
        
        neigh_v = set()
        for w in range(n):
            if w == u:
                continue
            if (min(v, w), max(v, w)) in edge_set:
                neigh_v.add(w)
        
        if len(neigh_u) < 2 or len(neigh_v) < 2:
            return False
            
        neigh_u_list = list(neigh_u)
        neigh_v_list = list(neigh_v)
        
        for a2 in neigh_u_list:
            for a4 in neigh_u_list:
                if a4 == a2:
                    continue
                # In Q3, a2 and a4 are distinct neighbors of 0.
                # They must not be connected to each other? 
                # In Q3, neighbors of 0 are 1, 2, 4. 1-2? No. 1-4? No. 2-4? No.
                # So a2 and a4 should NOT be connected.
                if (min(a2, a4), max(a2, a4)) in edge_set:
                    continue
                
                for a3 in neigh_v_list:
                    if a3 == a2 or a3 == a4:
                        continue
                    # a3 is neighbor of 1. a2 is neighbor of 0.
                    # In Q3, 2 (neighbor of 0) is connected to 3 (neighbor of 1).
                    # So edge (a2, a3) MUST exist.
                    if (min(a2, a3), max(a2, a3)) not in edge_set:
                        continue
                    
                    for a5 in neigh_v_list:
                        if a5 == a2 or a5 == a3 or a5 == a4:
                            continue
                        # a5 is neighbor of 1. a4 is neighbor of 0.
                        # In Q3, 4 (neighbor of 0) is connected to 5 (neighbor of 1).
                        # So edge (a4, a5) MUST exist.
                        if (min(a4, a5), max(a4, a5)) not in edge_set:
                            continue
                        
                        # Now we have 0->u, 1->v, 2->a2, 3->a3, 4->a4, 5->a5
                        # We need 6 and 7.
                        # 6 is connected to 2(a2), 4(a4), 7.
                        # 7 is connected to 3(a3), 5(a5), 6.
                        # 6 and 7 must be distinct from u,v,a2,a3,a4,a5.
                        
                        used = {u, v, a2, a3, a4, a5}
                        
                        # Find candidates for 6: common neighbors of a2 and a4, not in used
                        cn_a2a4 = []
                        for w in range(n):
                            if w in used:
                                continue
                            if (min(a2, w), max(a2, w)) in edge_set and (min(a4, w), max(a4, w)) in edge_set:
                                cn_a2a4.append(w)
                                
                        if not cn_a2a4:
                            continue
                            
                        # Find candidates for 7: common neighbors of a3 and a5, not in used
                        cn_a3a5 = []
                        for w in range(n):
                            if w in used:
                                continue
                            if (min(a3, w), max(a3, w)) in edge_set and (min(a5, w), max(a5, w)) in edge_set:
                                cn_a3a5.append(w)
                                
                        if not cn_a3a5:
                            continue
                            
                        for a6 in cn_a2a4:
                            for a7 in cn_a3a5:
                                if a7 == a6:
                                    continue
                                # Check edge (a6, a7)
                                if (min(a6, a7), max(a6, a7)) in edge_set:
                                    return True
        return False

    def try_add_edges(initial_set, edge_order):
        """Try to add edges from edge_order to initial_set, checking for Q3."""
        trial_set = initial_set.copy()
        added = 0
        for u, v in edge_order:
            if (u, v) in trial_set:
                continue
            if time.time() > deadline:
                break
            if not creates_q3(u, v, trial_set):
                trial_set.add((u, v))
                added += 1
        return trial_set, added

    # Strategy:
    # 1. Random restarts from empty set.
    # 2. Refine the current best by trying to add missing edges.
    
    random.seed(42)
    best_result = set()
    best_count = 0
    
    # Initial run: Natural order (deterministic baseline)
    trial_set, _ = try_add_edges(set(), all_edges)
    best_result = trial_set
    best_count = len(best_result)
    
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        
        # Phase 1: Random Restart from Empty
        random_order = all_edges[:]
        random.shuffle(random_order)
        trial_set, _ = try_add_edges(set(), random_order)
        
        if len(trial_set) > best_count:
            best_count = len(trial_set)
            best_result = trial_set.copy()
            
        # Phase 2: Refine Current Best
        # Try to add edges not in best_result to best_result
        # This is effective because the greedy order that built best_result might have
        # blocked some edges that could be added if we try them in a different context
        # or if we simply retry the "failed" edges (though they failed for a reason, 
        # maybe the set changed? No, set is fixed. 
        # Wait, if an edge was rejected in the greedy pass, it was because it created a Q3 
        # with the *current* set at that time. Since we only add edges, the set only grows.
        # So an edge that created a Q3 before will still create a Q3 (or the Q3 is still there).
        # Actually, if edge e was rejected because it completed a Q3 with existing edges E1,
        # and we later add more edges, that Q3 still exists.
        # So "Refine Best" by retrying rejected edges on the *same* set is useless.
        
        # HOWEVER, if we found a *different* good set in Phase 1, we update best_result.
        # The refinement should be: Take the NEW best_result (from Phase 1 or previous best)
        # and see if we can improve it by starting from it and shuffling the REMAINING edges.
        # But "remaining edges" are those not in best_result.
        # If we start from best_result and try to add remaining edges, we are just doing
        # another greedy pass on the complement.
        
        # Let's try: Start from best_result, shuffle ALL edges (or just remaining), and try to add.
        # Since best_result is already a valid graph, adding edges to it is safe.
        # The issue is that the *order* in which we try to add the missing edges might matter?
        # No, if an edge can be added to a graph G, it can be added regardless of the order 
        # of *other* edges added simultaneously, as long as we check validity at each step.
        # Wait, greedy is sequential. 
        # If edge A and B are both missing from G. 
        # If we add A, B might become invalid. If we add B, A might become invalid.
        # So order matters for the *final* set size.
        # So, shuffling the missing edges and trying to add them to best_result IS a valid refinement step.
        
        remaining = [e for e in all_edges if e not in best_result]
        if remaining:
            random.shuffle(remaining)
            # Use a copy to avoid modifying best_result until we know it's better
            trial_set2, _ = try_add_edges(best_result, remaining)
            
            if len(trial_set2) > best_count:
                best_count = len(trial_set2)
                best_result = trial_set2.copy()
                
        # Occasionally, try starting from K3,13 as well, just in case empty start is a local trap
        if attempts % 5 == 0:
             k313_set = set()
             for a in range(3):
                 for b in range(3, 16):
                     k313_set.add((a, b))
             # Add remaining edges to K3,13
             rem_k = [e for e in all_edges if e not in k313_set]
             random.shuffle(rem_k)
             trial_set3, _ = try_add_edges(k313_set, rem_k)
             if len(trial_set3) > best_count:
                 best_count = len(trial_set3)
                 best_result = trial_set3.copy()
                 
    # Convert back to list of edges
    result_edges = [[u, v] for u, v in best_result]
    return result_edges

# EVOLVE-BLOCK-END
