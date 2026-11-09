import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    n = 16
    adj = [[0] * n for _ in range(n)]
    edges = set()

    def add_edge(u, v):
        if u == v:
            return
        if adj[u][v]:
            return
        adj[u][v] = 1
        adj[v][u] = 1
        edges.add((min(u, v), max(u, v)))

    def remove_edge(u, v):
        if not adj[u][v]:
            return
        adj[u][v] = 0
        adj[v][u] = 0
        edges.discard((min(u, v), max(u, v)))

    # Q3 is isomorphic to K_{4,4} minus a perfect matching.
    # A graph G contains Q3 if and only if there exists a set of 8 vertices S and a bipartition (A, B) of S
    # such that the missing edges between A and B in G form a matching (i.e., are a subset of a perfect matching).

    # Precompute all 35 bipartitions of 8 items into two sets of 4.
    all_splits = []
    seen = set()
    for A in combinations(range(8), 4):
        B = tuple(i for i in range(8) if i not in A)
        key = frozenset([frozenset(A), frozenset(B)])
        if key not in seen:
            seen.add(key)
            mask_a = 0
            mask_b = 0
            for i in A:
                mask_a |= (1 << i)
            for i in B:
                mask_b |= (1 << i)
            all_splits.append((mask_a, mask_b, A, B))
    
    # all_splits now has 35 entries.

    def contains_q3_in_subset_fast(v_list):
        # Build local adjacency masks
        local_adj = [0] * 8
        for i in range(8):
            vi = v_list[i]
            mask = 0
            for j in range(8):
                if i != j and adj[vi][v_list[j]]:
                    mask |= (1 << j)
            local_adj[i] = mask
            
        for mask_a, mask_b, A_idx, B_idx in all_splits:
            covered = 0
            is_matching = True
            missing_count = 0
            
            for i in A_idx:
                for j in B_idx:
                    if not (local_adj[i] & (1 << j)):
                        missing_count += 1
                        if missing_count > 4:
                            is_matching = False
                            break
                        if (covered & (1 << i)) or (covered & (1 << j)):
                            is_matching = False
                            break
                        covered |= (1 << i)
                        covered |= (1 << j)
                if not is_matching:
                    break
            if is_matching:
                return True
        return False

    # Initialize with a good seed
    # K3,13 structure:
    for a in range(3):
        for b in range(3, 16):
            add_edge(a, b)
    add_edge(0, 1)
    add_edge(0, 2)
    add_edge(1, 2)
    
    best_edges = len(edges)
    best_adj = [row[:] for row in adj]
    best_edge_set = set(edges)
    
    # Simulated Annealing
    # Objective: Maximize number of edges subject to no Q3.
    # Penalty function: -100 * (number of Q3s) + (number of edges)
    # However, counting all Q3s is expensive.
    # Instead, we can use a local search that accepts moves that increase edges or maintain edges if they don't increase Q3s?
    # Actually, the "Max-SAT" framing suggests we should allow violations temporarily.
    # But checking if a move creates a Q3 is the bottleneck.
    
    # Let's stick to the "greedy with restarts" or "local search" but improve the move set.
    # The current program does single edge additions.
    # Let's try: 
    # 1. Start with a dense random graph or the K3,13 seed.
    # 2. Try to add edges. If it creates a Q3, remove one edge from the Q3 (not necessarily the new one) to resolve the conflict, if that results in a net gain or same number of edges but different structure.
    # Actually, a simpler and more robust approach for this specific problem (small n) is:
    # 1. Generate a random valid graph (or start from seed).
    # 2. Try to add a random missing edge.
    # 3. If it creates a Q3, try to fix it by removing a different edge from the Q3 (if possible, i.e., if the Q3 has >1 edge from the "new" part? No, the new edge is the only one that changed).
    #    Wait, if adding (u,v) creates a Q3, that Q3 must contain (u,v).
    #    If we remove a different edge (x,y) from that Q3, we break the Q3.
    #    Net change: +1 edge (u,v), -1 edge (x,y). Total edges unchanged.
    #    This allows us to "swap" edges and potentially reach a different local optimum.
    
    # Let's implement a "swap" move in addition to "add" move.
    
    # To make this efficient, we need to find the Q3 quickly when adding an edge.
    # The current code finds *any* Q3 in the subset. It doesn't return the specific Q3.
    # We need to modify contains_q3_in_subset_fast to return the Q3 (the 8 vertices and the specific split) if found.
    
    def find_q3_in_subset(v_list):
        local_adj = [0] * 8
        for i in range(8):
            vi = v_list[i]
            mask = 0
            for j in range(8):
                if i != j and adj[vi][v_list[j]]:
                    mask |= (1 << j)
            local_adj[i] = mask
            
        for mask_a, mask_b, A_idx, B_idx in all_splits:
            covered = 0
            is_matching = True
            missing_count = 0
            missing_edges = []
            
            for i in A_idx:
                for j in B_idx:
                    if not (local_adj[i] & (1 << j)):
                        missing_count += 1
                        if missing_count > 4:
                            is_matching = False
                            break
                        if (covered & (1 << i)) or (covered & (1 << j)):
                            is_matching = False
                            break
                        covered |= (1 << i)
                        covered |= (1 << j)
                        missing_edges.append((i, j))
                if not is_matching:
                    break
            if is_matching:
                # Return the 8 vertices and the specific missing edges (which form a matching)
                # The Q3 is the induced subgraph on v_list minus the missing edges?
                # No, Q3 is K_{4,4} minus a perfect matching.
                # The edges of the Q3 are all edges between A and B EXCEPT the missing ones.
                # Wait, the definition says "missing edges form a matching".
                # So the Q3 edges are the present edges between A and B that are NOT in the missing set?
                # No. Q3 is a subgraph.
                # If G contains Q3, there is an isomorphism.
                # The vertices of Q3 are mapped to v_list.
                # The edges of Q3 are mapped to edges in G.
                # In K_{4,4}, all 16 edges are present.
                # In Q3, 12 edges are present (the other 4 are missing).
                # So the edges of the Q3 in G are the edges between A and B that are PRESENT in G.
                # And there must be exactly 12 such edges, and the 4 missing ones must form a matching.
                # My check `is_matching` ensures the missing ones form a matching.
                # It also implicitly ensures there are <= 4 missing.
                # But does it ensure there are exactly 12 present?
                # If missing_count < 4, then present > 12. That would mean G contains K_{4,4} minus a matching of size < 4.
                # K_{4,4} minus a matching of size 3 is NOT Q3.
                # Q3 has exactly 12 edges.
                # So we must check `missing_count == 4`?
                # No, if missing_count < 4, it's a supergraph of Q3? No, Q3 is a specific graph.
                # If G has K_{4,4} minus 2 edges (which form a matching), it contains Q3?
                # Q3 is K_{4,4} minus a perfect matching (4 edges).
                # If G has K_{4,4} minus 2 edges, it contains Q3 as a subgraph?
                # Yes, you can just ignore the 2 extra edges.
                # So if missing_count <= 4 and forms a matching, it contains Q3.
                # Wait, if missing_count = 0, it's K_{4,4}, which contains Q3.
                # So the condition is: missing edges form a matching (size <= 4).
                # My code checks this.
                
                return v_list, A_idx, B_idx, missing_edges
        return None

    # Main Loop
    iterations = 0
    max_iterations = 100000
    last_improvement_time = start_time
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Pick random missing edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or adj[u][v]:
            continue
            
        add_edge(u, v)
        
        found_q3_info = None
        other_vertices = [i for i in range(n) if i != u and i != v]
        
        for combo in combinations(other_vertices, 6):
            vertices = (u, v) + tuple(combo)
            result = find_q3_in_subset(vertices)
            if result:
                found_q3_info = result
                break
                
        if found_q3_info:
            v_list, A_idx, B_idx, missing_edges = found_q3_info
            # We have a Q3. We must remove an edge to break it.
            # Option 1: Remove the new edge (u,v).
            # Option 2: Remove a different edge from the Q3.
            
            # Let's try to remove a random edge from the Q3 (excluding the new one if possible? or including?)
            # The Q3 edges are the ones between A and B that are PRESENT in G.
            # We need to identify which edges are present.
            # present_edges = []
            # for i in A_idx:
            #   for j in B_idx:
            #     if adj[v_list[i]][v_list[j]]:
            #       present_edges.append((v_list[i], v_list[j]))
            
            # If we remove a random present edge, we break the Q3.
            # Net change: 0 edges (added 1, removed 1).
            # This allows exploration.
            
            present_edges = []
            for i in A_idx:
                vi = v_list[i]
                for j in B_idx:
                    vj = v_list[j]
                    if adj[vi][vj]:
                        present_edges.append((vi, vj))
            
            if present_edges:
                # Pick a random edge to remove.
                # Prefer removing an edge that is NOT (u,v) to keep the new edge?
                # If we always keep the new edge, we might get stuck in a bad state.
                # Let's remove a random edge from present_edges.
                rem_u, rem_v = random.choice(present_edges)
                remove_edge(rem_u, rem_v)
                
                # After removal, we should check if the graph is now valid (no Q3).
                # It might have created a new Q3? No, removing an edge cannot create a Q3.
                # But it might have broken the Q3 we just found, but left another one?
                # The invariant was "no Q3" before adding (u,v).
                # Adding (u,v) created at least one Q3.
                # Removing an edge from that Q3 breaks that specific Q3.
                # Could it leave another Q3?
                # If there was another Q3 that didn't use (u,v), it would have existed before.
                # If there was another Q3 that used (u,v), removing (rem_u, rem_v) might not break it if (rem_u, rem_v) is not in it.
                # So we might still have a Q3!
                # We need to ensure the graph is Q3-free.
                
                # Simplest: if we remove an edge and are not sure, we can just revert to the state before adding (u,v) if we don't want to risk it.
                # Or, we can verify.
                
                # Let's try to verify if the graph is still valid.
                # This is expensive.
                
                # Alternative: Just remove (u,v) if we are unsure.
                # But that prevents exploration.
                
                # Let's assume that breaking one Q3 is enough? No.
                
                # Let's try this:
                # If we remove a random edge, we might still have a Q3.
                # To be safe, let's just remove (u,v) in most cases, and only try the swap if we can quickly verify.
                
                # Actually, let's just remove (u,v) for now, but occasionally try a swap and verify.
                
                if random.random() < 0.2:
                    # Try swap
                    # We already removed (rem_u, rem_v).
                    # Check if valid.
                    # This is too slow.
                    
                    # Let's just revert the swap and remove (u,v) instead.
                    add_edge(rem_u, rem_v)
                    remove_edge(u, v)
                else:
                    # Just remove (u,v)
                    remove_edge(u, v)
        else:
            # No Q3 found, keep the edge
            if len(edges) > best_edges:
                best_edges = len(edges)
                best_adj = [row[:] for row in adj]
                best_edge_set = set(edges)
                last_improvement_time = time.time()
        
    return [list(e) for e in best_edge_set]

if __name__ == "__main__":
    res = construct()
    print(len(res))
