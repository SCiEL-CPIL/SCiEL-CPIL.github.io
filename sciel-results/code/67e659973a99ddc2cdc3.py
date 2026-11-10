import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 20
    
    def time_ok():
        return time.time() < deadline

    # Total possible edges in K_20
    total_edges = n * (n - 1) // 2
    
    # Map each edge (u, v) with u < v to an index 0..199
    def edge_to_idx(u, v):
        if u > v:
            u, v = v, u
        return u * n - u * (u + 1) // 2 + (v - u - 1)
        
    # Precompute for each edge index, the set of other edges that conflict with it?
    # No, K3,3 involves 3 vertices on one side and 3 on the other. 
    # A violation is a set of 9 edges forming K3,3.
    # We want to find a subset of edges E' such that no K3,3 is fully contained in E'.
    # This is equivalent to finding a Maximum Independent Set in a 9-uniform hypergraph?
    # Or rather, we want to avoid any specific 9-edge set.
    
    # A more efficient constraint propagation approach for B&B:
    # We maintain a set of "forced" edges (must be in solution) and "forbidden" edges (cannot be in solution).
    # Actually, standard MIS on a graph is easier if we can reduce hypergraph to graph.
    # Can we reduce "no K3,3" to a graph conflict?
    # If we have 6 vertices {a,b,c,d,e,f}, the 9 edges forming K3,3 between {a,b,c} and {d,e,f} must not all be present.
    # This is a "not all" constraint.
    # This is a SAT-like problem.
    
    # However, for n=20, the number of K3,3 subgraphs is large.
    # Let's try a different structural approach: **Deterministic Local Search with Specific Swaps** is already in the seed.
    # The prompt asks for **Branch and Bound on Conflict Hypergraph**.
    
    # Let's implement a simple B&B for Maximum Independent Set on a Graph.
    # But our conflicts are hyperedges of size 9.
    # We can transform this: 
    # If we pick an edge e, we must remove enough other edges to break all K3,3s containing e.
    # This is complex.
    
    # Alternative: Use the fact that K3,3 is small.
    # Let's try a **Greedy Set Cover / Hitting Set** approach on the violations?
    # No, we want to MAXIMIZE edges, so we want to MINIMIZE the number of edges removed from the complete graph to destroy all K3,3s.
    # This is the **Minimum Hitting Set** problem for the family of all K3,3 subgraphs in K20.
    # We want to find a minimum set of edges S such that every K3,3 in K20 has at least one edge in S.
    # The answer is total_edges - |S|.
    
    # This is a standard hitting set problem.
    # We can solve it using Branch and Bound.
    
    # 1. Enumerate all K3,3 subgraphs in K20.
    # Number of ways to choose 3 vertices for left, 3 for right:
    # C(20, 6) * C(6, 3) / 2 ? No.
    # Choose 6 vertices: C(20, 6) = 38760.
    # Partition into two sets of 3: C(6, 3) / 2 = 10.
    # Total K3,3s = 38760 * 10 = 387,600.
    # Each K3,3 has 9 edges.
    # We have 190 edges.
    
    # We want to find min S subset of E such that for all K in K3,3s, S intersects E(K).
    
    # This is a classic Hitting Set.
    # B&B for Hitting Set:
    # State: (current_hitting_set, remaining_hypers, available_edges)
    # We can use a greedy upper bound: sum of 1/|hyper| for remaining? No.
    # Simple bound: If we have k remaining hyperedges, we need at least ceil(k / max_cover) more edges?
    # Or just: We need to hit all remaining. 
    # A simple bound: The number of remaining hyperedges is R. Each edge can hit at most M hyperedges.
    # So we need at least ceil(R/M) more edges.
    
    # Let's precompute:
    # 1. List of all K3,3 edge-sets (as bitmasks of edges).
    # 2. For each edge, the list of K3,3s it hits.
    
    # Bitmask of 190 bits fits in a Python integer (arbitrary precision).
    
    k33s = []
    # Precompute edge indices for all pairs
    edge_idx_map = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_map[(u, v)] = idx
            idx += 1
            
    # Generate all K3,3s
    # This might take a bit of time, but 387k is manageable.
    # Each K3,3 is a tuple of 9 edge indices.
    # We store them as a frozenset of edge indices? Or a bitmask?
    # Bitmask is faster for intersection checks.
    
    all_k33_bitmasks = []
    # To speed up generation, we can iterate over combinations of 6 vertices.
    # For each set of 6, iterate over partitions.
    
    # Optimization: We don't need to store all 387k masks if we can generate them on the fly?
    # But for B&B, we need to check if a candidate edge hits a hyperedge.
    # If we store the list of hyperedges, we can filter them as we add edges to the hitting set.
    
    # Let's generate the list of edge-tuples for all K3,3s.
    # 387,600 tuples of 9 ints. Memory is fine.
    
    # We also need for each edge, which K3,3s it belongs to.
    # edge_to_k33s = [[] for _ in range(total_edges)]
    
    # Generating all K3,3s might be slow in Python if not careful.
    # C(20,6) = 38760.
    # For each, 10 partitions.
    # Total 387,600.
    # Inside loop: get 9 edge indices.
    # This is ~4 million operations. Should be < 1s.
    
    k33_edge_lists = []
    for verts in itertools.combinations(range(n), 6):
        # Partition verts into two sets of 3
        # We need to avoid double counting (A,B) and (B,A).
        # Just pick the first 3 vs last 3? No, that misses partitions.
        # Iterate over all subsets of size 3 from the 6.
        # There are C(6,3)=10.
        # To avoid duplicates (S, V\S) and (V\S, S), we can just take all 10 and divide by 2?
        # Or enforce an order: e.g., min(S) < min(V\S).
        
        v_set = set(verts)
        seen_partitions = set()
        for left in itertools.combinations(verts, 3):
            left_set = frozenset(left)
            right_set = frozenset(v_set - left_set)
            # Canonical form: sort the two sets by their min element
            if min(left_set) < min(right_set):
                key = (left_set, right_set)
            else:
                key = (right_set, left_set)
                
            if key in seen_partitions:
                continue
            seen_partitions.add(key)
            
            l, r = key
            # Get edges between l and r
            edges = []
            for u in l:
                for v in r:
                    edges.append(edge_idx_map[(min(u,v), max(u,v))])
            k33_edge_lists.append(tuple(edges))
            
    # Now we have ~387k tuples.
    # For B&B, we represent the set of "active" K3,3s (those not yet hit) as a set of indices into k33_edge_lists?
    # Or better, we maintain a bitmask of active K3,3s? 387k bits is 48KB. Python int can handle it.
    # But checking if an edge hits a specific K3,3 requires knowing if the edge is in that tuple.
    
    # Let's invert: For each edge, list the indices of K3,3s that contain it.
    edge_to_k33_indices = [[] for _ in range(total_edges)]
    for i, edges in enumerate(k33_edge_lists):
        for e in edges:
            edge_to_k33_indices[e].append(i)
            
    # B&B State:
    # current_edges: list of edges chosen for hitting set
    # active_k33s: set of indices of K3,3s not yet hit
    # available_edges: set of edges not yet considered (can be added or skipped)
    
    # To make B&B efficient, we need a good branching strategy.
    # Branch on an edge e in active_k33s.
    # Option 1: Include e in hitting set.
    # Option 2: Exclude e. This means we must include ALL OTHER edges in every K3,3 containing e?
    # No, if we exclude e, we must ensure that every K3,3 containing e is hit by SOME OTHER edge.
    # This is complex to propagate directly.
    
    # Simpler B&B for Hitting Set (Minimum Set Cover variant):
    # We want to cover all K3,3s with edges.
    # This is Set Cover where Universe = K3,3s, Sets = {K3,3s containing edge e}.
    # We want min number of sets.
    
    # Standard B&B for Set Cover:
    # 1. If no active K3,3s, return 0.
    # 2. If time up, return infinity (or current best).
    # 3. Pick a K3,3 k that is hardest to cover (fewest available edges that can cover it).
    # 4. For each edge e that can cover k:
    #    - Recurse with e added to cover set, and all K3,3s covered by e removed from active.
    # 5. Return 1 + min(results).
    
    # Upper bound: Greedy Set Cover.
    # Lower bound: 
    #   - If max coverage of any edge is M, and we have R active K3,3s, we need at least ceil(R/M).
    #   - Also, if there are disjoint K3,3s, we need at least that many edges.
    
    # Let's implement this.
    
    # Precompute for each edge, the bitmask of K3,3s it covers.
    # 387k bits per edge. 190 edges.
    # Python ints are efficient for bit ops.
    
    edge_cover_masks = [0] * total_edges
    for e in range(total_edges):
        mask = 0
        for k_idx in edge_to_k33_indices[e]:
            mask |= (1 << k_idx)
        edge_cover_masks[e] = mask
            
    # Greedy Upper Bound
    def greedy_set_cover(active_mask, available_edges_set):
        count = 0
        current_active = active_mask
        # Copy available edges
        avail = list(available_edges_set)
        
        while current_active:
            if not time_ok():
                return float('inf')
            
            # Find edge that covers most active K3,3s
            best_edge = -1
            best_cover_count = 0
            for e in avail:
                cover = current_active & edge_cover_masks[e]
                cnt = cover.bit_count()
                if cnt > best_cover_count:
                    best_cover_count = cnt
                    best_edge = e
                    if best_cover_count == current_active.bit_count():
                        break # Perfect cover found
            
            if best_edge == -1:
                return float('inf') # Should not happen if active_mask is not 0 and edges exist
                
            count += 1
            current_active &= ~edge_cover_masks[best_edge]
            avail.remove(best_edge)
            
        return count

    # B&B
    best_solution_size = float('inf')
    best_solution_edges = []
    
    # Initial active mask: all K3,3s
    initial_active = (1 << len(k33_edge_lists)) - 1
    initial_available = set(range(total_edges))
    
    # Get initial upper bound
    ub = greedy_set_cover(initial_active, initial_available)
    if ub < best_solution_size:
        best_solution_size = ub
        # We don't track the actual edges for UB easily here without modifying greedy, 
        # but we just need the count to prune.
        # For the final answer, we need the actual set.
        # Let's modify greedy to return the set too, or just use B&B to find the exact set.
        # Since B&B will find the optimal, we just need the count for pruning.
        # We can re-run greedy to get the set if B&B finds a better one? 
        # Or just let B&B find the set.
        
    # To get the actual set from B&B, we track the path.
    
    def branch_and_bound(active_mask, available_edges_set, current_count, current_edges_list):
        nonlocal best_solution_size, best_solution_edges
        
        if not time_ok():
            return
            
        if current_count >= best_solution_size:
            return
            
        if active_mask == 0:
            if current_count < best_solution_size:
                best_solution_size = current_count
                best_solution_edges = current_edges_list[:]
            return
            
        # Lower Bound
        # 1. Max cover bound
        max_cover = 0
        for e in available_edges_set:
            c = (active_mask & edge_cover_masks[e]).bit_count()
            if c > max_cover:
                max_cover = c
        if max_cover == 0:
            return # Unsolvable with remaining edges? Shouldn't happen if active_mask != 0 and we started with full set.
            
        lb = (active_mask.bit_count() + max_cover - 1) // max_cover
        if current_count + lb >= best_solution_size:
            return
            
        # 2. Disjointness bound (optional, expensive to compute every time)
        # Skip for now, max_cover bound is decent.
        
        # Branching: Pick a K3,3 that has fewest covering edges available.
        # Iterate through bits of active_mask? 
        # We need to find a k in active_mask that minimizes |{e in avail | k in cover(e)}|.
        
        # This is slow if done naively.
        # Optimization: Precompute for each K3,3, the list of edges that cover it.
        # k33_to_edges = [list of edge indices]
        # We already have edge_to_k33_indices. We can invert it.
        
        # Let's precompute k33_to_edges outside.
        
        # To speed up B&B, we should precompute this.
        
        # For now, let's pick the first active K3,3? No, that's bad.
        # Let's pick one randomly? No, deterministic.
        # Let's pick the one with smallest degree in the current hypergraph.
        
        # Since we can't easily iterate active_mask bits and check degrees without precomputation,
        # let's precompute k33_to_edges.
        
        pass # Placeholder for logic structure
    
    # Precompute k33_to_edges
    k33_to_edges = [[] for _ in range(len(k33_edge_lists))]
    for e in range(total_edges):
        for k_idx in edge_to_k33_indices[e]:
            k33_to_edges[k_idx].append(e)
            
    # Refine B&B with k33_to_edges
    
    def branch_and_bound_refined(active_mask, available_edges_set, current_count, current_edges_list):
        nonlocal best_solution_size, best_solution_edges
        
        if not time_ok():
            return
            
        if current_count >= best_solution_size:
            return
            
        if active_mask == 0:
            if current_count < best_solution_size:
                best_solution_size = current_count
                best_solution_edges = current_edges_list[:]
            return
            
        # Lower Bound
        max_cover = 0
        for e in available_edges_set:
            c = (active_mask & edge_cover_masks[e]).bit_count()
            if c > max_cover:
                max_cover = c
        if max_cover == 0:
            return
            
        lb = (active_mask.bit_count() + max_cover - 1) // max_cover
        if current_count + lb >= best_solution_size:
            return
            
        # Find branching K3,3
        # We need to find a k such that (1 << k) is in active_mask
        # and len([e for e in k33_to_edges[k] if e in available_edges_set]) is minimized.
        
        # Iterating all k in active_mask is slow.
        # We can iterate k from 0 to N-1 and check if bit is set.
        # N is 387k. This is too slow for deep recursion.
        
        # Alternative: Maintain a list of active K3,3s?
        # Or just pick the first one found?
        # Let's try picking the first active K3,3. It might be suboptimal but fast.
        
        # Find first set bit in active_mask
        # active_mask is a Python int.
        # lsb = active_mask & -active_mask
        # k_idx = lsb.bit_length() - 1
        
        k_idx = (active_mask & -active_mask).bit_length() - 1
        
        # Get available edges for this k
        candidates = [e for e in k33_to_edges[k_idx] if e in available_edges_set]
        
        if not candidates:
            return # Dead end
            
        # Order candidates by how many OTHER active K3,3s they cover (descending)
        # This helps prune.
        def score_edge(e):
            return (active_mask & edge_cover_masks[e]).bit_count()
            
        candidates.sort(key=score_edge, reverse=True)
        
        for e in candidates:
            new_active = active_mask & ~edge_cover_masks[e]
            new_avail = available_edges_set - {e}
            new_count = current_count + 1
            new_list = current_edges_list + [e]
            
            branch_and_bound_refined(new_active, new_avail, new_count, new_list)
            
            if not time_ok():
                return

    # Start B&B
    # Recursion depth could be up to 190. Python default is 1000. OK.
    # But branching factor is high.
    # The "first bit" branching is weak. 
    # Let's try to improve the branching variable selection.
    # We can't easily find the min-degree active node without scanning all 387k.
    # However, we can approximate.
    
    # Let's run it.
    branch_and_bound_refined(initial_active, initial_available, 0, [])
    
    # If B&B didn't find anything (or timed out), fallback to greedy solution
    if best_solution_size == float('inf'):
        # Re-run greedy to get the actual set
        def greedy_set_cover_with_set(active_mask, available_edges_set):
            count = 0
            current_active = active_mask
            avail = list(available_edges_set)
            chosen = []
            
            while current_active:
                if not time_ok():
                    break
                best_edge = -1
                best_cover_count = 0
                for e in avail:
                    cover = current_active & edge_cover_masks[e]
                    cnt = cover.bit_count()
                    if cnt > best_cover_count:
                        best_cover_count = cnt
                        best_edge = e
                        if best_cover_count == current_active.bit_count():
                            break
                if best_edge == -1:
                    break
                chosen.append(best_edge)
                current_active &= ~edge_cover_masks[best_edge]
                avail.remove(best_edge)
            return chosen
            
        best_solution_edges = greedy_set_cover_with_set(initial_active, initial_available)
        
    # Construct the graph
    # The hitting set best_solution_edges are the edges to REMOVE.
    # The graph is K20 minus these edges.
    
    remove_set = set(best_solution_edges)
    
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            idx = edge_idx_map[(u, v)]
            if idx not in remove_set:
                edge_list.append([u, v])
                
    return edge_list

