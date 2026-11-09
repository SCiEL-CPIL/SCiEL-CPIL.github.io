import time
import sys
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency masks for fast checks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute edge set for O(1) lookup
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
            
    # Helper to check if a set of vertices forms a biclique [L, R]
    # L and R are lists of ints
    def is_biclique(L, R):
        if not L or not R:
            return False
        # Check all cross edges exist
        for u in L:
            mask_u = adj[u]
            for v in R:
                if not (mask_u & (1 << v)):
                    return False
        return True

    # Helper to check if a partition is valid
    # partition: list of [L, R]
    def is_valid_partition(partition):
        covered = set()
        for L, R in partition:
            if not L or not R:
                return False
            for u in L:
                for v in R:
                    e = (u, v) if u < v else (v, u)
                    if e in covered:
                        return False
                    covered.add(e)
                    if e not in edge_set:
                        return False
        if len(covered) != len(edge_set):
            return False
        return True

    # Initial solution: Greedy random
    def get_initial_solution():
        remaining_edges = list(edge_set)
        random.shuffle(remaining_edges)
        partition = []
        # Simple greedy: pick an edge, try to expand it to a maximal biclique
        while remaining_edges:
            # Pick a random remaining edge
            idx = random.randint(0, len(remaining_edges) - 1)
            u, v = remaining_edges[idx]
            
            # Start with this edge
            L = [u]
            R = [v]
            
            # Try to add other vertices to L or R
            # We will greedily add vertices that are connected to all in the other side
            changed = True
            while changed:
                changed = False
                # Try adding to L
                for w in range(n):
                    if w in L or w in R:
                        continue
                    # Check if w is connected to all in R
                    ok = True
                    mask_w = adj[w]
                    for r in R:
                        if not (mask_w & (1 << r)):
                            ok = False
                            break
                    if ok:
                        L.append(w)
                        changed = True
                        break # Restart scan to see if more can be added
                if changed:
                    continue
                # Try adding to R
                for w in range(n):
                    if w in L or w in R:
                        continue
                    # Check if w is connected to all in L
                    ok = True
                    mask_w = adj[w]
                    for l in L:
                        if not (mask_w & (1 << l)):
                            ok = False
                            break
                    if ok:
                        R.append(w)
                        changed = True
                        break
            partition.append([L, R])
            
            # Remove covered edges
            covered_in_this = set()
            for l in L:
                for r in R:
                    e = (l, r) if l < r else (r, l)
                    covered_in_this.add(e)
                    
            remaining_edges = [e for e in remaining_edges if e not in covered_in_this]
            
        return partition

    # Local Search Move: Move a vertex from one biclique to another
    # This is complex to implement generally. 
    # A simpler and effective move for biclique cover/partition is "Edge Swap" or "Vertex Swap".
    # However, since we must partition edges, moving a vertex changes the set of covered edges significantly.
    
    # Let's use a different strategy: "Re-optimization of a random subset of bicliques"
    # Or "Simulated Annealing on the partition structure"
    
    # Given the small size (30 nodes, ~150 edges), we can afford a more expensive validity check.
    
    # Let's try a "Hill Climbing" approach where we try to merge or split bicliques.
    # Actually, the most effective simple heuristic for small graphs is often:
    # 1. Start with a valid partition.
    # 2. Try to remove a biclique and re-partition its edges using other bicliques or new ones.
    # 3. Try to merge two bicliques if possible.
    
    # Let's implement a robust Iterated Local Search:
    # 1. Generate a random valid partition (by randomly assigning edges to bicliques and merging where possible).
    # 2. Apply local improvements (Merge, Split, Move).
    # 3. If stuck, perturb (randomly shuffle some edges/bicliques) and re-optimize.
    
    def random_valid_partition():
        # Assign each edge to a random "bucket"
        # Then for each bucket, find the maximum biclique covering those edges?
        # No, that's hard.
        
        # Easier: Start with each edge as a biclique [u, v].
        # Then randomly merge bicliques if the union is a biclique.
        partition = []
        for u, v in edges:
            partition.append([[u], [v]])
        
        # Randomly shuffle
        random.shuffle(partition)
        
        # Try to merge
        improved = True
        while improved:
            improved = False
            for i in range(len(partition)):
                if improved: break
                for j in range(i + 1, len(partition)):
                    L1, R1 = partition[i]
                    L2, R2 = partition[j]
                    # Try merging
                    # Check if L1 U L2 and R1 U R2 is a biclique
                    # But we must ensure no edge duplication.
                    # Since they are disjoint sets of edges, if the union of vertices forms a biclique,
                    # it's valid IF all cross edges exist.
                    # Wait, if L1 and R2 have an edge that is NOT in the graph, the merge is invalid.
                    # If L1 and R2 have an edge that IS in the graph but was assigned to a different biclique, 
                    # merging would duplicate it? No, the edges are defined by the vertex sets.
                    # The condition is: The set of edges in the merged biclique is exactly the union of edges in b1 and b2.
                    # This implies:
                    # 1. All edges between L1 and R2 must exist in the graph.
                    # 2. All edges between L2 and R1 must exist in the graph.
                    # 3. No edge between L1 and R2 or L2 and R1 can be "missing" (which would mean the biclique is invalid).
                    # 4. No edge between L1 and R2 or L2 and R1 can be "extra" (i.e., not in b1 or b2).
                    #    But since b1 and b2 are valid bicliques, all edges in L1xR1 and L2xR2 are present.
                    #    The potential problem is if L1xR2 contains an edge that is NOT in b1 or b2.
                    #    But if L1xR2 is a complete bipartite subgraph in G, and we merge, the new biclique covers L1xR1 U L2xR2 U L1xR2 U L2xR1.
                    #    For this to be a valid partition step, the new set of edges must be exactly the union of the old sets.
                    #    This requires that L1xR2 and L2xR1 contain NO edges other than those already in b1 or b2?
                    #    No, that's impossible unless L1xR2 is empty or subset of existing edges.
                    #    Actually, if L1 and R2 are connected, those edges are "new" to the partition set unless they were already covered.
                    #    But b1 and b2 are disjoint in edges. So if an edge exists in L1xR2, it was NOT covered by b1 or b2.
                    #    Therefore, merging b1 and b2 would cover an edge that was previously uncovered (or covered by a third biclique).
                    #    This violates the "partition" property if that edge is covered by another biclique.
                    #    So, merging is ONLY valid if L1xR2 and L2xR1 contain NO edges from G.
                    #    Wait, if they contain edges from G, those edges must be uncovered? No, all edges are covered by the partition.
                    #    So if L1xR2 has an edge, it must be in some biclique. If it's not in b1 or b2, it's in b3.
                    #    Merging b1 and b2 would "steal" that edge? No, the biclique definition covers ALL edges between L and R.
                    #    So if L1xR2 has an edge, the merged biclique covers it.
                    #    If that edge was also in b3, we have a conflict (duplicate coverage).
                    #    Therefore, a merge is valid ONLY IF there are NO edges between L1 and R2, and NO edges between L2 and R1.
                    #    In that case, the union L1 U L2, R1 U R2 is a biclique (since L1-R1, L2-R2 are complete, and cross is empty).
                    #    And the edge sets are disjoint.
                    
                    # Check if any edge exists between L1 and R2
                    conflict = False
                    for u in L1:
                        for v in R2:
                            if (u, v) in edge_set or (v, u) in edge_set:
                                conflict = True
                                break
                        if conflict: break
                    if not conflict:
                        for u in L2:
                            for v in R1:
                                if (u, v) in edge_set or (v, u) in edge_set:
                                    conflict = True
                                    break
                            if conflict: break
                    
                    if not conflict:
                        # Merge
                        new_L = list(set(L1 + L2))
                        new_R = list(set(R1 + R2))
                        # Ensure no vertex is in both L and R of the new biclique?
                        # The problem says "nonempty disjoint sides".
                        # If a vertex was in L1 and R2, it would be in both new_L and new_R.
                        # But L1 and R2 are disjoint sets of vertices?
                        # In a biclique [L, R], L and R are disjoint.
                        # In b1, L1 and R1 are disjoint. In b2, L2 and R2 are disjoint.
                        # Is it possible that a vertex is in L1 and R2?
                        # If so, there is an edge (u, v) in b1 where u in L1, v in R1.
                        # And an edge (x, y) in b2 where x in L2, y in R2.
                        # If u (in L1) is also in R2, then u is connected to all in L2 (since u in R2).
                        # So (u, x) is an edge.
                        # Is (u, x) in b1 or b2?
                        # u in L1, x in L2. L1 and L2 are on the same side? No, they are from different bicliques.
                        # This is getting complicated.
                        # Standard biclique partition usually allows vertices to appear in multiple bicliques on different sides.
                        # But within a single biclique, L and R must be disjoint.
                        # So if we merge, we must ensure new_L and new_R are disjoint.
                        if not set(new_L).intersection(new_R):
                            partition.pop(j)
                            partition[i] = [new_L, new_R]
                            improved = True
                            
        return partition

    # The "No cross edges" merge condition is very restrictive.
    # A better strategy:
    # Represent the solution as an assignment of edges to bicliques.
    # Start with a greedy solution.
    # Then perform "Local Search":
    # 1. Try to move a vertex from one side of a biclique to the other side (if valid).
    # 2. Try to move a vertex from one biclique to another.
    # 3. Try to merge two bicliques (relaxed condition: just check if union is a biclique and edge sets are disjoint).
    
    # Let's use a simpler, more robust approach for small N:
    # 1. Generate a random partition by assigning each edge to a random biclique index.
    # 2. For each biclique index, extract the vertices and check if it's a valid biclique.
    #    If not, it's an invalid state. We need to maintain validity.
    
    # Given the complexity of maintaining validity during moves, and the small size (N=30),
    # we can use a "Restart" strategy with a strong local search.
    
    # Local Search:
    # State: List of bicliques [L, R].
    # Move 1: Merge. If bicliques i and j can be merged (union is biclique, edge sets disjoint), do it.
    # Move 2: Split. If a biclique can be split into two valid bicliques that cover the same edges, do it? (Usually increases count, so we avoid unless necessary to enable a merge).
    # Move 3: "Swap". Move an edge from biclique A to biclique B if B can absorb it.
    
    # Let's stick to the "Merge" local search but improve the initial solution.
    # The current program's weakness is the initial greedy might be poor.
    # We will use "Randomized Greedy" with multiple restarts and keep the best.
    
    def randomized_greedy():
        remaining_edges = list(edge_set)
        random.shuffle(remaining_edges)
        partition = []
        
        while remaining_edges:
            # Pick a random edge from remaining
            idx = random.randint(0, len(remaining_edges) - 1)
            u, v = remaining_edges[idx]
            
            # Try to build a large biclique containing (u, v)
            # Start with L=[u], R=[v]
            L = [u]
            R = [v]
            
            # Greedily add vertices
            # We want to maximize the number of edges covered.
            # A vertex w can be added to L if it is connected to all in R.
            # A vertex w can be added to R if it is connected to all in L.
            
            # We will iterate until no more vertices can be added.
            # To avoid bias, we try adding in random order.
            vertices_to_try = list(range(n))
            random.shuffle(vertices_to_try)
            
            changed = True
            while changed:
                changed = False
                for w in vertices_to_try:
                    if w in L or w in R:
                        continue
                    # Try adding to L
                    mask_w = adj[w]
                    can_add_L = True
                    for r in R:
                        if not (mask_w & (1 << r)):
                            can_add_L = False
                            break
                    if can_add_L:
                        L.append(w)
                        changed = True
                        break # Restart
                    # Try adding to R
                    can_add_R = True
                    for l in L:
                        if not (mask_w & (1 << l)):
                            can_add_R = False
                            break
                    if can_add_R:
                        R.append(w)
                        changed = True
                        break
            
            partition.append([L, R])
            
            # Remove covered edges
            covered = set()
            for l in L:
                for r in R:
                    e = (l, r) if l < r else (r, l)
                    covered.add(e)
            remaining_edges = [e for e in remaining_edges if e not in covered]
            
        return partition

    def local_search(partition):
        # Try to merge bicliques
        improved = True
        while improved:
            improved = False
            for i in range(len(partition)):
                if improved: break
                for j in range(i + 1, len(partition)):
                    L1, R1 = partition[i]
                    L2, R2 = partition[j]
                    
                    # Check if union is a valid biclique
                    # 1. L1 U L2 and R1 U R2 must be disjoint
                    if set(L1 + L2).intersection(set(R1 + R2)):
                        continue
                    
                    # 2. All cross edges must exist
                    # Check L1 vs R2
                    valid = True
                    for u in L1:
                        for v in R2:
                            if not (adj[u] & (1 << v)):
                                valid = False
                                break
                        if not valid: break
                    if valid:
                        for u in L2:
                            for v in R1:
                                if not (adj[u] & (1 << v)):
                                    valid = False
                                    break
                            if not valid: break
                    
                    if not valid:
                        continue
                    
                    # 3. Edge sets must be disjoint (no overlap)
                    # Since L1/R1 and L2/R2 are disjoint sets of edges in the partition,
                    # the only overlap can come from the cross terms L1-R2 and L2-R1.
                    # If those edges exist in the graph, they are covered by the new biclique.
                    # But were they covered by the old bicliques? No, because L1 and R2 were in different bicliques.
                    # So if an edge exists in L1-R2, it was NOT covered by b1 or b2.
                    # But it MUST be covered by some biclique in the partition (validity).
                    # If it's covered by b3, then merging b1 and b2 would create a duplicate coverage.
                    # Therefore, for a merge to be valid in a partition,
                    # there must be NO edges between L1 and R2, and NO edges between L2 and R1.
                    # Wait, this logic holds if the partition is exact.
                    # If there is an edge (u,v) with u in L1, v in R2, it is covered by some biclique.
                    # If it's not b1 or b2, it's b3.
                    # Merging b1 and b2 covers it again. Invalid.
                    # So, merge is valid ONLY IF no such edges exist.
                    
                    # Check for existence of edges in L1-R2
                    has_edge = False
                    for u in L1:
                        for v in R2:
                            if (u, v) in edge_set or (v, u) in edge_set:
                                has_edge = True
                                break
                        if has_edge: break
                    if has_edge:
                        continue
                        
                    has_edge = False
                    for u in L2:
                        for v in R1:
                            if (u, v) in edge_set or (v, u) in edge_set:
                                has_edge = True
                                break
                        if has_edge: break
                    if has_edge:
                        continue
                    
                    # If we reach here, the merge is valid and reduces count by 1.
                    new_L = list(set(L1 + L2))
                    new_R = list(set(R1 + R2))
                    partition.pop(j)
                    partition[i] = [new_L, new_R]
                    improved = True
                    
        return partition

    # Main Loop: Iterated Local Search
    start_time = time.time()
    deadline = start_time + 55 # 5s margin
    
    best_partition = None
    best_count = float('inf')
    
    # Initial best
    p = randomized_greedy()
    p = local_search(p)
    best_partition = p
    best_count = len(p)
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Perturb the best solution or generate new random
        if restarts % 10 == 0:
            # Full restart
            p = randomized_greedy()
        else:
            # Perturb best: randomly swap a few edges?
            # Simpler: just generate a new random solution to explore different basins.
            # Since N is small, randomized_greedy is fast.
            p = randomized_greedy()
            
        p = local_search(p)
        
        if len(p) < best_count:
            best_count = len(p)
            best_partition = p
            
        # Early stop if we found a perfect biclique (count 1)?
        # Or if count is very low?
        if best_count == 1:
            break

    # Final local search on the best found
    best_partition = local_search(best_partition)
    
    # Format output
    final_result = []
    for bic in best_partition:
        l = [int(x) for x in bic[0]]
        r = [int(x) for x in bic[1]]
        final_result.append([l, r])
        
    return final_result

if __name__ == "__main__":
    construct()
