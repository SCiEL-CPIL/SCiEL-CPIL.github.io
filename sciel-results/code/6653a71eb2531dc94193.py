import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)
        
        # Internal C edges
        internal_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                if v in adj[u]:
                    internal_edges.add((u, v))
        
        # Edges from I to C
        i_c_edges = {}
        for v in I:
            nbrs = adj[v] & C_set
            i_c_edges[v] = nbrs
        
        # Group I vertices by their exact neighborhood in C
        groups = defaultdict(list)
        for v in I:
            if i_c_edges[v]:
                groups[frozenset(i_c_edges[v])].append(v)
        
        # Sort groups by shadow size descending
        sorted_groups = sorted(groups.items(), key=lambda x: -len(x[0]))
        
        best_partition = None
        best_count = float('inf')
        
        # Strategy: 
        # 1. Greedily form large cliques covering I vertices and their shadows.
        # 2. For remaining C-C edges, use a "Maximal Clique" heuristic based on 
        #    common neighbors, trying to pack as many edges into each clique as possible.
        #    This is a structural change from simple greedy edge-picking to a 
        #    clique-expansion approach that respects the chordal nature better 
        #    by finding large independent sets in the complement (cliques in G).
        
        # We will run a few variations of the heuristic to find the best partition.
        
        for strategy_idx in range(3):
            if time.time() > deadline - 10.0:
                break
                
            partition = []
            used_c_edges = set()
            
            # Process I groups
            # Order can be varied, but for now we stick to descending shadow size 
            # as it tends to cover more edges per clique.
            
            group_order = sorted_groups
            if strategy_idx == 1:
                # Try ascending? Might be risky but worth a shot if time permits.
                # Actually, descending is usually better for "covering" because 
                # large shadows are harder to cover with small cliques.
                group_order = sorted(sorted_groups, key=lambda x: len(x[0]))
            elif strategy_idx == 2:
                # Randomized ordering based on a seed?
                # Let's just stick to deterministic for now to avoid noise, 
                # but maybe shuffle slightly? No, keep it simple.
                pass
            
            for shadow, i_vertices in group_order:
                if not i_vertices:
                    continue
                
                # Find a large clique within shadow using unused C-C edges
                available = list(shadow)
                
                if not available:
                    continue
                
                # Heuristic: Find a maximal clique in the induced subgraph on 'available'
                # that uses only unused edges.
                # We use a greedy approach: pick vertex with max degree in the 
                # "available" subgraph (considering only unused edges).
                
                def get_unused_degree(u, avail_set):
                    cnt = 0
                    for w in avail_set:
                        if u != w:
                            e = (min(u, w), max(u, w))
                            if e in internal_edges and e not in used_c_edges:
                                cnt += 1
                    return cnt
                
                # Sort available by unused degree descending
                available.sort(key=lambda u: -get_unused_degree(u, set(available)))
                
                clique = []
                for u in available:
                    ok = True
                    for w in clique:
                        e = (min(u, w), max(u, w))
                        if e not in internal_edges or e in used_c_edges:
                            ok = False
                            break
                    if ok:
                        clique.append(u)
                
                if len(clique) >= 1:
                    new_clique = sorted(i_vertices + clique)
                    partition.append(new_clique)
                    # Mark C-C edges as used
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            used_c_edges.add((min(clique[i], clique[j]), max(clique[i], clique[j])))
                
                # Cover remaining cross edges for this group's I vertices
                # For each I vertex v, edges to C vertices not in clique
                # Wait, the previous code added individual edges for remaining neighbors.
                # This is inefficient. We should try to group them if possible?
                # But since I vertices in the same group have the SAME shadow,
                # and we already took a clique from the shadow, the remaining neighbors
                # are disjoint from the clique. 
                # Can we form cliques with multiple I vertices and remaining C vertices?
                # Yes! If we have multiple I vertices in the group, and multiple remaining C vertices
                # that form a clique among themselves AND are connected to all selected I vertices,
                # we can form a larger clique.
                
                # Let's refine this part.
                # We have `i_vertices` and `remaining_shadow = shadow - set(clique)`.
                # We want to cover edges between `i_vertices` and `remaining_shadow`.
                # Since all `i_vertices` are connected to all `remaining_shadow` (by definition of shadow),
                # we just need to cover the edges within `remaining_shadow` (C-C edges) and 
                # the "I-C" edges.
                # Actually, the "I-C" edges are covered if we put any I vertex and its neighbor C vertex in a clique.
                # If we form a clique {i1, i2, c1, c2}, it covers (i1,c1), (i1,c2), (i2,c1), (i2,c2) and (c1,c2).
                # So we should try to pack multiple I vertices and multiple C vertices from `remaining_shadow`
                # into cliques, provided the C vertices form a clique in the unused graph.
                
                rem_i = list(i_vertices)
                rem_c = list(shadow - set(clique))
                
                while rem_i and rem_c:
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Find a clique in rem_c using unused edges
                    avail_c = rem_c
                    avail_c.sort(key=lambda u: -get_unused_degree(u, set(avail_c)))
                    
                    c_clique = []
                    for u in avail_c:
                        ok = True
                        for w in c_clique:
                            e = (min(u, w), max(u, w))
                            if e not in internal_edges or e in used_c_edges:
                                ok = False
                                break
                        if ok:
                            c_clique.append(u)
                    
                    if not c_clique:
                        # No C-C edges available in rem_c? Or no edges at all?
                        # If no edges, we can still form cliques of size 2: {i, c}
                        # But we need to cover all edges.
                        # If c_clique is empty, it means no unused C-C edges among rem_c.
                        # We can still pair each i with each c?
                        # To minimize cliques, we should group as many i's and c's as possible.
                        # If there are no C-C edges, we cannot have a clique with >1 C vertices.
                        # So max clique size is 1 C + many I's?
                        # Wait, if no C-C edges, we can't have 2 C's in a clique.
                        # So we can have {i1, i2, ..., im, c1}. This covers all edges from i's to c1.
                        # We need to cover edges to all c's in rem_c.
                        # So we would need one clique per c?
                        # {i1...im, c1}, {i1...im, c2}, ...
                        # This uses len(rem_c) cliques.
                        # Alternatively, {i1, c1}, {i1, c2}... uses more.
                        # So grouping all I's with one C is best.
                        
                        # Let's just take the first c
                        c_clique = [rem_c[0]]
                        
                    # How many I's can we include?
                    # All remaining I's are connected to all C's in c_clique (since they are in shadow).
                    # So we can include ALL remaining I's in this clique.
                    new_clique = sorted(rem_i + c_clique)
                    partition.append(new_clique)
                    
                    # Mark C-C edges in c_clique as used
                    for i_idx in range(len(c_clique)):
                        for j_idx in range(i_idx + 1, len(c_clique)):
                            used_c_edges.add((min(c_clique[i_idx], c_clique[j_idx]), max(c_clique[i_idx], c_clique[j_idx])))
                    
                    # Remove used C's from rem_c
                    rem_c = [c for c in rem_c if c not in c_clique]
                    # Remove all I's (since they are covered by this clique for ALL their neighbors? 
                    # NO. The clique {i1..im, c1..ck} covers edges (i, c) for i in I, c in C_clique.
                    # It does NOT cover edges (i, c') for c' not in C_clique.
                    # So we cannot remove I's unless we've covered ALL their neighbors.
                    # My previous logic "rem_i = []" was wrong if we don't cover all rem_c.
                    
                    # Correction: 
                    # We covered edges between rem_i and c_clique.
                    # We still need to cover edges between rem_i and (rem_c - c_clique).
                    # So we keep rem_i.
                    # But we remove c_clique from rem_c.
                    
                    # Wait, if we keep rem_i, we will form more cliques.
                    # Is it better to split I's?
                    # If we have I={i1, i2} and C={c1, c2} with no edge (c1,c2).
                    # Option 1: {i1, i2, c1}, {i1, i2, c2}. (2 cliques). Covers all 4 edges.
                    # Option 2: {i1, c1}, {i1, c2}, {i2, c1}, {i2, c2}. (4 cliques).
                    # Option 1 is better.
                    # So keeping all I's together is good.
                    
                # After this loop, rem_c should be empty (we always pick at least 1 c).
                # rem_i might still have elements if we ran out of time or something, 
                # but logically rem_c becomes empty.
                # Wait, if rem_c becomes empty, the loop terminates.
                # All edges from i_vertices to shadow are covered?
                # Yes, because we partitioned shadow into c_cliques.
                # Each i is in every clique formed with its group's I's?
                # No, I kept rem_i constant.
                # So in each iteration, we formed a clique with ALL current rem_i and a subset of rem_c.
                # This covers all edges from rem_i to that subset of rem_c.
                # Since we eventually empty rem_c, all edges are covered.
                # And all I vertices are in all these cliques.
                # This is valid.

            # Cover remaining C-C edges
            rem_c_edges = internal_edges - used_c_edges
            rem_set = set(rem_c_edges)
            
            while rem_set:
                if time.time() > deadline - 3.0:
                    break
                
                # Find a maximal clique in the remaining graph
                # Heuristic: pick edge with max common neighbors
                best_edge = None
                max_common = -1
                for (a, b) in rem_set:
                    common = (adj[a] & adj[b]) & set(range(k))
                    # Filter to only remaining edges? 
                    # Common neighbors must be connected to both a and b in the REMAINING graph.
                    # So we check if edges (c,a) and (c,b) are in rem_set.
                    valid_common = 0
                    for c in common:
                        if (min(a,c), max(a,c)) in rem_set and (min(b,c), max(b,c)) in rem_set:
                            valid_common += 1
                    
                    if valid_common > max_common:
                        max_common = valid_common
                        best_edge = (a, b)
                
                if best_edge is None:
                    # No edges? Shouldn't happen if rem_set is not empty.
                    break
                
                a, b = best_edge
                clique = [a, b]
                
                # Expand clique
                # Candidates are common neighbors of clique in remaining graph
                candidates = []
                for c in range(k):
                    if c in clique:
                        continue
                    ok = True
                    for u in clique:
                        if (min(u,c), max(u,c)) not in rem_set:
                            ok = False
                            break
                    if ok:
                        candidates.append(c)
                
                # Sort candidates by degree in remaining graph?
                def rem_degree(u):
                    cnt = 0
                    for w in range(k):
                        if u != w and (min(u,w), max(u,w)) in rem_set:
                            cnt += 1
                    return cnt
                
                candidates.sort(key=lambda u: -rem_degree(u))
                
                for c in candidates:
                    # Check if c can be added (it should, based on candidate generation, 
                    # but we must ensure it connects to ALL current clique members)
                    # The candidate generation above ensures this.
                    clique.append(c)
                
                if len(clique) >= 2:
                    partition.append(sorted(clique))
                    # Remove edges covered by this clique
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            rem_set.discard((min(clique[i], clique[j]), max(clique[i], clique[j])))
            
            cnt = len(partition)
            if cnt < best_count:
                best_count = cnt
                best_partition = partition
        
        if best_partition is None:
            return []
        
        cleaned = []
        for clique in best_partition:
            cleaned.append([int(x) for x in clique])
        return cleaned

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        results.append(res)

    return results
