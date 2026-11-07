"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_host_adj(n, edges):
    host_adj = [set() for _ in range(n)]
    for u, v in edges:
        host_adj[u].add(v)
        host_adj[v].add(u)
    return host_adj

def _expand_biclique(start_v, remaining_edges, n, max_iter=200):
    """Expand a biclique starting from vertex start_v.
    Returns (L, R, removed_count) where L and R are sets of vertices."""
    # Find neighbors of start_v in remaining graph
    R = set()
    for e in remaining_edges:
        if start_v in e:
            other = list(e)[0] if list(e)[0] != start_v else list(e)[1]
            R.add(other)
    
    if not R:
        return set(), set(), 0
    
    L = {start_v}
    
    # Build adjacency in remaining graph for fast lookups
    rem_adj = [set() for _ in range(n)]
    for e in remaining_edges:
        u, v = list(e)
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    
    changed = True
    iter_count = 0
    while changed and iter_count < max_iter:
        changed = False
        iter_count += 1
        
        # Expand L: find w not in L∪R such that R ⊆ rem_adj[w]
        best_w_L = None
        best_score_L = -1
        for w in range(n):
            if w in L or w in R:
                continue
            if R.issubset(rem_adj[w]):
                # Score by how many remaining edges this vertex covers
                score = sum(1 for r in R if frozenset((w, r)) in remaining_edges)
                if score > best_score_L:
                    best_score_L = score
                    best_w_L = w
        if best_w_L is not None:
            L.add(best_w_L)
            changed = True
        
        # Expand R: find w not in L∪R such that L ⊆ rem_adj[w]
        best_w_R = None
        best_score_R = -1
        for w in range(n):
            if w in L or w in R:
                continue
            if L.issubset(rem_adj[w]):
                # Score by how many remaining edges this vertex covers
                score = sum(1 for l in L if frozenset((l, w)) in remaining_edges)
                if score > best_score_R:
                    best_score_R = score
                    best_w_R = w
        if best_w_R is not None:
            R.add(best_w_R)
            changed = True
    
    # Count removed edges
    removed_count = 0
    for l in L:
        for r in R:
            if frozenset((l, r)) in remaining_edges:
                removed_count += 1
    
    return L, R, removed_count

def _shrink_biclique(L, R, remaining_edges, n, host_adj):
    """Try to shrink L or R to remove vertices that don't contribute unique edges.
    A vertex in L is redundant if all its cross-edges to R are also covered by other L vertices."""
    if not L or not R:
        return L, R
    
    L_set = set(L)
    R_set = set(R)
    
    # For each vertex in L, check if removing it still covers all its edges via other L vertices
    # Actually, we want to remove vertices that have few unique edges
    # A vertex l in L has unique edges to r in R where no other l' in L is connected to r
    # But that's not quite right. We want to minimize |L| + |R| while keeping all cross edges.
    
    # Simpler approach: try removing each vertex and see if the biclique is still valid
    # and covers the same edges
    
    changed = True
    while changed:
        changed = False
        
        # Try removing from L
        for l in list(L_set):
            if len(L_set) <= 1:
                break
            # Check if l's edges to R are all covered by other L vertices
            # i.e., for each r in R, is there l' in L\{l} with (l',r) in remaining_edges?
            redundant = True
            for r in R_set:
                edge = frozenset((l, r))
                if edge not in remaining_edges:
                    continue  # This edge isn't in remaining, so l doesn't "cover" it
                # Check if another L vertex covers this edge
                covered_by_other = False
                for l2 in L_set:
                    if l2 == l:
                        continue
                    if frozenset((l2, r)) in remaining_edges:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            if redundant:
                L_set.remove(l)
                changed = True
        
        # Try removing from R
        for r in list(R_set):
            if len(R_set) <= 1:
                break
            redundant = True
            for l in L_set:
                edge = frozenset((l, r))
                if edge not in remaining_edges:
                    continue
                covered_by_other = False
                for r2 in R_set:
                    if r2 == r:
                        continue
                    if frozenset((l, r2)) in remaining_edges:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            if redundant:
                R_set.remove(r)
                changed = True
    
    return L_set, R_set

def _try_merge_bicliques(bicliques, host_adj, n, all_edges_set):
    """Try to merge compatible bicliques. Returns merged list."""
    if len(bicliques) <= 1:
        return bicliques
    
    merged = True
    while merged:
        merged = False
        new_bicliques = []
        i = 0
        while i < len(bicliques):
            if i >= len(bicliques):
                break
            L1, R1 = set(bicliques[i][0]), set(bicliques[i][1])
            j = i + 1
            found_merge = False
            while j < len(bicliques):
                L2, R2 = set(bicliques[j][0]), set(bicliques[j][1])
                
                # Try merging: new_L = L1 ∪ L2, new_R = R1 ∪ R2
                new_L = L1 | L2
                new_R = R1 | R2
                
                if new_L & new_R:
                    j += 1
                    continue
                
                # Check if all cross pairs are host edges
                valid = True
                for l in new_L:
                    for r in new_R:
                        if r not in host_adj[l]:
                            valid = False
                            break
                    if not valid:
                        break
                
                # Check that the union of edges is exactly the union of both bicliques' edges
                if valid:
                    edges1 = set()
                    for l in L1:
                        for r in R1:
                            edges1.add(frozenset((l, r)))
                    edges2 = set()
                    for l in L2:
                        for r in R2:
                            edges2.add(frozenset((l, r)))
                    union_edges = edges1 | edges2
                    # Verify union_edges are all in all_edges_set
                    if union_edges.issubset(all_edges_set):
                        new_bicliques.append([sorted(new_L), sorted(new_R)])
                        merged = True
                        found_merge = True
                        break
                j += 1
            
            if not found_merge:
                new_bicliques.append(bicliques[i])
            i += 1
        
        bicliques = new_bicliques
    
    return bicliques

def _validate_partition(bicliques, all_edges_set, n):
    """Validate that bicliques form a valid partition of all edges."""
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for l in left:
            for r in right:
                e = frozenset((l, r))
                if e in covered:
                    return False
                covered.add(e)
                if e not in all_edges_set:
                    return False
    return covered == all_edges_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    
    # Build static adjacency sets for host graph
    host_adj = _build_host_adj(n, P['edges'])
    
    # Initialize remaining edges as a set of frozensets
    all_edges_set = set()
    for u, v in P['edges']:
        if u < v:
            all_edges_set.add(frozenset((u, v)))
        else:
            all_edges_set.add(frozenset((v, u)))
    
    best_solution = None
    best_count = float('inf')
    
    # Multiple attempts with different random seeds for candidate ordering
    for attempt in range(80):
        if time.time() > deadline:
            break
        
        random.seed(attempt)
        
        current_remaining = set(all_edges_set)
        bicliques = []
        valid_solution = True
        
        while current_remaining:
            if time.time() > deadline:
                valid_solution = False
                break
            
            # Compute degrees
            deg = [0] * n
            for e in current_remaining:
                u, v = list(e)
                deg[u] += 1
                deg[v] += 1
            
            # Pick candidates: mix of high-degree and random
            nonzero_verts = [v for v in range(n) if deg[v] > 0]
            if not nonzero_verts:
                break
            
            # Sort by degree descending, take top few, add some random
            sorted_verts = sorted(nonzero_verts, key=lambda x: deg[x], reverse=True)
            top_k = min(12, len(sorted_verts))
            candidates = sorted_verts[:top_k]
            
            # Add a few random candidates for diversity
            if len(nonzero_verts) > top_k:
                extra = random.sample(nonzero_verts, min(6, len(nonzero_verts) - top_k))
                for e_v in extra:
                    if e_v not in candidates:
                        candidates.append(e_v)
            
            # Try each candidate, pick the one giving the largest biclique
            best_L = None
            best_R = None
            best_removed = 0
            
            for v_start in candidates:
                L, R, removed = _expand_biclique(v_start, current_remaining, n)
                if removed > best_removed:
                    best_removed = removed
                    best_L = L
                    best_R = R
            
            if best_removed == 0 or best_L is None:
                # Fallback: single edge
                e = next(iter(current_remaining))
                u, v = list(e)
                bicliques.append([[int(u)], [int(v)]])
                current_remaining.remove(e)
                continue
            
            # Remove cross edges
            for l in best_L:
                for r in best_R:
                    e = frozenset((l, r))
                    if e in current_remaining:
                        current_remaining.remove(e)
            
            bicliques.append([sorted(best_L), sorted(best_R)])
        
        if not valid_solution:
            continue
        
        # Try to merge compatible bicliques
        bicliques = _try_merge_bicliques(bicliques, host_adj, n, all_edges_set)
        
        # Local search polish: try to replace two bicliques with one smaller one
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(bicliques)):
                    if time.time() > deadline:
                        break
                    # Try to see if we can find a single biclique covering edges of both
                    L_i, R_i = set(bicliques[i][0]), set(bicliques[i][1])
                    L_j, R_j = set(bicliques[j][0]), set(bicliques[j][1])
                    
                    # Edges covered by biclique i and j
                    edges_i = set()
                    for l in L_i:
                        for r in R_i:
                            edges_i.add(frozenset((l, r)))
                    edges_j = set()
                    for l in L_j:
                        for r in R_j:
                            edges_j.add(frozenset((l, r)))
                    combined_edges = edges_i | edges_j
                    
                    # Try to find a biclique that covers exactly combined_edges
                    # This is hard, so try greedy expansion from a vertex in combined_edges
                    if combined_edges:
                        # Pick a vertex with high degree in combined_edges
                        deg_in_combined = {}
                        for e in combined_edges:
                            u, v = list(e)
                            deg_in_combined[u] = deg_in_combined.get(u, 0) + 1
                            deg_in_combined[v] = deg_in_combined.get(v, 0) + 1
                        
                        if deg_in_combined:
                            start_v = max(deg_in_combined, key=deg_in_combined.get)
                            # Expand biclique on combined_edges only
                            temp_remaining = combined_edges
                            L_new, R_new, removed = _expand_biclique(start_v, temp_remaining, n)
                            
                            # Check if L_new, R_new covers exactly combined_edges
                            if L_new and R_new:
                                covered_new = set()
                                for l in L_new:
                                    for r in R_new:
                                        e = frozenset((l, r))
                                        if e in combined_edges:
                                            covered_new.add(e)
                                
                                if covered_new == combined_edges:
                                    # Verify it's a valid biclique (all cross pairs are host edges)
                                    valid_bic = True
                                    for l in L_new:
                                        for r in R_new:
                                            if r not in host_adj[l]:
                                                valid_bic = False
                                                break
                                        if not valid_bic:
                                            break
                                    
                                    if valid_bic:
                                        # Replace bicliques i and j with the new one
                                        new_bicliques = []
                                        for k in range(len(bicliques)):
                                            if k != i and k != j:
                                                new_bicliques.append(bicliques[k])
                                        new_bicliques.append([sorted(L_new), sorted(R_new)])
                                        
                                        # Validate
                                        if _validate_partition(new_bicliques, all_edges_set, n):
                                            bicliques = new_bicliques
                                            improved = True
                                            break
                            if improved:
                                break
                if improved:
                    break
        
        # Validate
        if _validate_partition(bicliques, all_edges_set, n):
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_solution = bicliques
                if best_count <= 3:
                    break
    
    if best_solution is None:
        # Fallback: per-vertex star decomposition
        bicliques = []
        for u in range(n):
            R_u = [v for v in host_adj[u] if v > u]
            if R_u:
                bicliques.append([[u], sorted(R_u)])
        best_solution = bicliques
        best_count = len(bicliques)
    
    # Convert to required format
    final_result = []
    for bic in best_solution:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        final_result.append([left, right])
    
    return final_result

# EVOLVE-BLOCK-END
