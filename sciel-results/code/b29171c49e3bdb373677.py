"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start = time.time()
    deadline = start + 80.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    all_edges_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        all_edges_set.add((min(u, v), max(u, v)))
    
    remaining = set(all_edges_set)
    bicliques = []
    
    # Phase 1: Greedy set-cover with maximal biclique expansion
    while remaining and time.time() < deadline:
        # Find the best biclique to cover: maximize edges covered / (size of biclique)
        # Use a heuristic: try each remaining edge as a seed, expand greedily
        best_biclique = None
        best_covered = set()
        best_score = -1
        
        # Sample a subset of remaining edges as seeds for speed
        rem_list = list(remaining)
        if len(rem_list) > 60:
            step = len(rem_list) / 60.0
            idxs = set()
            for i in range(60):
                idxs.add(int(i * step))
            idxs.add(0)
            idxs.add(len(rem_list) - 1)
            seeds = [rem_list[i] for i in sorted(idxs)]
        else:
            seeds = rem_list
        
        for u, v in seeds:
            # Start with this edge, try to expand to a maximal biclique
            L = {u}
            R = {v}
            
            # Iteratively expand: add vertices that connect to all on the other side
            changed = True
            while changed:
                changed = False
                # Try adding to L
                for x in range(n):
                    if x in L:
                        continue
                    # x must connect to all in R via remaining edges
                    if all((min(x, r), max(x, r)) in remaining for r in R):
                        L.add(x)
                        changed = True
                # Try adding to R
                for y in range(n):
                    if y in R:
                        continue
                    # y must connect to all in L via remaining edges
                    if all((min(l, y), max(l, y)) in remaining for l in L):
                        R.add(y)
                        changed = True
            
            # Ensure L and R are disjoint
            if L & R:
                continue
            
            # Count covered edges
            covered = set()
            for l in L:
                for r in R:
                    k = (min(l, r), max(l, r))
                    if k in remaining:
                        covered.add(k)
            
            if not covered:
                continue
            
            # Score: prefer covering more edges, with slight preference for smaller bicliques
            score = len(covered) / max(1, len(L) + len(R))
            if score > best_score:
                best_score = score
                best_biclique = (L, R)
                best_covered = covered
        
        if best_biclique is None or not best_covered:
            # Fallback: single edge
            k = next(iter(remaining))
            best_biclique = ({k[0]}, {k[1]})
            best_covered = {k}
        
        bicliques.append(best_biclique)
        remaining -= best_covered
    
    # Handle any remaining edges (shouldn't happen, but safety)
    if remaining:
        for k in list(remaining):
            bicliques.append(({k[0]}, {k[1]}))
        remaining = set()
    
    # Phase 2: Try to merge bicliques
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Try merging: L = L1|L2, R = R1|R2
                L = set(L1) | set(L2)
                R = set(R1) | set(R2)
                
                if L & R:
                    continue
                
                # Check if every cross pair is an edge in the original graph
                ok = True
                for l in L:
                    for r in R:
                        if (min(l, r), max(l, r)) not in all_edges_set:
                            ok = False
                            break
                    if not ok:
                        break
                
                if ok:
                    # Ensure the union's cross edges exactly match the union of the two bicliques' cross edges
                    new_edges = set()
                    for l in L:
                        for r in R:
                            new_edges.add((min(l, r), max(l, r)))
                    
                    old_edges = set()
                    for l in L1:
                        for r in R1:
                            old_edges.add((min(l, r), max(l, r)))
                    for l in L2:
                        for r in R2:
                            old_edges.add((min(l, r), max(l, r)))
                    
                    if new_edges == old_edges:
                        bicliques[i] = (L, R)
                        bicliques.pop(j)
                        changed = True
                        break
            if changed:
                break
    
    # Phase 3: Try to reduce bicliques by moving vertices
    if time.time() < deadline:
        for _iter in range(5):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(bicliques)):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    
                    # Try moving a vertex from L1 to L2 (or vice versa) to enable merging
                    for v in list(L1):
                        new_L1 = L1 - {v}
                        new_L2 = L2 | {v}
                        if not new_L1 or not new_L2:
                            continue
                        # Check if both bicliques are still valid
                        valid1 = all((min(l, r), max(l, r)) in all_edges_set for l in new_L1 for r in R1)
                        valid2 = all((min(l, r), max(l, r)) in all_edges_set for l in new_L2 for r in R2)
                        if valid1 and valid2:
                            # Check if this enables a merge
                            merged_L = new_L1 | new_L2
                            merged_R = R1 | R2
                            if merged_L & merged_R:
                                continue
                            ok = all((min(l, r), max(l, r)) in all_edges_set for l in merged_L for r in merged_R)
                            if ok:
                                # Verify edge coverage is preserved
                                old_edges = set()
                                for l in L1:
                                    for r in R1:
                                        old_edges.add((min(l, r), max(l, r)))
                                for l in L2:
                                    for r in R2:
                                        old_edges.add((min(l, r), max(l, r)))
                                new_edges = set()
                                for l in merged_L:
                                    for r in merged_R:
                                        new_edges.add((min(l, r), max(l, r)))
                                if new_edges == old_edges:
                                    bicliques[i] = (merged_L, merged_R)
                                    bicliques.pop(j)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Convert to lists of ints
    result = []
    for L, R in bicliques:
        result.append([list(int(x) for x in sorted(L)), list(int(x) for x in sorted(R))])
    
    return result
# EVOLVE-BLOCK-END
