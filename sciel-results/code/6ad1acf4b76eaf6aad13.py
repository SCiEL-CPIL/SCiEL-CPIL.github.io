"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Preprocess adjacency for fast bitmask operations
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Map edges to indices for bitmask tracking
    edge_list = []
    edge_to_idx = {}
    for u, v in edges:
        if u == v: continue
        eu, ev = (u, v) if u < v else (v, u)
        edge_list.append((eu, ev))
        edge_to_idx[(eu, ev)] = len(edge_list) - 1
        
    # Ensure unique edges (graph might have duplicates in input? Assuming simple graph but safe to check)
    # The problem states "pinned graph edges", implying a simple graph.
    # Let's re-verify unique mapping if needed, but standard random graphs are simple.
    
    E = len(edge_list)
    # If E is large (>60), bitmask might be slow, but here E~150, Python int handles it fine.
    
    # Initial state: all edges are uncovered
    uncovered_mask = (1 << E) - 1
    
    # Helper to calculate coverage mask of a biclique (L, R)
    def get_coverage_mask(L, R):
        mask = 0
        for u in L:
            for v in R:
                if u == v: continue
                eu, ev = (u, v) if u < v else (v, u)
                if (eu, ev) in edge_to_idx:
                    mask |= (1 << edge_to_idx[(eu, ev)])
        return mask

    # Helper to expand a seed (L, R) to a maximal biclique
    # We want to maximize the number of *uncovered* edges covered.
    def expand_biclique(L, R, rem_mask):
        L = set(L)
        R = set(R)
        
        # Iteratively add vertices that cover new edges in rem_mask
        # A vertex w can be added to L if it is adjacent to all vertices in R
        # and covers at least one edge in rem_mask with R.
        # Similarly for R.
        
        changed = True
        while changed:
            changed = False
            
            # Candidates for L
            # Must be adjacent to all in R
            if R:
                cand_mask = (1 << n) - 1
                for v in R:
                    cand_mask &= adj_mask[v]
            else:
                cand_mask = (1 << n) - 1
            
            # Remove existing L and R
            for x in L:
                cand_mask &= ~(1 << x)
            for x in R:
                cand_mask &= ~(1 << x)
                
            # Check candidates
            c = cand_mask
            while c:
                lb = c & -c
                w = lb.bit_length() - 1
                c ^= lb
                
                # Does w cover any new edge in rem_mask?
                # Edges are (w, v) for v in R
                covers_new = False
                for v in R:
                    eu, ev = (w, v) if w < v else (v, w)
                    if (eu, ev) in edge_to_idx:
                        idx = edge_to_idx[(eu, ev)]
                        if (rem_mask >> idx) & 1:
                            covers_new = True
                            break
                            
                if covers_new:
                    L.add(w)
                    changed = True
                    
            # Candidates for R
            if L:
                cand_mask = (1 << n) - 1
                for u in L:
                    cand_mask &= adj_mask[u]
            else:
                cand_mask = (1 << n) - 1
                
            for x in L:
                cand_mask &= ~(1 << x)
            for x in R:
                cand_mask &= ~(1 << x)
                
            c = cand_mask
            while c:
                lb = c & -c
                w = lb.bit_length() - 1
                c ^= lb
                
                covers_new = False
                for u in L:
                    eu, ev = (w, u) if w < u else (u, w)
                    if (eu, ev) in edge_to_idx:
                        idx = edge_to_idx[(eu, ev)]
                        if (rem_mask >> idx) & 1:
                            covers_new = True
                            break
                            
                if covers_new:
                    R.add(w)
                    changed = True
                    
        return L, R

    bicliques = []
    
    # Greedy Set Cover with Candidate Generation
    # To improve performance and quality, we generate a pool of candidates each iteration.
    
    while uncovered_mask:
        # Generate candidates
        candidates = []
        
        # 1. Star candidates: For each vertex, connect to all uncovered neighbors
        for u in range(n):
            R = set()
            # Find neighbors v such that (u,v) is in uncovered_mask
            # We can iterate through uncovered edges or check adj_mask & uncovered
            # Since E is small, iterating edges is fine, but let's use bitmask if possible.
            # Actually, checking adj_mask[u] against uncovered edges is tricky without edge-to-vertex map.
            # Let's just iterate neighbors.
            for v in range(n):
                if u == v: continue
                eu, ev = (u, v) if u < v else (v, u)
                if (eu, ev) in edge_to_idx:
                    idx = edge_to_idx[(eu, ev)]
                    if (uncovered_mask >> idx) & 1:
                        R.add(v)
            if R:
                # Expand the star {u} x R
                L, R_exp = expand_biclique({u}, R, uncovered_mask)
                if L and R_exp and not (L & R_exp):
                    candidates.append((L, R_exp))
        
        # 2. Random Edge Candidates: Pick random uncovered edges and expand
        # Get list of uncovered edges
        unc_edges = []
        m = uncovered_mask
        while m:
            lb = m & -m
            idx = lb.bit_length() - 1
            m ^= lb
            unc_edges.append(edge_list[idx])
            
        if not unc_edges:
            break
            
        # Sample up to 50 random edges
        import random
        random.seed(42) # Deterministic for reproducibility, or use time? 
        # Better to use a deterministic seed based on iteration count or just fixed for stability?
        # Fixed seed is safer for "improvement" consistency.
        
        sample_size = min(50, len(unc_edges))
        sampled_edges = random.sample(unc_edges, sample_size)
        
        for u, v in sampled_edges:
            L, R_exp = expand_biclique({u}, {v}, uncovered_mask)
            if L and R_exp and not (L & R_exp):
                candidates.append((L, R_exp))
                
        # 3. High Degree Edge Candidates
        # Calculate degree in uncovered graph
        deg = [0] * n
        for u, v in unc_edges:
            deg[u] += 1
            deg[v] += 1
            
        # Sort by sum of degrees
        scored_edges = sorted(unc_edges, key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
        for u, v in scored_edges[:20]:
            L, R_exp = expand_biclique({u}, {v}, uncovered_mask)
            if L and R_exp and not (L & R_exp):
                candidates.append((L, R_exp))

        # Evaluate candidates and pick the best one (max coverage)
        best_L = None
        best_R = None
        best_count = 0
        best_mask = 0
        
        for L, R in candidates:
            if not L or not R: continue
            if L & R: continue # Valid biclique requires disjoint sides
            
            # Check validity: All cross pairs must be edges
            # This is guaranteed by expansion, but let's be safe.
            valid = True
            for u in L:
                for v in R:
                    eu, ev = (u, v) if u < v else (v, u)
                    if (eu, ev) not in edge_to_idx:
                        valid = False
                        break
                if not valid: break
            if not valid: continue
            
            # Calculate coverage
            cov_mask = get_coverage_mask(L, R)
            # Intersect with uncovered
            actual_cov = cov_mask & uncovered_mask
            count = bin(actual_cov).count('1')
            
            if count > best_count:
                best_count = count
                best_L = L
                best_R = R
                best_mask = actual_cov
                
        if best_L is None:
            # Fallback: pick a single edge
            # Find any uncovered edge
            m = uncovered_mask
            lb = m & -m
            idx = lb.bit_length() - 1
            u, v = edge_list[idx]
            bicliques.append([[int(u)], [int(v)]])
            uncovered_mask &= ~(1 << idx)
            continue
            
        # Add best biclique
        bicliques.append([sorted(best_L), sorted(best_R)])
        uncovered_mask &= ~best_mask
        
    # Final conversion to ensure types are correct
    result = []
    for L, R in bicliques:
        result.append([[int(x) for x in L], [int(x) for x in R]])
        
    return result
# EVOLVE-BLOCK-END
