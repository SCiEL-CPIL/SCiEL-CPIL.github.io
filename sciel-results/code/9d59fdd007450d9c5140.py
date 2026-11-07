import time
import random
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build data structures
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(_edge_key(u, v))
        
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    def expand_biclique(L, R):
        """
        Expand L and R to a maximal biclique in the original graph.
        Returns (frozenset(L), frozenset(R))
        """
        L_set = set(L)
        R_set = set(R)
        changed = True
        while changed:
            changed = False
            # Try adding to L
            for w in range(n):
                if w not in L_set and w not in R_set:
                    # w can join L if connected to all in R
                    if all(r in adj[w] for r in R_set):
                        L_set.add(w)
                        changed = True
            # Try adding to R
            for w in range(n):
                if w not in L_set and w not in R_set:
                    # w can join R if connected to all in L
                    if all(l in adj[w] for l in L_set):
                        R_set.add(w)
                        changed = True
        return (frozenset(L_set), frozenset(R_set))

    def get_cross_edges(L, R):
        """Get the set of edge keys covered by biclique (L, R)."""
        covered = set()
        for u in L:
            for v in R:
                covered.add(_edge_key(u, v))
        return covered

    # Phase 1: Generate a diverse pool of candidate bicliques
    # We use multiple random restarts to find different maximal bicliques
    candidates = set()
    
    # 1. Add bicliques starting from every edge (might be too many, so we sample)
    # But for n=30, edges ~160, this is feasible if expansion is fast.
    # Let's try to generate candidates from random subsets of edges or specific structures.
    
    # Strategy: 
    # A. Generate from high-degree edges.
    # B. Generate from random edge pairs to find larger structures.
    # C. Generate "star" bicliques (center + neighbors).
    
    # Add star bicliques
    for v in range(n):
        L = {v}
        R = set(adj[v])
        if R:
            b = expand_biclique(L, R)
            if b[0] and b[1]:
                candidates.add(b)
            # Also try swapping
            L2 = set(adj[v])
            R2 = {v}
            if L2:
                b2 = expand_biclique(L2, R2)
                if b2[0] and b2[1]:
                    candidates.add(b2)
    
    # Generate from random edges
    random.seed(42)
    edge_list = list(edge_set)
    num_random_trials = 200
    trial_time_limit = (deadline - start_time - 30.0) / max(1, num_random_trials)
    
    for i in range(num_random_trials):
        if time.time() > deadline - 30.0:
            break
        
        # Pick a random edge to start
        u, v = random.choice(edge_list)
        L = {u}
        R = {v}
        
        # Sometimes start with a pair of edges to encourage larger bicliques
        if i % 3 == 0 and len(edge_list) > 1:
             # Pick another edge sharing a vertex? Or just random?
             # Let's just expand the single edge, the expansion loop handles growth.
             pass
             
        b = expand_biclique(L, R)
        if b[0] and b[1]:
            candidates.add(b)
            
        # Also try starting with a random vertex and a random neighbor
        if i % 2 == 0:
             v_start = random.randint(0, n-1)
             if adj[v_start]:
                 v_nei = random.choice(list(adj[v_start]))
                 L = {v_start}
                 R = {v_nei}
                 b = expand_biclique(L, R)
                 if b[0] and b[1]:
                     candidates.add(b)

    # Ensure we have enough candidates. If we have very few, generate more aggressively.
    # Also, we should include "single edge" bicliques as a fallback, but they are rarely optimal 
    # unless the graph is very sparse or has no structure. 
    # However, for safety, let's add all single-edge bicliques to the candidate pool.
    for u, v in edge_set:
        candidates.add((frozenset({u}), frozenset({v})))
        candidates.add((frozenset({v}), frozenset({u})))

    # Convert candidates to a list for easier access
    candidate_list = list(candidates)
    
    # Phase 2: Greedy Set Cover
    remaining_edges = set(edge_set)
    selected_bicliques = []
    
    # Pre-compute cross edges for all candidates to speed up the loop?
    # There might be thousands of candidates. 
    # Let's compute on the fly but cache if possible. 
    # Actually, computing cross edges is O(|L|*|R|). 
    # Let's just do it in the loop.
    
    while remaining_edges:
        if time.time() > deadline - 10.0:
            break
            
        best_biclique = None
        best_cover_count = 0
        
        # To speed up, we can filter candidates that don't cover any remaining edges.
        # But checking that is O(|L|*|R|) anyway.
        
        # Optimization: Only consider candidates that have at least one edge in remaining_edges.
        # We can do a quick check.
        
        for L, R in candidate_list:
            if time.time() > deadline - 10.0:
                break
                
            # Quick check: does this biclique cover any remaining edge?
            # We need to check if any (u,v) in LxR is in remaining_edges.
            # If not, skip.
            
            cover_count = 0
            # Iterate through L and R
            for u in L:
                for v in R:
                    ek = _edge_key(u, v)
                    if ek in remaining_edges:
                        cover_count += 1
                        if cover_count > best_cover_count:
                            # Early exit if we can't beat the best? 
                            # Max possible is len(remaining_edges) but we don't know it exactly without counting all.
                            # Just count all for this candidate.
                            pass
                            
            if cover_count > best_cover_count:
                best_cover_count = cover_count
                best_biclique = (L, R)
                
        if best_biclique is None or best_cover_count == 0:
            # Fallback: pick a random remaining edge
            ek = next(iter(remaining_edges))
            u, v = ek
            best_biclique = (frozenset({u}), frozenset({v}))
            best_cover_count = 1
            
        L, R = best_biclique
        selected_bicliques.append((L, R))
        
        # Remove covered edges
        for u in L:
            for v in R:
                ek = _edge_key(u, v)
                if ek in remaining_edges:
                    remaining_edges.remove(ek)
                    
    # Phase 3: Refinement (Merge Pass)
    # Try to merge pairs of bicliques if possible.
    # A merge is valid if:
    # 1. The union of L's and R's forms a valid biclique in the original graph.
    # 2. The set of edges covered by the new biclique is exactly the union of edges covered by the two old ones.
    
    current_bicliques = [(list(L), list(R)) for L, R in selected_bicliques]
    
    merged = True
    while merged:
        if time.time() > deadline - 5.0:
            break
        merged = False
        
        for i in range(len(current_bicliques)):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, len(current_bicliques)):
                if time.time() > deadline - 5.0:
                    break
                
                Li, Ri = current_bicliques[i]
                Lj, Rj = current_bicliques[j]
                
                # Try 4 orientations of merge
                merges = [
                    (set(Li) | set(Lj), set(Ri) | set(Rj)),
                    (set(Li) | set(Rj), set(Ri) | set(Lj)),
                    (set(Ri) | set(Lj), set(Li) | set(Rj)),
                    (set(Ri) | set(Rj), set(Li) | set(Lj))
                ]
                
                for NL, NR in merges:
                    if not NL or not NR:
                        continue
                    if NL & NR:
                        continue
                    
                    # Check if valid biclique in original graph
                    valid = True
                    new_cross_edges = set()
                    for u in NL:
                        for v in NR:
                            ek = _edge_key(u, v)
                            if ek not in edge_set:
                                valid = False
                                break
                            new_cross_edges.add(ek)
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    # Check if new_cross_edges equals union of old cross edges
                    old_edges_i = set()
                    for u in Li:
                        for v in Ri:
                            old_edges_i.add(_edge_key(u, v))
                            
                    old_edges_j = set()
                    for u in Lj:
                        for v in Rj:
                            old_edges_j.add(_edge_key(u, v))
                            
                    if new_cross_edges == (old_edges_i | old_edges_j):
                        # Merge!
                        current_bicliques[i] = (list(NL), list(NR))
                        current_bicliques.pop(j)
                        merged = True
                        break
            if merged:
                break
                
    # Final Validation
    covered = set()
    valid = True
    for L, R in current_bicliques:
        if not L or not R:
            valid = False
            break
        if set(L) & set(R):
            valid = False
            break
        for u in L:
            for v in R:
                ek = _edge_key(u, v)
                if ek not in edge_set:
                    valid = False
                    break
                if ek in covered:
                    valid = False
                    break
                covered.add(ek)
                
    if not valid or covered != edge_set:
        # Fallback to single edges if validation fails
        result = []
        for u, v in edges:
            result.append([[int(u)], [int(v)]])
        return result
        
    # Format output
    result = []
    for L, R in current_bicliques:
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        result.append([L_int, R_int])
        
    return result
