"""Optimized biclique partition via maximal biclique enumeration + branch-and-bound set cover."""

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _seed_solution():
    """Fallback: one biclique per vertex (star decomposition)."""
    n = P['n']
    edges = P['edges']
    adj = {u: [] for u in range(n)}
    for a, b in edges:
        adj[a].append(b)
    result = []
    for u in range(n):
        if adj[u]:
            result.append([[u], list(adj[u])])
    return result


def _generate_candidates():
    """Generate candidate bicliques: stars and maximal bicliques via Bron-Kerbosch on bipartite graph."""
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for a, b in edges:
        if a > b:
            a, b = b, a
        edge_set.add((a, b))
    
    # Adjacency
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    
    candidates = []
    seen = set()
    
    def add_candidate(L, R):
        if not L or not R:
            return
        # Canonical form: sort both, ensure first element of L < first element of R (or use frozenset)
        L_sorted = tuple(sorted(L))
        R_sorted = tuple(sorted(R))
        # Normalize orientation: the smaller min vertex goes to L
        if L_sorted[0] < R_sorted[0]:
            key = (L_sorted, R_sorted)
        else:
            key = (R_sorted, L_sorted)
        if key not in seen:
            seen.add(key)
            L_list = list(L_sorted)
            R_list = list(R_sorted)
            if L_list[0] > R_list[0]:
                L_list, R_list = R_list, L_list
            candidates.append((L_list, R_list))
    
    # Star candidates: for each vertex u, biclique [{u}, N(u)]
    for u in range(n):
        if adj[u]:
            add_candidate([u], list(adj[u]))
    
    # Maximal bicliques via Bron-Kerbosch on the bipartite representation
    # A biclique (L,R) in G corresponds to a clique in the bipartite graph B
    # where B has vertex set V(G) and edges between u in L and v in R
    # We enumerate maximal cliques in B using Bron-Kerbosch with pivoting
    
    # Build the bipartite graph's adjacency (same as G since biclique cross pairs are edges)
    # Actually, for a biclique (L, R), we need all (u,v) for u in L, v in R to be in edge_set
    # The "clique" in the bipartite sense means: for any u1,u2 in L and v1,v2 in R,
    # all four cross pairs must exist. This is equivalent to:
    #   L is a set where all vertices share common neighbors in R
    #   R is a set where all vertices share common neighbors in L
    
    # Alternative: enumerate all maximal bicliques using the algorithm:
    # Start with each edge, try to expand
    
    # Use a different approach: enumerate maximal bicliques by growing from each edge
    # For efficiency, use the following:
    # A biclique (L,R) is maximal if:
    # 1. No vertex can be added to L (i.e., no v not in L with N(v) ∩ R = R... wait)
    # 2. Actually: (L,R) is maximal if no vertex can be added to L or R while maintaining the biclique property
    
    # Let's use the standard algorithm:
    # For each vertex u, for each vertex v in N(u):
    #   Start with L={u}, R={v}
    #   Expand: add any vertex that connects to all of R to L, or all of L to R
    #   Get maximal biclique
    
    # This can be exponential. For n=30, let's try a bounded approach.
    
    # Better: use the bipartite clique enumeration.
    # The bipartite graph B has vertices 0..n-1, and edge (u,v) in B iff (u,v) in G.
    # A biclique (L,R) in G = a "complete bipartite subgraph" in G = a clique in B?
    # No! A clique in B means all pairs in the clique are edges in G.
    # But a biclique (L,R) requires: all (u,v) for u in L, v in R are edges,
    # but (u1,u2) for u1,u2 in L need NOT be edges, and (v1,v2) for v1,v2 in R need NOT be edges.
    
    # So biclique enumeration is different from clique enumeration.
    
    # Algorithm for maximal bicliques:
    # 1. For each pair (u,v) that is an edge, start with L={u}, R={v}
    # 2. Expand: repeatedly add any vertex x to L if N(x) ⊇ R, or to R if N(x) ⊇ L
    # 3. The result is a maximal biclique containing edge (u,v)
    
    # This is still potentially exponential. For n=30 with ~200 edges, let's try it
    # with a time limit and deduplication.
    
    import time
    start_time = time.time()
    MAX_TIME = 25  # seconds for candidate generation
    
    for i, (a, b) in enumerate(edges):
        if time.time() - start_time > MAX_TIME:
            break
        L = {a}
        R = {b}
        changed = True
        while changed:
            changed = False
            # Try to add vertices to L
            for x in range(n):
                if x not in L and x not in R:
                    if adj[x].issuperset(R):
                        L.add(x)
                        changed = True
            # Try to add vertices to R
            for x in range(n):
                if x not in L and x not in R:
                    if adj[x].issuperset(L):
                        R.add(x)
                        changed = True
        add_candidate(list(L), list(R))
    
    # Also try starting from single vertices (stars are already covered)
    # And try some random seeds for diversity
    import random
    random.seed(42)
    for _ in range(200):
        if time.time() - start_time > MAX_TIME:
            break
        # Pick a random edge
        a, b = random.choice(edges)
        L = {a}
        R = {b}
        # Randomly expand
        for _ in range(20):
            changed = False
            for x in range(n):
                if x not in L and x not in R:
                    if adj[x].issuperset(R) and random.random() < 0.5:
                        L.add(x)
                        changed = True
            for x in range(n):
                if x not in L and x not in R:
                    if adj[x].issuperset(L) and random.random() < 0.5:
                        R.add(x)
                        changed = True
            if not changed:
                break
        add_candidate(list(L), list(R))
    
    return candidates


def _solve_branch_and_bound(candidates, edge_indices, time_limit=20):
    """
    Solve minimum set cover using branch-and-bound.
    candidates: list of (L, R) tuples
    edge_indices: dict mapping edge (a,b) -> index
    Returns: list of candidate indices that form a cover, or None if not solved.
    """
    import time
    start_time = time.time()
    
    n_edges = len(edge_indices)
    
    # Build edge membership: for each candidate, which edges it covers
    cand_edges = []
    for L, R in candidates:
        covered = set()
        for u in L:
            for v in R:
                key = (min(u, v), max(u, v))
                if key in edge_indices:
                    covered.add(edge_indices[key])
        if covered:
            cand_edges.append(covered)
        else:
            cand_edges.append(set())
    
    n_cands = len(cand_edges)
    
    # Remove candidates that are subsets of other candidates (dominance)
    # If cand A's edge set is a subset of cand B's edge set, A is dominated by B
    # (B covers everything A covers and possibly more)
    # Keep only maximal candidates
    keep = [True] * n_cands
    for i in range(n_cands):
        if not keep[i]:
            continue
        for j in range(n_cands):
            if i == j or not keep[j]:
                continue
            if cand_edges[i] <= cand_edges[j] and len(cand_edges[i]) < len(cand_edges[j]):
                keep[i] = False
                break
    
    filtered_cands = [i for i in range(n_cands) if keep[i]]
    filtered_cand_edges = [cand_edges[i] for i in filtered_cands]
    n_filtered = len(filtered_cands)
    
    if n_filtered == 0:
        return None
    
    # Sort candidates by size descending (greedy heuristic for initial upper bound)
    order = sorted(range(n_filtered), key=lambda i: -len(filtered_cand_edges[i]))
    
    # Greedy initial solution
    uncovered = set(range(n_edges))
    greedy_sol = []
    while uncovered:
        best_i = -1
        best_count = 0
        for i in order:
            count = len(filtered_cand_edges[i] & uncovered)
            if count > best_count:
                best_count = count
                best_i = i
        if best_i == -1 or best_count == 0:
            break
        greedy_sol.append(best_i)
        uncovered -= filtered_cand_edges[best_i]
    
    if uncovered:
        # Greedy failed to cover all (shouldn't happen if we have stars)
        return None
    
    upper_bound = len(greedy_sol)
    best_sol = list(greedy_sol)
    
    # Branch and bound
    # State: (current_selection, uncovered_set, depth)
    # Lower bound: for each uncovered edge, it must be covered by some candidate
    # Simple LB: ceil(|uncovered| / max_candidate_size)
    # Better LB: maximum number of pairwise disjoint edges (each needs separate candidate)
    # Even better: for each uncovered edge, count how many candidates can cover it;
    #   LB = max over edges of 1 (trivial)
    # Use: LB = size of maximum matching in the "conflict" graph? Too complex.
    # Use: LB = ceil(|uncovered| / max_cover) where max_cover = max |cand ∩ uncovered|
    
    # Precompute: for each edge, which candidates cover it
    edge_to_cands = [[] for _ in range(n_edges)]
    for i, ce in enumerate(filtered_cand_edges):
        for e in ce:
            edge_to_cands[e].append(i)
    
    # Check if every edge is covered by at least one candidate
    for e in range(n_edges):
        if not edge_to_cands[e]:
            return None  # Infeasible
    
    nodes_explored = 0
    best_sol_found = [True]  # mutable flag
    
    def branch_and_bound(selection, uncovered, depth):
        nonlocal nodes_explored, upper_bound, best_sol
        nodes_explored += 1
        
        if time.time() - start_time > time_limit:
            return
        
        if not uncovered:
            if len(selection) < upper_bound:
                upper_bound = len(selection)
                best_sol = list(selection)
            return
        
        # Lower bound
        # LB1: ceil(|uncovered| / max_cover)
        max_cover = 0
        for i in range(n_filtered):
            if filtered_cand_edges[i] & uncovered:
                c = len(filtered_cand_edges[i] & uncovered)
                if c > max_cover:
                    max_cover = c
        if max_cover == 0:
            return  # Can't cover remaining edges
        lb1 = (len(uncovered) + max_cover - 1) // max_cover
        
        # LB2: For each uncovered edge, it needs at least 1 candidate.
        # Better: find a set of edges such that no single candidate covers two of them.
        # This is the maximum "independent set" in the conflict graph.
        # Approximate: greedy
        # Pick an uncovered edge e, add 1 to LB, remove all edges that share a candidate with e
        lb2 = 0
        remaining = set(uncovered)
        while remaining:
            # Pick edge with fewest covering candidates
            e = min(remaining, key=lambda x: len(edge_to_cands[x]))
            lb2 += 1
            # Remove all edges that share any candidate with e
            to_remove = set()
            for cand_idx in edge_to_cands[e]:
                to_remove.update(filtered_cand_edges[cand_idx])
            remaining -= to_remove
            if not remaining:
                break
        
        lb = max(lb1, lb2)
        
        if len(selection) + lb >= upper_bound:
            return
        
        # Choose the next edge to cover (the one with fewest candidates)
        e = min(uncovered, key=lambda x: len(edge_to_cands[x]))
        
        # Branch on which candidate covers edge e
        # Order candidates by how much they cover (greedy)
        cands_for_e = sorted(edge_to_cands[e], key=lambda i: -len(filtered_cand_edges[i] & uncovered))
        
        for cand_idx in cands_for_e:
            if time.time() - start_time > time_limit:
                return
            new_uncovered = uncovered - filtered_cand_edges[cand_idx]
            selection.append(cand_idx)
            branch_and_bound(selection, new_uncovered, depth + 1)
            selection.pop()
        
        return
    
    branch_and_bound([], set(range(n_edges)), 0)
    
    if best_sol:
        # Convert filtered indices back to original candidate indices
        return [filtered_cands[i] for i in best_sol]
    return None


def construct():
    import time
    start = time.time()
    
    # Try the optimized approach
    try:
        candidates = _generate_candidates()
        
        # Build edge index
        edge_indices = {}
        for i, (a, b) in enumerate(P['edges']):
            key = (min(a, b), max(a, b))
            edge_indices[key] = i
        
        if candidates:
            sol = _solve_branch_and_bound(candidates, edge_indices, time_limit=40)
            if sol is not None:
                # Build the result
                result = []
                for idx in sol:
                    L, R = candidates[idx]
                    # Verify all cross pairs are edges
                    valid = True
                    for u in L:
                        for v in R:
                            key = (min(u, v), max(u, v))
                            if key not in edge_indices:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and L and R:
                        result.append([list(L), list(R)])
                
                # Verify partition: every edge covered exactly once
                if result:
                    edge_count = {}
                    for L, R in result:
                        for u in L:
                            for v in R:
                                key = (min(u, v), max(u, v))
                                edge_count[key] = edge_count.get(key, 0) + 1
                    
                    all_edges = set(edge_indices.keys())
                    covered = set(edge_count.keys())
                    
                    if covered == all_edges and all(c == 1 for c in edge_count.values()):
                        # Valid solution!
                        # Ensure types are correct
                        final_result = []
                        for L, R in result:
                            L_int = [int(x) for x in L]
                            R_int = [int(x) for x in R]
                            final_result.append([L_int, R_int])
                        return final_result
    except Exception:
        pass
    
    # Fallback to seed
    return _seed_solution()
