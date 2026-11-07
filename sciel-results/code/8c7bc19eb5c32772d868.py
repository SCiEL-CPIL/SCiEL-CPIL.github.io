"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

import sys
sys.setrecursionlimit(10000)

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(frozenset((u, v)))
    
    # Total edges
    m = len(edge_set)
    
    if m == 0:
        return []
    
    # Strategy: find a good set of candidate bicliques, then solve minimum set cover
    # using branch-and-bound with memoization
    
    # Step 1: Generate candidate bicliques
    # For each edge (u,v), try to grow it to a maximal biclique
    # Also try single-vertex seeds
    
    candidates = []
    seen = set()
    
    # For each edge, grow to maximal biclique
    for u, v in edges:
        # Start with ({u}, {v})
        L = {u}
        R = {v}
        
        # Grow iteratively
        changed = True
        while changed:
            changed = False
            # Try adding to L
            for x in range(n):
                if x not in L and x not in R:
                    if all(x in adj[y] for y in R):
                        L.add(x)
                        changed = True
            # Try adding to R
            for x in range(n):
                if x not in L and x not in R:
                    if all(x in adj[y] for y in L):
                        R.add(x)
                        changed = True
        
        # Canonical form: sort and ensure L < R by min vertex
        L_sorted = tuple(sorted(L))
        R_sorted = tuple(sorted(R))
        if L_sorted[0] > R_sorted[0]:
            L_sorted, R_sorted = R_sorted, L_sorted
        
        key = (L_sorted, R_sorted)
        if key not in seen:
            seen.add(key)
            candidates.append((L_sorted, R_sorted))
    
    # Also try growing from single vertices
    for u in range(n):
        for v in adj[u]:
            if u > v:
                continue
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                for x in range(n):
                    if x not in L and x not in R:
                        if all(x in adj[y] for y in R):
                            L.add(x)
                            changed = True
                for x in range(n):
                    if x not in L and x not in R:
                        if all(x in adj[y] for y in L):
                            R.add(x)
                            changed = True
            L_sorted = tuple(sorted(L))
            R_sorted = tuple(sorted(R))
            if L_sorted[0] > R_sorted[0]:
                L_sorted, R_sorted = R_sorted, L_sorted
            key = (L_sorted, R_sorted)
            if key not in seen:
                seen.add(key)
                candidates.append((L_sorted, R_sorted))
    
    # Remove dominated candidates: if biclique A's edge set is subset of biclique B's edge set, drop A
    # Compute edge sets for each candidate
    cand_edge_sets = []
    for L, R in candidates:
        es = set()
        for u in L:
            for v in R:
                e = frozenset((u, v))
                if e in edge_set:
                    es.add(e)
        cand_edge_sets.append(es)
    
    # Filter out dominated
    kept = []
    for i in range(len(candidates)):
        dominated = False
        for j in range(len(candidates)):
            if i == j:
                continue
            if cand_edge_sets[i] < cand_edge_sets[j]:  # strict subset
                dominated = True
                break
        if not dominated:
            kept.append(i)
    
    candidates = [candidates[i] for i in kept]
    cand_edge_sets = [cand_edge_sets[i] for i in kept]
    
    # If too many candidates, keep only the largest ones
    if len(candidates) > 500:
        # Sort by size descending, keep top 500
        order = sorted(range(len(candidates)), key=lambda i: -len(cand_edge_sets[i]))
        kept_idx = order[:500]
        candidates = [candidates[i] for i in kept_idx]
        cand_edge_sets = [cand_edge_sets[i] for i in kept_idx]
    
    # Step 2: Solve minimum set cover using branch-and-bound
    # Each edge must be covered exactly once (partition)
    # Actually, the problem says "partition all pinned graph edges exactly once"
    # So we need an exact cover, not just a cover
    
    # Map edges to indices
    edge_list = list(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    
    # For each candidate, get the set of edge indices it covers
    cand_edge_indices = []
    for es in cand_edge_sets:
        idxs = frozenset(edge_to_idx[e] for e in es)
        cand_edge_indices.append(idxs)
    
    # We need to select a subset of candidates such that every edge is covered exactly once
    # This is exact cover / partition problem
    
    # Use DLX (Dancing Links) or simple backtracking with memoization
    
    # For memoization, use the set of uncovered edges as state
    # But that might be too large. Let's use a different approach.
    
    # Greedy first to get an upper bound
    remaining = set(range(m))
    greedy_cover = []
    while remaining:
        best_i = -1
        best_count = -1
        for i, idxs in enumerate(cand_edge_indices):
            count = len(idxs & remaining)
            if count > best_count:
                best_count = count
                best_i = i
        if best_i == -1:
            break
        greedy_cover.append(best_i)
        remaining -= cand_edge_indices[best_i]
    
    upper_bound = len(greedy_cover)
    
    # Now try to improve with backtracking
    best_solution = greedy_cover[:]
    best_count = upper_bound
    
    # Sort candidates by size descending for better pruning
    order = sorted(range(len(candidates)), key=lambda i: -len(cand_edge_indices[i]))
    sorted_cands = [order[i] for i in range(len(order))]
    sorted_idx = [cand_edge_indices[i] for i in sorted_cands]
    
    # Backtracking search
    # State: set of uncovered edges
    # Try to cover them with minimum number of bicliques
    
    def backtrack(uncovered, current, depth):
        nonlocal best_solution, best_count
        if not uncovered:
            if depth < best_count:
                best_count = depth
                best_solution = current[:]
            return True
        if depth >= best_count:
            return False
        # Lower bound: at least ceil(len(uncovered) / max_possible)
        # For simplicity, use 1
        # Find an uncovered edge with fewest candidates
        # For each edge in uncovered, count how many remaining candidates cover it
        # This is expensive, so just pick the first uncovered edge
        edge = next(iter(uncovered))
        for i, idxs in enumerate(sorted_idx):
            if edge in idxs:
                new_uncovered = uncovered - idxs
                current.append(sorted_cands[i])
                if backtrack(new_uncovered, current, depth + 1):
                    current.pop()
                    return True
                current.pop()
        return False
    
    # Try to improve
    # Start with empty set and try to find a solution with fewer than upper_bound bicliques
    # Use iterative deepening
    for limit in range(1, upper_bound):
        if backtrack(set(range(m)), [], 0) and best_count <= limit:
            break
    
    # Convert best_solution to the required format
    result = []
    for i in best_solution:
        L, R = candidates[i]
        left = list(L)
        right = list(R)
        if left and right and not set(left) & set(right):
            result.append([left, right])
    
    # Verify and fallback if invalid
    covered = {}
    for L, R in [(b[0], b[1]) for b in result]:
        for u in L:
            for v in R:
                e = frozenset((u, v))
                if e in edge_set:
                    covered[e] = covered.get(e, 0) + 1
    
    valid = True
    for e in edge_set:
        if covered.get(e, 0) != 1:
            valid = False
            break
    
    if not valid or len(result) == 0:
        # Fallback: one biclique per edge
        result = []
        for u, v in edges:
            result.append([[u], [v]])
    
    return result
# EVOLVE-BLOCK-END
