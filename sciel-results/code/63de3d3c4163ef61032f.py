"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

import sys
from functools import lru_cache

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
    
    m = len(edge_set)
    
    if m == 0:
        return []
    
    # Enumerate all maximal bicliques
    # A biclique (L, R) is maximal if we can't add any vertex to L or R while maintaining the biclique property
    # Use recursive enumeration: for each seed edge, grow to maximal
    
    def find_maximal_bicliques():
        """Find all maximal bicliques in the graph."""
        maximal_bics = []
        seen = set()
        
        # For each edge, try to grow a maximal biclique with u in L, v in R
        for u, v in edges:
            if u > v:
                continue  # avoid duplicates
            
            # Start with L={u}, R={v}
            L = {u}
            R = {v}
            
            # Grow to maximal
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
            
            # Check if this is a new maximal biclique
            key = (frozenset(L), frozenset(R))
            key2 = (frozenset(R), frozenset(L))  # symmetric
            
            if key not in seen and key2 not in seen:
                # Verify maximality: can't add any vertex to either side
                is_maximal = True
                for x in range(n):
                    if x not in L and x not in R:
                        if all(x in adj[y] for y in R):
                            is_maximal = False
                            break
                        if all(x in adj[y] for y in L):
                            is_maximal = False
                            break
                if is_maximal:
                    seen.add(key)
                    seen.add(key2)
                    # Store edges covered by this biclique
                    bic_edges = frozenset(frozenset((a, b)) for a in L for b in R if frozenset((a, b)) in edge_set)
                    maximal_bics.append((frozenset(L), frozenset(R), bic_edges))
        
        return maximal_bics
    
    maximal_bics = find_maximal_bicliques()
    
    # If we have too many, also add smaller bicliques from greedy
    # But first, try to solve exact cover with maximal bicliques
    
    # Build edge -> list of biclique indices mapping
    edge_to_bics = {e: [] for e in edge_set}
    for i, (L, R, bic_edges) in enumerate(maximal_bics):
        for e in bic_edges:
            edge_to_bics[e].append(i)
    
    # Check if all edges are covered by at least one maximal biclique
    uncovered = [e for e in edge_set if not edge_to_bics[e]]
    if uncovered:
        # Need to add non-maximal bicliques for uncovered edges
        for e in uncovered:
            u, v = tuple(e)
            L = frozenset([u])
            R = frozenset([v])
            bic_edges = frozenset([e])
            maximal_bics.append((L, R, bic_edges))
            edge_to_bics[e].append(len(maximal_bics) - 1)
    
    # Now solve exact cover: select minimum number of bicliques such that every edge is covered exactly once
    # Use backtracking with memoization
    
    all_edges = frozenset(edge_set)
    num_bics = len(maximal_bics)
    
    # Precompute: for each biclique, the set of edges it covers
    bic_edge_sets = [bic[2] for bic in maximal_bics]
    
    # Memoized backtracking
    # State: frozenset of remaining edges to cover
    # We want minimum number of bicliques to cover all remaining edges exactly once
    
    sys.setrecursionlimit(100000)
    
    @lru_cache(maxsize=None)
    def solve(remaining):
        """Return minimum number of bicliques needed to cover 'remaining' edges exactly once."""
        if not remaining:
            return 0
        
        # Pick the edge with fewest candidate bicliques
        best_edge = None
        best_count = float('inf')
        for e in remaining:
            # Count bicliques that cover e and don't conflict with already-covered edges
            # But we need bicliques whose edge set is a subset of remaining
            count = 0
            for i in range(num_bics):
                if bic_edge_sets[i] <= remaining:
                    if e in bic_edge_sets[i]:
                        count += 1
            if count < best_count:
                best_count = count
                best_edge = e
            if count == 0:
                return float('inf')  # impossible
        
        if best_count == 0:
            return float('inf')
        
        best = float('inf')
        for i in range(num_bics):
            if bic_edge_sets[i] <= remaining and best_edge in bic_edge_sets[i]:
                new_remaining = remaining - bic_edge_sets[i]
                sub = solve(new_remaining)
                if sub + 1 < best:
                    best = sub + 1
                if best == 1:  # can't do better than 1
                    break
        
        return best
    
    # Try to solve
    result_count = solve(all_edges)
    
    if result_count == float('inf'):
        # Fallback: use greedy
        remaining = set(edge_set)
        cover = []
        while remaining:
            # Find largest biclique covering remaining edges
            best_L, best_R, best_edges = None, None, set()
            for u, v in list(remaining):
                L = {u}
                R = {v}
                changed = True
                while changed:
                    changed = False
                    for x in range(n):
                        if x not in L and x not in R:
                            if all(frozenset((x, y)) in remaining for y in R):
                                L.add(x)
                                changed = True
                    for x in range(n):
                        if x not in L and x not in R:
                            if all(frozenset((x, y)) in remaining for y in L):
                                R.add(x)
                                changed = True
                bic_edges = set()
                for a in L:
                    for b in R:
                        e = frozenset((a, b))
                        if e in remaining:
                            bic_edges.add(e)
                if not best_edges or len(bic_edges) > len(best_edges):
                    best_L, best_R, best_edges = L, R, bic_edges
            
            if not best_edges:
                e = next(iter(remaining))
                u, v = tuple(e)
                best_L, best_R, best_edges = {u}, {v}, {e}
            
            cover.append((best_L, best_R))
            remaining -= best_edges
        
        result_count = len(cover)
    
    # Now reconstruct the actual bicliques
    # We need to find which bicliques form the optimal cover
    # Re-run backtracking to get the actual selection
    
    @lru_cache(maxsize=None)
    def solve_with_path(remaining):
        """Return (count, selected_biclique_indices)."""
        if not remaining:
            return (0, ())
        
        best_edge = None
        best_count = float('inf')
        for e in remaining:
            count = 0
            for i in range(num_bics):
                if bic_edge_sets[i] <= remaining and e in bic_edge_sets[i]:
                    count += 1
            if count < best_count:
                best_count = count
                best_edge = e
            if count == 0:
                return (float('inf'), ())
        
        if best_count == 0:
            return (float('inf'), ())
        
        best = (float('inf'), ())
        for i in range(num_bics):
            if bic_edge_sets[i] <= remaining and best_edge in bic_edge_sets[i]:
                new_remaining = remaining - bic_edge_sets[i]
                sub_count, sub_path = solve_with_path(new_remaining)
                if sub_count + 1 < best[0]:
                    best = (sub_count + 1, (i,) + sub_path)
                if best[0] == 1:
                    break
        
        return best
    
    count, path = solve_with_path(all_edges)
    
    if count == float('inf'):
        # Fallback to greedy
        remaining = set(edge_set)
        cover = []
        while remaining:
            best_L, best_R, best_edges = None, None, set()
            for u, v in list(remaining):
                L = {u}
                R = {v}
                changed = True
                while changed:
                    changed = False
                    for x in range(n):
                        if x not in L and x not in R:
                            if all(frozenset((x, y)) in remaining for y in R):
                                L.add(x)
                                changed = True
                    for x in range(n):
                        if x not in L and x not in R:
                            if all(frozenset((x, y)) in remaining for y in L):
                                R.add(x)
                                changed = True
                bic_edges = set()
                for a in L:
                    for b in R:
                        e = frozenset((a, b))
                        if e in remaining:
                            bic_edges.add(e)
                if not best_edges or len(bic_edges) > len(best_edges):
                    best_L, best_R, best_edges = L, R, bic_edges
            
            if not best_edges:
                e = next(iter(remaining))
                u, v = tuple(e)
                best_L, best_R, best_edges = {u}, {v}, {e}
            
            cover.append((best_L, best_R))
            remaining -= best_edges
    else:
        cover = [(maximal_bics[i][0], maximal_bics[i][1]) for i in path]
    
    # Convert to required format
    result = []
    for L, R in cover:
        left = sorted([int(x) for x in L])
        right = sorted([int(x) for x in R])
        if left and right and not set(left) & set(right):
            result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
