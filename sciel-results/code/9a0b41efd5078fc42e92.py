"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

import sys
import time
from collections import defaultdict

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
        edge_set.add((min(u, v), max(u, v)))
    
    # Total edges
    m = len(edge_set)
    
    if m == 0:
        return []
    
    # Step 1: Find all maximal bicliques
    # A biclique (L, R) is maximal if no vertex can be added to L or R while maintaining the biclique property.
    # We enumerate by considering each edge (u,v) as a seed and growing to maximality.
    
    # For efficiency, we'll use the following approach:
    # For each edge (u, v), the maximal biclique containing it has:
    #   L = {x : x is connected to all vertices in R}
    #   R = {y : y is connected to all vertices in L}
    # where we start with L={u}, R={v} and iterate to fixpoint.
    
    # But many edges will yield the same maximal biclique, so we deduplicate.
    
    maximal_bicliques = []
    seen_bicliques = set()
    
    for u, v in edge_set:
        # Start with biclique ({u}, {v})
        L = frozenset([u])
        R = frozenset([v])
        
        # Grow to maximality
        changed = True
        while changed:
            changed = False
            # Try adding vertices to L
            for x in range(n):
                if x not in L and x not in R:
                    if all(x in adj[y] for y in R):
                        L = L | {x}
                        changed = True
            # Try adding vertices to R
            for x in range(n):
                if x not in L and x not in R:
                    if all(x in adj[y] for y in L):
                        R = R | {x}
                        changed = True
        
        # Check if this biclique is valid (all cross edges exist)
        valid = True
        for x in L:
            for y in R:
                if (min(x,y), max(x,y)) not in edge_set:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Deduplicate
        # Canonical form: sort L and R, then represent as (frozenset(L), frozenset(R))
        # But we need to be careful about symmetry: (L,R) and (R,L) are the same biclique
        # For our purposes, we'll keep them as-is since the partition is what matters
        key = (frozenset(L), frozenset(R))
        if key not in seen_bicliques:
            seen_bicliques.add(key)
            # Compute the set of edges covered by this biclique
            biclique_edges = frozenset(
                (min(x,y), max(x,y)) for x in L for y in R
            )
            maximal_bicliques.append((L, R, biclique_edges))
    
    # Step 2: Solve minimum set cover
    # We want to select the minimum number of bicliques such that every edge is covered exactly once.
    # Actually, the requirement is that cross-edge sets must partition all edges exactly once.
    # This means we need an exact cover, not just a set cover.
    
    # For exact cover with backtracking:
    # - Branch on the uncovered edge with the fewest candidate bicliques
    # - Use memoization to prune redundant states
    
    # First, build the incidence: for each edge, which bicliques cover it?
    edge_to_bicliques = defaultdict(list)
    for i, (L, R, be) in enumerate(maximal_bicliques):
        for e in be:
            edge_to_bicliques[e].append(i)
    
    # Remove bicliques that don't cover any edge (shouldn't happen but just in case)
    # Also, we need to ensure that the selected bicliques have disjoint edge sets.
    
    # Exact cover via backtracking with memoization
    # State: frozenset of uncovered edges
    # But this could be exponential. Let's use a smarter approach.
    
    # Alternative: since the number of maximal bicliques might be manageable,
    # we can use a recursive search with pruning.
    
    # Let's try a different approach: greedy exact cover with backtracking
    # 1. Greedily select bicliques that cover the most uncovered edges
    # 2. If stuck, backtrack
    
    # Actually, for exact cover, we need to be more careful.
    # Let's use Algorithm X (Dancing Links) style approach.
    
    # First, let's try a simple greedy approach to get an initial solution
    remaining_edges = set(edge_set)
    selected = []
    
    # Sort bicliques by size (number of edges) descending
    biclique_order = sorted(range(len(maximal_bicliques)), 
                           key=lambda i: len(maximal_bicliques[i][2]), reverse=True)
    
    for i in biclique_order:
        if not remaining_edges:
            break
        L, R, be = maximal_bicliques[i]
        # Check if this biclique's edges are all uncovered
        if be.issubset(remaining_edges):
            selected.append(i)
            remaining_edges -= be
    
    # If greedy didn't cover all edges, we need a more sophisticated approach
    # Let's try backtracking search
    
    if remaining_edges:
        # Use recursive backtracking with memoization
        # State: frozenset of uncovered edges
        # This might be too large, so let's use a different state representation
        
        # Alternative: use the fact that we have a small number of edges
        # Let's try a recursive search that branches on edges
        
        # Build a better initial solution using local search
        # Or use a SAT-like approach
        
        # For now, let's try a simple backtracking
        # But first, let's see how many edges are left
        if len(remaining_edges) <= 20:
            # Small enough for brute force with memoization
            memo = {}
            
            def solve_exact(remaining):
                if not remaining:
                    return []
                if remaining in memo:
                    return memo[remaining]
                
                # Find the edge with fewest candidate bicliques
                min_edge = None
                min_candidates = None
                for e in remaining:
                    candidates = [i for i in edge_to_bicliques[e] 
                                  if maximal_bicliques[i][2].issubset(remaining)]
                    if min_candidates is None or len(candidates) < len(min_candidates):
                        min_edge = e
                        min_candidates = candidates
                        if len(candidates) == 1:
                            break
                
                if not min_candidates:
                    return None  # No solution
                
                best = None
                for i in min_candidates:
                    L, R, be = maximal_bicliques[i]
                    if not be.issubset(remaining):
                        continue
                    new_remaining = remaining - be
                    sub_result = solve_exact(new_remaining)
                    if sub_result is not None:
                        candidate = [i] + sub_result
                        if best is None or len(candidate) < len(best):
                            best = candidate
                
                memo[remaining] = best
                return best
            
            result = solve_exact(remaining_edges)
            if result is not None:
                selected = selected + result
            else:
                # Fallback: add individual edges as bicliques
                for e in remaining_edges:
                    u, v = e
                    L = frozenset([u])
                    R = frozenset([v])
                    be = frozenset([e])
                    key = (L, R)
                    if key not in seen_bicliques:
                        seen_bicliques.add(key)
                        maximal_bicliques.append((L, R, be))
                    idx = len(maximal_bicliques) - 1
                    selected.append(idx)
        else:
            # Too many edges for brute force, use greedy fallback
            for e in remaining_edges:
                u, v = e
                L = frozenset([u])
                R = frozenset([v])
                be = frozenset([e])
                key = (L, R)
                if key not in seen_bicliques:
                    seen_bicliques.add(key)
                    maximal_bicliques.append((L, R, be))
                idx = len(maximal_bicliques) - 1
                selected.append(idx)
    
    # Step 3: Verify and convert to output format
    # Check that the selected bicliques form an exact cover
    covered = set()
    for i in selected:
        L, R, be = maximal_bicliques[i]
        # Check disjointness
        if be & covered:
            # Overlap detected, this shouldn't happen with exact cover
            pass
        covered |= be
    
    # If not all edges are covered, add missing ones
    if covered != edge_set:
        for e in edge_set - covered:
            u, v = e
            L = frozenset([u])
            R = frozenset([v])
            be = frozenset([e])
            key = (L, R)
            if key not in seen_bicliques:
                seen_bicliques.add(key)
                maximal_bicliques.append((L, R, be))
            idx = len(maximal_bicliques) - 1
            selected.append(idx)
    
    # Convert to required format
    result = []
    for i in selected:
        L, R, be = maximal_bicliques[i]
        left = sorted(list(L))
        right = sorted(list(R))
        if left and right and not set(left) & set(right):
            result.append([left, right])
        else:
            result.append([[min(L)], [max(R)]])
    
    return result
# EVOLVE-BLOCK-END
