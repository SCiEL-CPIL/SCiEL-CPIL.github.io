"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix and set
    adj = [[False]*n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((u, v) if u < v else (v, u))
    
    # Helper: check if a set of vertices forms a clique
    def is_clique(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if not adj[vertices[i]][vertices[j]]:
                    return False
        return True
    
    # Helper: check if a set of vertices forms an independent set
    def is_independent(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if adj[vertices[i]][vertices[j]]:
                    return False
        return True
    
    # Find all maximal cliques
    all_cliques = []
    all_vertices = list(range(n))
    
    # Use Bron-Kerbosch with pivot for maximal cliques
    def bron_kerbosch(r, p, x):
        if not p and not x:
            if len(r) > 1:  # Only cliques of size > 1 are useful (size 1 is both clique and independent)
                all_cliques.append(frozenset(r))
            return
        
        # Choose pivot to minimize branching
        if p or x:
            # Simple pivot: pick vertex with max neighbors in P
            pivot = max(list(p) + list(x), key=lambda v: len([u for u in p if adj[v][u]]))
            candidates = [v for v in p if not adj[v][pivot]]
        else:
            candidates = list(p)
        
        for v in candidates:
            bron_kerbosch(r + [v], [u for u in p if adj[v][u]], [u for u in x if adj[v][u]])
            p = [u for u in p if u != v]
            x.append(v)
    
    bron_kerbosch([], all_vertices, [])
    
    # Find all maximal independent sets (cliques in complement)
    all_independent = []
    def bron_kerbosch_independent(r, p, x):
        if not p and not x:
            if len(r) > 1:
                all_independent.append(frozenset(r))
            return
        
        if p or x:
            pivot = max(list(p) + list(x), key=lambda v: len([u for u in p if not adj[v][u]]))
            candidates = [v for v in p if adj[v][pivot]]  # Non-neighbors in original = neighbors in complement
        else:
            candidates = list(p)
        
        for v in candidates:
            bron_kerbosch_independent(r + [v], [u for u in p if not adj[v][u]], [u for u in x if not adj[v][u]])
            p = [u for u in p if u != v]
            x.append(v)
    
    bron_kerbosch_independent([], all_vertices, [])
    
    # Also include all singletons (they are both cliques and independent sets)
    # And all pairs that are edges (cliques) or non-edges (independent)
    candidate_sets = set(all_cliques) | set(all_independent)
    
    # Add all maximal cliques/independent sets of size 1 (singletons)
    for v in range(n):
        candidate_sets.add(frozenset([v]))
    
    # Add all size-2 cliques (edges) and size-2 independent sets (non-edges)
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                candidate_sets.add(frozenset([i, j]))
            else:
                candidate_sets.add(frozenset([i, j]))
    
    # Add all size-3 sets that are cliques or independent
    for combo in combinations(range(n), 3):
        if is_clique(list(combo)) or is_independent(list(combo)):
            candidate_sets.add(frozenset(combo))
    
    # Sort candidates by size descending for better pruning
    candidates = sorted(candidate_sets, key=len, reverse=True)
    
    # Remove subsets of other candidates (keep only maximal ones for efficiency)
    # Actually, we want to keep all valid sets, but for branch and bound,
    # larger sets are better. Let's keep all but sort by size.
    
    best_solution = None
    best_count = n  # Upper bound: each vertex in its own part
    
    # Deadline guard
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s
    
    # Branch and bound
    def min_sets_to_cover(remaining_mask, current_count, current_parts, candidates_list):
        nonlocal best_solution, best_count, deadline
        
        if time.time() > deadline:
            return
        
        if current_count >= best_count:
            return
        
        if remaining_mask == 0:
            if current_count < best_count:
                best_count = current_count
                best_solution = list(current_parts)
            return
        
        # Lower bound: at least 1 set needed if any vertices remain
        # Better lower bound: find max size candidate that fits in remaining
        # and compute ceil(remaining_count / max_size)
        remaining_vertices = [i for i in range(n) if (remaining_mask >> i) & 1]
        rem_count = len(remaining_vertices)
        
        # Find the largest candidate that is a subset of remaining
        max_size = 1
        for c in candidates_list:
            if len(c) > max_size:
                c_mask = 0
                for v in c:
                    c_mask |= (1 << v)
                if c_mask & remaining_mask == c_mask:
                    max_size = len(c)
                    if max_size == rem_count:
                        break
        
        lb = (rem_count + max_size - 1) // max_size
        if current_count + lb >= best_count:
            return
        
        # Try each candidate that is a subset of remaining
        # Sort by size descending for better pruning
        for c in candidates_list:
            c_mask = 0
            for v in c:
                c_mask |= (1 << v)
            
            if c_mask & remaining_mask != c_mask:
                continue
            
            if time.time() > deadline:
                return
            
            new_mask = remaining_mask & ~c_mask
            current_parts.append(list(c))
            min_sets_to_cover(new_mask, current_count + 1, current_parts, candidates_list)
            current_parts.pop()
        
        return
    
    full_mask = (1 << n) - 1
    min_sets_to_cover(full_mask, 0, [], candidates)
    
    if best_solution is None:
        # Fallback: each vertex in its own part
        return [[i] for i in range(n)]
    
    # Convert to list of lists of ints
    result = []
    for part in best_solution:
        result.append([int(v) for v in part])
    
    return result
# EVOLVE-BLOCK-END
