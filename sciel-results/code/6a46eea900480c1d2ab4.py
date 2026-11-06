import time
import random
from collections import defaultdict

_DEADLINE = 135.0
_START = time.monotonic()

def _elapsed():
    return time.monotonic() - _START

def _random_graph(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _validate(n, edges, cliques):
    edge_set = set(edges)
    seen = set()
    for c in cliques:
        if len(c) < 2:
            return False
        k = len(c)
        for i in range(k):
            a = c[i]
            for j in range(i + 1, k):
                b = c[j]
                e = (a, b) if a < b else (b, a)
                if e not in edge_set:
                    return False
                if e in seen:
                    return False
                seen.add(e)
    return seen == edge_set

def _bron_kerbosch(adj, n):
    """
    Enumerates all maximal cliques in a graph given by adjacency bitmasks.
    Returns a list of lists of vertices.
    """
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(list(R))
            return
        
        # Choose a pivot u from P union X
        # Heuristic: pick u with max degree in P
        union_px = P | X
        if union_px:
            # Pick a pivot
            # Iterate to find best pivot
            best_u = -1
            best_deg = -1
            temp = union_px
            while temp:
                u = (temp & -temp).bit_length() - 1
                deg = (adj[u] & P).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_u = u
                temp &= temp - 1
            
            # Candidates are P \ N(u)
            candidates = P & ~adj[best_u]
        else:
            candidates = P
            
        temp = candidates
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            if _elapsed() > _DEADLINE:
                return # Timeout in enumeration
            
            bk(R | (1 << v), P & adj[v], X & adj[v])
            P &= ~(1 << v)
            X |= (1 << v)
            
            if _elapsed() > _DEADLINE:
                return

    bk(0, (1 << n) - 1, 0)
    return cliques

def _solve_instance(n, p, seed):
    edges = _random_graph(n, p, seed)
    edge_set = set(edges)
    if not edges:
        return []
    
    # Build adjacency masks
    adj = [0] * n
    for (u, v) in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # 1. Enumerate Maximal Cliques
    if _elapsed() > _DEADLINE - 10:
        # Fallback to greedy if no time
        return _greedy_fallback(n, p, seed, adj, edges)
        
    max_cliques = _bron_kerbosch(adj, n)
    
    # If no cliques found (shouldn't happen for dense graphs with edges), fallback
    if not max_cliques:
        return _greedy_fallback(n, p, seed, adj, edges)
        
    # 2. Exact Cover via Backtracking
    # Map edges to indices
    edge_to_idx = {}
    for i, e in enumerate(edges):
        edge_to_idx[e] = i
        
    # For each clique, get the bitmask of edges it covers
    # And a list of edge indices
    clique_edge_masks = []
    clique_edge_indices = []
    
    for c in max_cliques:
        mask = 0
        indices = []
        k = len(c)
        for i in range(k):
            u = c[i]
            for j in range(i+1, k):
                v = c[j]
                e = (u, v) if u < v else (v, u)
                idx = edge_to_idx[e]
                mask |= (1 << idx)
                indices.append(idx)
        clique_edge_masks.append(mask)
        clique_edge_indices.append(indices)
        
    # Precompute: for each edge, which cliques cover it?
    # We need to check conflicts.
    # If we pick clique i, we cover edges in mask_i.
    # These edges must not be covered by any previously picked clique.
    
    total_edges = len(edges)
    full_mask = (1 << total_edges) - 1
    covered_mask = 0
    current_cliques = []
    
    # Heuristic: Sort cliques by size descending?
    # Or use MRV: pick the edge with fewest candidate cliques.
    
    # Let's use a recursive backtracking with MRV.
    # State: covered_mask, list of chosen cliques
    # To speed up, we can precompute for each edge the list of clique indices containing it.
    edge_to_cliques = defaultdict(list)
    for i, mask in enumerate(clique_edge_masks):
        # Iterate set bits
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            edge_to_cliques[idx].append(i)
            m &= m - 1
            
    # Sort cliques by size (larger first) to prefer covering more
    clique_sizes = [len(c) for c in max_cliques]
    sorted_clique_indices = sorted(range(len(max_cliques)), key=lambda i: -clique_sizes[i])
    
    best_solution = None
    best_size = float('inf')
    
    def backtrack(covered_mask, chosen):
        nonlocal best_solution, best_size
        
        if _elapsed() > _DEADLINE - 5:
            return
            
        if covered_mask == full_mask:
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_solution = list(chosen)
            return
            
        # MRV: Find an uncovered edge with the fewest candidate cliques
        # We need to check which edges are uncovered.
        # Uncovered edges are those NOT in covered_mask.
        
        # To find the edge with min candidates, we iterate all edges?
        # That is O(E * C) which might be slow.
        # Optimization: Only consider edges that are uncovered.
        
        min_count = float('inf')
        min_edge_idx = -1
        min_candidates = []
        
        # Iterate through all edges to find one that is uncovered
        # This is O(E).
        for e_idx in range(total_edges):
            if (covered_mask >> e_idx) & 1:
                continue # Already covered
            
            # Get candidates
            candidates = edge_to_cliques[e_idx]
            
            # Filter candidates that are compatible with current covered_mask
            # A candidate c is compatible if (clique_edge_masks[c] & covered_mask) == 0
            
            valid_candidates = []
            for c_idx in candidates:
                if (clique_edge_masks[c_idx] & covered_mask) == 0:
                    valid_candidates.append(c_idx)
                    
            if not valid_candidates:
                return # Dead end
                
            if len(valid_candidates) < min_count:
                min_count = len(valid_candidates)
                min_edge_idx = e_idx
                min_candidates = valid_candidates
                
                if min_count == 1:
                    break # Can't do better than 1
                    
        if min_edge_idx == -1:
            return # No uncovered edges? Should have been caught by full_mask check
            
        # Sort candidates by size (larger first) to prune faster
        min_candidates.sort(key=lambda i: -clique_sizes[i])
        
        for c_idx in min_candidates:
            new_mask = covered_mask | clique_edge_masks[c_idx]
            chosen.append(c_idx)
            backtrack(new_mask, chosen)
            chosen.pop()
            
            if _elapsed() > _DEADLINE - 5:
                return

    backtrack(0, [])
    
    if best_solution:
        return [max_cliques[i] for i in best_solution]
    else:
        # Fallback if backtracking fails or times out
        return _greedy_fallback(n, p, seed, adj, edges)

def _greedy_fallback(n, p, seed, adj, edges):
    """
    Greedy maximal clique packing.
    """
    used = set()
    cliques = []
    
    # Sort edges by triangle count (constraint)
    edge_tri_count = {}
    for (u, v) in edges:
        if u > v:
            u, v = v, u
        cand = adj[u] & adj[v]
        cand &= ~((1 << (v + 1)) - 1)
        edge_tri_count[(u, v)] = cand.bit_count()
        
    sorted_edges = sorted(edges, key=lambda e: edge_tri_count.get((min(e[0], e[1]), max(e[0], e[1])), 0))
    
    for (u, v) in sorted_edges:
        if _elapsed() > _DEADLINE:
            break
        if (u, v) in used:
            continue
            
        max_uv = max(u, v)
        current_clique = [u, v]
        cand = adj[u] & adj[v]
        cand &= ~((1 << (max_uv + 1)) - 1)
        
        while cand and _elapsed() < _DEADLINE:
            best_w = -1
            best_deg = float('inf')
            temp = cand
            while temp:
                w = (temp & -temp).bit_length() - 1
                deg = adj[w].bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_w = w
                temp &= temp - 1
                
            if best_w < 0:
                break
                
            can_add = True
            for x in current_clique:
                if not (adj[best_w] >> x) & 1:
                    can_add = False
                    break
                    
            if can_add:
                current_clique.append(best_w)
                cand = cand & adj[best_w]
                new_max = max(current_clique)
                cand &= ~((1 << (new_max + 1)) - 1)
            else:
                cand &= ~(1 << best_w)
                
        k = len(current_clique)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = current_clique[i], current_clique[j]
                e = (a, b) if a < b else (b, a)
                used.add(e)
                
        cliques.append(current_clique)
        
    for (u, v) in edges:
        if (u, v) not in used:
            cliques.append([u, v])
            
    if not _validate(n, edges, cliques):
        cliques = [[u, v] for (u, v) in edges]
        
    return cliques

def construct():
    results = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
