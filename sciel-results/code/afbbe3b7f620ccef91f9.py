import time
import random
import itertools
import numpy as np

# Use the graph from the seed program
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _get_biclique_edges(left, right, adj):
    edges = set()
    for u in left:
        for v in right:
            if v in adj[u]:
                edges.add((min(u, v), max(u, v)))
    return edges

def _validate(bicliques, adj, all_edges_set):
    """Check that bicliques form a valid partition."""
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        # Check disjointness of sides within a biclique
        if set(left).intersection(right):
            return False
        # Check cross pairs are all edges
        for u in left:
            for v in right:
                if v not in adj[u]:
                    return False
                e = (min(u, v), max(u, v))
                if e in covered:
                    return False
                covered.add(e)
    # Check all edges covered
    if covered != all_edges_set:
        return False
    return True

def _expand_biclique(start_left, start_right, adj, n):
    """Expand a biclique to be maximal."""
    left = set(start_left)
    right = set(start_right)
    changed = True
    while changed:
        changed = False
        # Add to left
        for u in range(n):
            if u not in left and u not in right:
                if right.issubset(adj[u]):
                    left.add(u)
                    changed = True
        # Add to right
        for v in range(n):
            if v not in left and v not in right:
                if left.issubset(adj[v]):
                    right.add(v)
                    changed = True
    # Ensure disjoint
    right -= left
    left -= right
    if not left or not right:
        return None
    return (left, right)

def _generate_candidates(adj, n, all_edges_set, num_candidates=500):
    """Generate a pool of candidate bicliques."""
    candidates = set()
    
    # 1. Maximal bicliques containing each edge
    for e in all_edges_set:
        u, v = e
        # Try u in left, v in right
        bc = _expand_biclique({u}, {v}, adj, n)
        if bc:
            candidates.add((frozenset(bc[0]), frozenset(bc[1])))
        # Try v in left, u in right
        bc = _expand_biclique({v}, {u}, adj, n)
        if bc:
            candidates.add((frozenset(bc[0]), frozenset(bc[1])))
            
    # 2. Random expansions to find diverse candidates
    random.seed(42)
    for _ in range(200):
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v and v in adj[u]:
            bc = _expand_biclique({u}, {v}, adj, n)
            if bc:
                candidates.add((frozenset(bc[0]), frozenset(bc[1])))
        
        # Also try random pairs of vertices to start expansion
        l = random.randint(0, n-1)
        r = random.randint(0, n-1)
        if l != r:
            bc = _expand_biclique({l}, {r}, adj, n)
            if bc:
                candidates.add((frozenset(bc[0]), frozenset(bc[1])))

    # Convert to list of (left_set, right_set, edge_set)
    cand_list = []
    for l, r in candidates:
        edges = _get_biclique_edges(l, r, adj)
        if edges:
            cand_list.append((set(l), set(r), edges))
            
    return cand_list

def _greedy_disjoint_cover(candidates, all_edges_set, deadline):
    """Greedy select disjoint bicliques to cover all edges."""
    remaining = set(all_edges_set)
    selected = []
    
    # Sort candidates by size descending
    candidates_sorted = sorted(candidates, key=lambda x: len(x[2]), reverse=True)
    
    while remaining:
        if time.time() > deadline:
            break
        best_cand = None
        best_new_edges = 0
        
        for l, r, edges in candidates_sorted:
            # Check disjointness with already selected
            disjoint = True
            for sel_l, sel_r, sel_edges in selected:
                if edges.intersection(sel_edges):
                    disjoint = False
                    break
            if not disjoint:
                continue
            
            new_edges = edges.difference(_union_edges(selected))
            if len(new_edges) > best_new_edges:
                best_new_edges = len(new_edges)
                best_cand = (l, r, edges)
                
        if best_cand is None:
            # Fallback: pick a single edge
            e = next(iter(remaining))
            u, v = e
            best_cand = ({u}, {v}, {e})
            
        selected.append(best_cand)
        remaining -= best_cand[2]
        
    return [(l, r) for l, r, _ in selected]

def _union_edges(selected):
    u = set()
    for l, r, e in selected:
        u.update(e)
    return u

def _local_search_merge(bicliques, adj, all_edges_set, candidates, deadline):
    """Try to reduce number of bicliques by merging 2 into 1."""
    bicliques = [list(bc) for bc in bicliques]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_bicliques = len(bicliques)
        for i in range(n_bicliques):
            for j in range(i+1, n_bicliques):
                if time.time() > deadline:
                    break
                li, ri = bicliques[i]
                lj, rj = bicliques[j]
                
                edges_i = _get_biclique_edges(li, ri, adj)
                edges_j = _get_biclique_edges(lj, rj, adj)
                target_edges = edges_i.union(edges_j)
                
                # Try to find a biclique in candidates that covers target_edges
                # and is disjoint from all other bicliques
                found = False
                for cl, cr, ce in candidates:
                    if not target_edges.issubset(ce):
                        continue
                    # Check disjointness with others
                    disjoint = True
                    for k in range(n_bicliques):
                        if k == i or k == j:
                            continue
                        lk, rk = bicliques[k]
                        ek = _get_biclique_edges(lk, rk, adj)
                        if ce.intersection(ek):
                            disjoint = False
                            break
                    if disjoint:
                        # Replace i and j with this candidate
                        bicliques[i] = [list(cl), list(cr)]
                        del bicliques[j]
                        improved = True
                        found = True
                        break
                if found:
                    break
            if improved:
                break
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 50 # Safety margin
    
    n = P['n']
    adj = _build_adjacency()
    all_edges_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    
    # Generate candidates
    candidates = _generate_candidates(adj, n, all_edges_set, 1000)
    
    best_result = None
    best_count = float('inf')
    
    # Initial greedy cover
    bicliques = _greedy_disjoint_cover(candidates, all_edges_set, deadline)
    
    if bicliques:
        if _validate(bicliques, adj, all_edges_set):
            best_result = bicliques
            best_count = len(bicliques)
    
    # Local search improvement
    if best_result and time.time() < deadline:
        improved = _local_search_merge(best_result, adj, all_edges_set, candidates, deadline)
        if _validate(improved, adj, all_edges_set) and len(improved) < best_count:
            best_result = improved
            best_count = len(improved)
            
    # Try a few random restarts of the greedy phase if time permits
    if time.time() < deadline:
        for trial in range(5):
            if time.time() > deadline:
                break
            # Shuffle candidates to get different greedy picks
            random.shuffle(candidates)
            bicliques = _greedy_disjoint_cover(candidates, all_edges_set, deadline)
            if bicliques and _validate(bicliques, adj, all_edges_set):
                if len(bicliques) < best_count:
                    best_count = len(bicliques)
                    best_result = bicliques
                    # Local search on this new best
                    if time.time() < deadline:
                        improved = _local_search_merge(best_result, adj, all_edges_set, candidates, deadline)
                        if _validate(improved, adj, all_edges_set) and len(improved) < best_count:
                            best_result = improved
                            best_count = len(improved)

    # Final fallback
    if best_result is None:
        best_result = []
        for u, v in P['edges']:
            best_result.append([u, v]) # This is invalid format, need [left, right]
        # Proper fallback: single edge bicliques
        best_result = [[[u], [v]] for u, v in P['edges']]
        
    # Convert to proper types
    result = []
    for left, right in best_result:
        result.append([list(left), list(right)])
        
    return result
