"""Biclique partition via exact set cover on maximal bicliques with greedy fallback."""
# EVOLVE-BLOCK-START
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _find_all_maximal_bicliques(n, edge_set):
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
        
    max_bics = set()
    visited_states = set()
    
    def expand_memo(L, R):
        L_f = frozenset(L)
        R_f = frozenset(R)
        state = (L_f, R_f)
        if state in visited_states:
            return
        visited_states.add(state)
        
        can_add_L = []
        can_add_R = []
        
        for w in range(n):
            if w in L or w in R:
                continue
            if all(r in adj[w] for r in R):
                can_add_L.append(w)
                
        for w in range(n):
            if w in L or w in R:
                continue
            if all(l in adj[w] for l in L):
                can_add_R.append(w)
                
        if not can_add_L and not can_add_R:
            max_bics.add((L_f, R_f))
            return
            
        for w in can_add_L:
            expand_memo(L | {w}, R)
            
        for w in can_add_R:
            expand_memo(L, R | {w})

    for u, v in edge_set:
        expand_memo({u}, {v})
        
    return list(max_bics)

def _greedy_set_cover(edge_set, biclique_edges):
    remaining = set(edge_set)
    selected = []
    available = list(range(len(biclique_edges)))
    
    while remaining and available:
        best_idx = -1
        best_cover = 0
        for i in available:
            covered = len(remaining & biclique_edges[i][2])
            if covered > best_cover:
                best_cover = covered
                best_idx = i
        if best_idx == -1:
            break
        L, R, b_edges = biclique_edges[best_idx]
        selected.append((L, R))
        remaining -= b_edges
        available.remove(best_idx)
        
    # Fallback for any remaining edges (should not happen if bicliques cover all)
    for e in remaining:
        u, v = e
        selected.append((frozenset({u}), frozenset({v})))
        
    return selected

def _solve_set_cover(n, edge_set, bicliques):
    if not edge_set:
        return []
        
    biclique_edges = []
    for L, R in bicliques:
        edges = set()
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e in edge_set:
                    edges.add(e)
        if edges:
            biclique_edges.append((L, R, frozenset(edges)))
            
    if not biclique_edges:
        return []

    # Greedy initial solution
    greedy_sol = _greedy_set_cover(edge_set, biclique_edges)
    best_count = len(greedy_sol)
    best_solution = greedy_sol
    
    # Sort by size descending
    biclique_edges.sort(key=lambda x: len(x[2]), reverse=True)
    
    edge_to_bics = {e: [] for e in edge_set}
    for i, (_, _, b_edges) in enumerate(biclique_edges):
        for e in b_edges:
            edge_to_bics[e].append(i)
            
    nodes_visited = 0
    limit = 500000 # Node limit
    
    def lower_bound(remaining_edges):
        if not remaining_edges:
            return 0
        max_size = max((len(b[2]) for b in biclique_edges), default=1)
        if max_size == 0:
            return 0
        return (len(remaining_edges) + max_size - 1) // max_size
        
    def solve(remaining_edges, available_set, current_count, current_sel):
        nonlocal best_solution, best_count, nodes_visited
        
        if nodes_visited > limit:
            return
        nodes_visited += 1
        
        if not remaining_edges:
            if current_count < best_count:
                best_count = current_count
                best_solution = list(current_sel)
            return
            
        if current_count >= best_count:
            return
            
        lb = lower_bound(remaining_edges)
        if current_count + lb >= best_count:
            return
            
        # Pick edge with fewest options
        min_options = len(edge_to_bics[next(iter(remaining_edges))])
        target_edge = next(iter(remaining_edges))
        for e in remaining_edges:
            if len(edge_to_bics[e]) < min_options:
                min_options = len(edge_to_bics[e])
                target_edge = e
                if min_options == 1:
                    break
                    
        candidates = [i for i in available_set if target_edge in biclique_edges[i][2]]
        candidates.sort(key=lambda i: len(biclique_edges[i][2]), reverse=True)
        
        for i in candidates:
            L, R, b_edges = biclique_edges[i]
            new_remaining = remaining_edges - b_edges
            new_available = available_set - {i}
            
            current_sel.append((L, R))
            solve(new_remaining, new_available, current_count + 1, current_sel)
            current_sel.pop()
            
    all_indices = set(range(len(biclique_edges)))
    solve(frozenset(edge_set), all_indices, 0, [])
    
    return best_solution

def construct():
    n = P['n']
    edge_set = set()
    for a, b in P['edges']:
        edge_set.add((min(a, b), max(a, b)))
        
    max_bics = _find_all_maximal_bicliques(n, edge_set)
    solution = _solve_set_cover(n, edge_set, max_bics)
    
    result = []
    for L, R in solution:
        L_list = [int(x) for x in sorted(L)]
        R_list = [int(x) for x in sorted(R)]
        result.append([L_list, R_list])
        
    return result
# EVOLVE-BLOCK-END
