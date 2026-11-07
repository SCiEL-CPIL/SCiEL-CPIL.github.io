"""Different strategy: set cover over edges using all maximal bicliques as candidate sets, with local search."""
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _get_biclique_edges(L, R, adj):
    """Return set of edges in biclique L x R. Only valid if all cross pairs are edges."""
    edges = set()
    for u in L:
        for v in R:
            if v in adj[u]:
                edges.add((min(u,v), max(u,v)))
            else:
                return None  # invalid biclique
    return edges if edges else None

def _generate_maximal_bicliques(adj):
    """Generate all maximal bicliques. For n=30, we can do this by trying all pairs of vertices as seeds."""
    n = P['n']
    maximal = []
    seen = set()
    
    # Try all possible (u, v) as initial biclique
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if v not in adj[u]:
                continue
            # Start with L={u}, R={v}
            L = {u}
            R = {v}
            # Expand greedily to maximal
            changed = True
            while changed:
                changed = False
                # Add vertices to L
                for x in range(n):
                    if x in L:
                        continue
                    # x can be added to L if x is connected to all in R
                    if all(y in adj[x] for y in R):
                        L.add(x)
                        changed = True
                # Add vertices to R
                for y in range(n):
                    if y in R:
                        continue
                    # y can be added to R if y is connected to all in L
                    if all(x in adj[y] for x in L):
                        R.add(y)
                        changed = True
            
            key = (frozenset(L), frozenset(R))
            # Check if this is a subset of an already found maximal biclique
            is_subset = False
            for (L2, R2) in seen:
                if L <= L2 and R <= R2:
                    is_subset = True
                    break
            if not is_subset:
                # Remove any bicliques that are subsets of this one
                new_seen = set()
                for (L2, R2) in seen:
                    if not (L2 <= L and R2 <= R):
                        new_seen.add((L2, R2))
                new_seen.add(key)
                seen = new_seen
                maximal.append((L, R))
    
    return maximal

def _greedy_set_cover(candidates, all_edges, adj):
    """Greedy set cover: pick candidate covering most uncovered edges."""
    uncovered = set(all_edges)
    selected = []
    cand_edges = []
    for L, R in candidates:
        e = _get_biclique_edges(L, R, adj)
        if e is not None:
            cand_edges.append((L, R, e))
    
    while uncovered:
        best = None
        best_count = -1
        for i, (L, R, e) in enumerate(cand_edges):
            count = len(e & uncovered)
            if count > best_count:
                best_count = count
                best = i
        if best is None or best_count == 0:
            break
        L, R, e = cand_edges[best]
        selected.append((L, R))
        uncovered -= e
        # Remove candidates that are subset of selected
        cand_edges.pop(best)
    
    return selected

def _local_search(bicliques, all_edges, adj, deadline):
    """Local search: try to replace one biclique with multiple smaller ones, or merge."""
    start_time = time.time()
    current = bicliques[:]
    
    def _edge_set(L, R):
        return _get_biclique_edges(L, R, adj)
    
    while time.time() < deadline - 5:
        improved = False
        # Try removing a biclique and covering its edges with others
        for i in range(len(current)):
            if time.time() > deadline - 5:
                break
            L_i, R_i = current[i]
            edges_i = _edge_set(L_i, R_i)
            if edges_i is None:
                continue
            # Can we cover edges_i with other bicliques?
            other_edges = set()
            for j in range(len(current)):
                if j == i:
                    continue
                L_j, R_j = current[j]
                e_j = _edge_set(L_j, R_j)
                if e_j is not None:
                    other_edges |= e_j
            if edges_i <= other_edges:
                # Remove biclique i
                new_current = current[:i] + current[i+1:]
                # Verify all edges still covered
                all_covered = set()
                for L, R in new_current:
                    e = _edge_set(L, R)
                    if e is not None:
                        all_covered |= e
                if all_covered >= all_edges:
                    current = new_current
                    improved = True
                    break
        if improved:
            continue
        # Try swapping: replace one biclique with a different one
        for i in range(len(current)):
            if time.time() > deadline - 5:
                break
            L_i, R_i = current[i]
            edges_i = _edge_set(L_i, R_i)
            if edges_i is None:
                continue
            # Try to find a replacement biclique that covers edges_i and maybe more
            # For simplicity, skip complex swaps for now
            pass
        if not improved:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    adj = _build_adj()
    all_edges = set((min(u,v), max(u,v)) for u, v in P['edges'])
    
    # Generate candidate bicliques
    candidates = _generate_maximal_bicliques(adj)
    
    if not candidates:
        # Fallback: star decomposition
        return [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
    
    # Greedy set cover
    solution = _greedy_set_cover(candidates, all_edges, adj)
    
    # Local search
    solution = _local_search(solution, all_edges, adj, deadline)
    
    # Convert to required format
    result = []
    for L, R in solution:
        result.append([list(L), list(R)])
    
    # Verify
    for L, R in result:
        for u in L:
            for v in R:
                if (min(u,v), max(u,v)) not in all_edges:
                    # Invalid - fallback
                    return [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
    
    return result
