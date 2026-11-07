import time
import random
from collections import deque

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _ek(e):
    a, b = e
    return (a, b) if a < b else (b, a)

def _adj(n, edges):
    A = [set() for _ in range(n)]
    for u, v in edges:
        A[u].add(v)
        A[v].add(u)
    return A

def _grow_biclique(seed_l, seed_r, edge_set, adj, n):
    left = set(seed_l)
    right = set(seed_r)
    if not left or not right or (left & right):
        return None
    # Verify
    for u in left:
        for v in right:
            if _ek((u, v)) not in edge_set:
                return None
    changed = True
    while changed:
        changed = False
        for w in range(n):
            if w in left or w in right:
                continue
            if right.issubset(adj[w]):
                left.add(w)
                changed = True
        for w in range(n):
            if w in left or w in right:
                continue
            if left.issubset(adj[w]):
                right.add(w)
                changed = True
    return (frozenset(left), frozenset(right))

def _biclique_edges(l, r):
    return frozenset(_ek((u, v)) for u in l for v in r)

def _is_biclique_edge_set(edges, adj, n):
    # Check if a set of edges forms a biclique
    # 1. Get vertices
    vs = set()
    for u, v in edges:
        vs.add(u)
        vs.add(v)
    if not vs:
        return None
    # 2. Check connectedness of the graph induced by these edges
    # Build adjacency list for this subgraph
    sub_adj = {v: set() for v in vs}
    for u, v in edges:
        sub_adj[u].add(v)
        sub_adj[v].add(u)
    
    # BFS
    start = next(iter(vs))
    visited = {start}
    queue = [start]
    while queue:
        u = queue.pop(0)
        for v in sub_adj[u]:
            if v not in visited:
                visited.add(v)
                queue.append(v)
    if len(visited) != len(vs):
        return None # Not connected, cannot be a single biclique (unless empty, but we handle that)
        
    # 3. Check Bipartite
    color = {start: 0}
    queue = [start]
    while queue:
        u = queue.pop(0)
        for v in sub_adj[u]:
            if v not in color:
                color[v] = 1 - color[u]
                queue.append(v)
            elif color[v] == color[u]:
                return None # Not bipartite

    # 4. Check Complete Bipartite
    # Partitions
    L = {v for v in vs if color[v] == 0}
    R = {v for v in vs if color[v] == 1}
    
    # Check if all pairs in L x R are in edges
    for u in L:
        for v in R:
            if _ek((u, v)) not in edges:
                return None
                
    return (L, R)

def _generate_candidates(edge_set, adj, n, edges):
    candidates = set()
    # Grow from every edge
    for u, v in edges:
        bc1 = _grow_biclique({u}, {v}, edge_set, adj, n)
        if bc1:
            candidates.add(bc1)
        bc2 = _grow_biclique({v}, {u}, edge_set, adj, n)
        if bc2:
            candidates.add(bc2)
            
    # Convert to list of (edges_frozenset, L, R)
    cand_list = []
    for l, r in candidates:
        e = _biclique_edges(l, r)
        cand_list.append((e, l, r))
    return cand_list

def _greedy_set_cover(cand_list, all_edges):
    remaining = set(all_edges)
    selected = []
    
    while remaining:
        best_idx = -1
        best_overlap = 0
        for i, (e, l, r) in enumerate(cand_list):
            # Optimization: if e is subset of remaining, overlap is len(e)
            # else len(e & remaining)
            overlap = len(e & remaining)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        if best_idx == -1:
            break # Should not happen if candidates cover all edges
        e, l, r = cand_list[best_idx]
        selected.append((l, r))
        remaining -= e
    return selected

def _local_search(selected, cand_list, all_edges, adj, n, deadline):
    # selected is list of (L, R)
    # We want to minimize len(selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Remove redundant
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            e_i = _biclique_edges(selected[i][0], selected[i][1])
            # Check if e_i is covered by others
            covered = set()
            for j in range(len(selected)):
                if i == j:
                    continue
                e_j = _biclique_edges(selected[j][0], selected[j][1])
                covered |= e_j
            if e_i.issubset(covered):
                selected.pop(i)
                improved = True
                break
        
        if improved:
            continue

        # 2. Merge
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(selected)):
                if time.time() > deadline:
                    break
                l_i, r_i = selected[i]
                l_j, r_j = selected[j]
                e_i = _biclique_edges(l_i, r_i)
                e_j = _biclique_edges(l_j, r_j)
                e_union = e_i | e_j
                
                # Check if e_union is a biclique
                new_bc = _is_biclique_edge_set(e_union, adj, n)
                if new_bc:
                    new_l, new_r = new_bc
                    # Replace i, j with new_bc
                    selected.pop(j)
                    selected.pop(i)
                    selected.append((new_l, new_r))
                    improved = True
                    break
            if improved:
                break
                
        # 3. Swap (optional, might be slow)
        # Try to replace selected[i] with a candidate c such that e_i is subset of e_c
        # and this allows removing another biclique?
        # For now, skip complex swaps.
        
    return selected

def _refine_partition(partition, cand_list, all_edges, adj, n, deadline):
    """
    Try to improve the partition by replacing bicliques with larger ones from cand_list
    that cover the same edges plus more, allowing removal of other bicliques.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Compute current edge sets
        current_edges = [_biclique_edges(l, r) for l, r in partition]
        
        for i in range(len(partition)):
            if time.time() > deadline:
                break
            # Try replacing partition[i] with a larger candidate
            for c_idx, (c_e, c_l, c_r) in enumerate(cand_list):
                # Candidate must cover all edges of partition[i]
                if not current_edges[i].issubset(c_e):
                    continue
                # If candidate is the same size, skip
                if len(c_e) == len(current_edges[i]):
                    continue
                
                # Create new partition
                new_partition = [p for j, p in enumerate(partition) if j != i]
                new_partition.append((c_l, c_r))
                
                # Remove redundant bicliques
                new_edges = [_biclique_edges(l, r) for l, r in new_partition]
                changed = True
                while changed:
                    changed = False
                    for k in range(len(new_partition)):
                        covered = set()
                        for m in range(len(new_partition)):
                            if m != k:
                                covered |= new_edges[m]
                        if new_edges[k].issubset(covered):
                            new_partition.pop(k)
                            new_edges.pop(k)
                            changed = True
                            break
                
                # Check if all edges are still covered
                total_covered = set()
                for e in new_edges:
                    total_covered |= e
                
                if total_covered == all_edges and len(new_partition) < len(partition):
                    partition = new_partition
                    improved = True
                    break
            if improved:
                break
    return partition

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    all_edge_set = frozenset(_ek(e) for e in edges)
    adj = _adj(n, edges)
    
    # 1. Generate Candidates
    cand_list = _generate_candidates(all_edge_set, adj, n, edges)
    
    # 2. Greedy Set Cover
    selected = _greedy_set_cover(cand_list, all_edge_set)
    
    # 3. Local Search
    selected = _local_search(selected, cand_list, all_edge_set, adj, n, deadline)
    
    # 4. Refine by replacing with larger bicliques
    selected = _refine_partition(selected, cand_list, all_edge_set, adj, n, deadline)
    
    # 5. Final validation
    covered = set()
    valid = True
    for l, r in selected:
        if not l or not r:
            valid = False
            break
        if l & r:
            valid = False
            break
        e = _biclique_edges(l, r)
        if not e.issubset(all_edge_set):
            valid = False
            break
        if e & covered:
            valid = False
            break
        covered |= e
    if covered != all_edge_set:
        valid = False
        
    if not valid:
        # Fallback
        selected = [({u}, {v}) for u, v in edges]
    
    # Convert to output format
    result = []
    for l, r in selected:
        result.append([list(l), list(r)])
        
    return result
