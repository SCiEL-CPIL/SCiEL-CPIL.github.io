"""Rewritten program: longest simple path via matching-based seed + local search."""
import random
from collections import deque
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    # Build adjacency matrix for scipy
    # Bipartite graph: Left nodes 0..n-1 (out), Right nodes 0..n-1 (in)
    # Edge (u, v) in G becomes edge (u, v) in bipartite graph if u != v
    row = []
    col = []
    for u, v in P['edges']:
        if u != v:
            row.append(u)
            col.append(v)
            
    # Ensure we have enough entries if no edges (though N=100, E~90)
    # scipy requires int arrays
    row = np.array(row, dtype=int)
    col = np.array(col, dtype=int)
    
    # Create sparse matrix
    # shape is (n, n)
    # We need to be careful: maximum_bipartite_matching expects a bipartite graph.
    # The scipy function takes a square matrix where A[i,j] indicates edge from left i to right j.
    # It finds a maximum matching.
    
    # Initialize with zeros
    data = np.ones(len(row))
    A = csr_matrix((data, (row, col)), shape=(n, n))
    
    # Find maximum bipartite matching
    # Returns left_indices, right_indices
    left, right = maximum_bipartite_matching(A)
    
    # left[i] is the right node matched to left node i, or -1
    # right[j] is the left node matched to right node j, or -1
    
    # Reconstruct paths from the matching
    # Each vertex has at most one outgoing (from left) and one incoming (to right).
    # This forms a set of disjoint directed paths and cycles.
    
    # next_node[u] = v if u is matched to v (outgoing from u)
    # prev_node[v] = u if v is matched from u (incoming to v)
    next_node = [-1] * n
    prev_node = [-1] * n
    
    for u in range(n):
        if left[u] != -1:
            v = left[u]
            next_node[u] = v
            prev_node[v] = u
            
    # Find all components (paths and cycles)
    # A path starts at a node with prev_node[start] == -1
    # A cycle has no start/end in this sense, but we can break it.
    
    components = []
    visited = [False] * n
    
    for i in range(n):
        if prev_node[i] == -1 and next_node[i] != -1:
            # Start of a path
            path = [i]
            cur = i
            while next_node[cur] != -1:
                cur = next_node[cur]
                path.append(cur)
            components.append(path)
            for node in path:
                visited[node] = True
                
    # Handle cycles (nodes where prev_node and next_node are both not -1, and not visited)
    # Also handle single nodes? No, matching edges imply at least 2 nodes per component if they are part of the matching.
    # Unmatched nodes are isolated.
    
    for i in range(n):
        if not visited[i] and next_node[i] != -1:
            # This is part of a cycle
            path = [i]
            cur = i
            while next_node[cur] != -1:
                cur = next_node[cur]
                path.append(cur)
                if cur == i:
                    break
            # The path is a cycle. We can break it to make a path.
            # The matching doesn't specify direction, but we followed next_node.
            # This path is valid.
            components.append(path)
            for node in path:
                visited[node] = True
                
    # Select the longest component as seed
    best_seed = []
    for comp in components:
        if len(comp) > len(best_seed):
            best_seed = comp
            
    # If no components (empty matching), fallback to a single node
    if not best_seed:
        best_seed = [0]
        
    # Now, use the existing local search logic but start with this strong seed
    # We will also try to extend this seed and improve it.
    
    def extend_path(path):
        path = list(path)
        changed = True
        while changed:
            changed = False
            # Try extending at the front
            front = path[0]
            candidates = [v for v in adj[front] if v not in set(path)]
            if candidates:
                best_v = None
                best_ext = 0
                for v in candidates:
                    ext = [v]
                    seen = {v} | set(path)
                    while True:
                        next_cands = [w for w in adj[ext[-1]] if w not in seen]
                        if not next_cands:
                            break
                        w = min(next_cands, key=lambda x: (len(adj[x] - seen), x))
                        ext.append(w)
                        seen.add(w)
                    if len(ext) > best_ext:
                        best_ext = len(ext)
                        best_v = v
                if best_v is not None:
                    path.insert(0, best_v)
                    changed = True
            # Try extending at the back
            back = path[-1]
            candidates = [v for v in adj[back] if v not in set(path)]
            if candidates:
                best_v = None
                best_ext = 0
                for v in candidates:
                    ext = [v]
                    seen = {v} | set(path)
                    while True:
                        next_cands = [w for w in adj[ext[-1]] if w not in seen]
                        if not next_cands:
                            break
                        w = min(next_cands, key=lambda x: (len(adj[x] - seen), x))
                        ext.append(w)
                        seen.add(w)
                    if len(ext) > best_ext:
                        best_ext = len(ext)
                        best_v = v
                if best_v is not None:
                    path.append(best_v)
                    changed = True
        return path
        
    def try_improve(path):
        path = list(path)
        n_path = len(path)
        if n_path < 3:
            return path
        improved = True
        while improved:
            improved = False
            for i in range(n_path - 2):
                for j in range(i + 2, n_path):
                    new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                    valid = True
                    for k in range(len(new_path) - 1):
                        if new_path[k+1] not in adj[new_path[k]]:
                            valid = False
                            break
                    if valid and len(new_path) > len(path):
                        path = new_path
                        improved = True
                        break
                if improved:
                    break
        return path

    # Start with the matching-derived seed
    best_path = extend_path(best_seed)
    best_path = try_improve(best_path)
    best_path = extend_path(best_path)
    
    # Also try the old greedy strategies to see if they beat the matching seed
    def greedy_path(start, strategy='warnsdorff'):
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            cands = [v for v in adj[cur] if v not in seen]
            if not cands:
                break
            if strategy == 'warnsdorff':
                v = min(cands, key=lambda w: (len(adj[w] - seen), w))
            elif strategy == 'min_degree':
                v = min(cands, key=lambda w: (len(adj[w]), w))
            elif strategy == 'max_degree':
                v = max(cands, key=lambda w: (len(adj[w]), w))
            else:
                v = cands[0]
            path.append(v)
            seen.add(v)
        return path
        
    for start in range(n):
        for strat in ['warnsdorff', 'min_degree', 'max_degree']:
            p = greedy_path(start, strat)
            p = extend_path(p)
            p = try_improve(p)
            p = extend_path(p)
            if len(p) > len(best_path):
                best_path = p

    # Final validation
    seen = set()
    valid_path = []
    for v in best_path:
        if v not in seen:
            valid_path.append(v)
            seen.add(v)
    final_path = [valid_path[0]]
    for i in range(1, len(valid_path)):
        if valid_path[i] in adj[final_path[-1]]:
            final_path.append(valid_path[i])
            
    if not final_path:
        final_path = [0]
        
    return final_path
