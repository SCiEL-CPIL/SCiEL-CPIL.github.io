"""Improved packing of trees T1..T20 into K20."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def _norm_edge(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    trees = instance_trees(20, 611)
    
    # Track used edges in K20 (normalized)
    used_edges = set()
    result = [None] * 20
    
    # Process trees from smallest to largest (less constrained first)
    # This gives more flexibility for larger trees
    order = list(range(20))  # indices 0 up to 19 (T1 first)
    
    for idx in order:
        if time.time() > deadline:
            break
        k = idx + 1  # tree size
        tree_edges = trees[idx]
        
        if k == 1:
            # T1: single vertex, no edges. Pick any vertex.
            for v in range(20):
                result[idx] = [v]
                break
            continue
        
        if k == 2:
            # T2: single edge (0,1). Find any unused edge.
            found = False
            for u in range(20):
                for w in range(u + 1, 20):
                    e = _norm_edge(u, w)
                    if e not in used_edges:
                        result[idx] = [u, w]
                        used_edges.add(e)
                        found = True
                        break
                if found:
                    break
            if not found:
                result[idx] = None
            continue
        
        # For k >= 3, use backtracking with forward checking
        tree_n = k
        tree_edge_list = tree_edges
        
        # Build adjacency for the tree
        tree_adj = [[] for _ in range(tree_n)]
        for u, v in tree_edge_list:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        
        # Get vertex ordering: start with highest degree vertex, then BFS
        degrees = [len(tree_adj[v]) for v in range(tree_n)]
        # Find vertex with max degree (tie-break by lowest index)
        root = max(range(tree_n), key=lambda v: (degrees[v], -v))
        
        # BFS from root
        visited = [False] * tree_n
        bfs_order = [root]
        visited[root] = True
        q = deque([root])
        while q:
            u = q.popleft()
            for v in tree_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    bfs_order.append(v)
                    q.append(v)
        
        # For each tree vertex, precompute which tree neighbors are "before" it in BFS order
        pos_in_bfs = [0] * tree_n
        for i, v in enumerate(bfs_order):
            pos_in_bfs[v] = i
        
        # assignment[tree_v] = host_v or None
        assignment = [None] * tree_n
        # Track which host vertices are used
        host_used = [False] * 20
        # Temporarily used edges during backtracking
        temp_used_edges = set()
        
        # Precompute for each position, which neighbors are already assigned
        pre_assigned_neighbors = [[] for _ in range(tree_n)]
        for pos in range(tree_n):
            tv = bfs_order[pos]
            for nb in tree_adj[tv]:
                if pos_in_bfs[nb] < pos:
                    pre_assigned_neighbors[pos].append(nb)
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == tree_n:
                return True
            
            tree_v = bfs_order[pos]
            # Try each unused host vertex
            for host_v in range(20):
                if host_used[host_v]:
                    continue
                
                # Check if all edges from already-assigned neighbors are valid
                valid = True
                for nb in pre_assigned_neighbors[pos]:
                    e = _norm_edge(assignment[nb], host_v)
                    if e in used_edges or e in temp_used_edges:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Tentatively assign
                assignment[tree_v] = host_v
                host_used[host_v] = True
                # Mark edges as temporarily used
                added_edges = []
                for nb in pre_assigned_neighbors[pos]:
                    e = _norm_edge(assignment[nb], host_v)
                    temp_used_edges.add(e)
                    added_edges.append(e)
                
                if backtrack(pos + 1):
                    # Commit the edges
                    for e in added_edges:
                        used_edges.add(e)
                    return True
                
                # Undo
                for e in added_edges:
                    temp_used_edges.discard(e)
                host_used[host_v] = False
                assignment[tree_v] = None
        
        if backtrack(0):
            result[idx] = assignment[:]
        else:
            result[idx] = None
    
    return result
# EVOLVE-BLOCK-END
