"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    trees = instance_trees(20, 611)
    
    # Process trees from largest to smallest (most constrained first)
    order = list(range(19, -1, -1))  # indices 19 down to 0 (T20 first)
    
    # Track used edges in K20 (as frozensets of vertex pairs)
    used_edges = set()
    result = [None] * 20
    
    # Build adjacency for each tree once
    tree_adjs = []
    for idx in range(20):
        k = idx + 1
        tree_edges = trees[idx]
        if k <= 2:
            tree_adjs.append([])
            continue
        tree_adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        tree_adjs.append(tree_adj)
    
    # Get BFS order for each tree
    tree_bfs_orders = []
    for idx in range(20):
        k = idx + 1
        if k <= 2:
            tree_bfs_orders.append(list(range(k)))
            continue
        tree_adj = tree_adjs[idx]
        visited = [False] * k
        bfs_order = [0]
        visited[0] = True
        q = deque([0])
        while q:
            u = q.popleft()
            for v in tree_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    bfs_order.append(v)
                    q.append(v)
        tree_bfs_orders.append(bfs_order)
    
    for idx in order:
        k = idx + 1  # tree size
        tree_edges = trees[idx]
        tree_adj = tree_adjs[idx]
        bfs_order = tree_bfs_orders[idx]
        
        if time.time() > deadline:
            break
        
        if k == 1:
            for v in range(20):
                result[idx] = [v]
                break
            continue
        
        if k == 2:
            found = False
            for u in range(20):
                for w in range(u + 1, 20):
                    if (u, w) not in used_edges and (w, u) not in used_edges:
                        result[idx] = [u, w]
                        used_edges.add((u, w))
                        found = True
                        break
                if found:
                    break
            if not found:
                result[idx] = None
            continue
        
        # For k >= 3, use backtracking to embed the tree
        tree_n = k
        assignment = [None] * tree_n
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == tree_n:
                return True
            
            tree_v = bfs_order[pos]
            # Try each host vertex
            for host_v in range(20):
                if host_v in assignment:
                    continue
                
                # Check if all edges from already-assigned neighbors are valid
                valid = True
                for nb in tree_adj[tree_v]:
                    if assignment[nb] is not None:
                        edge = (assignment[nb], host_v)
                        if edge in used_edges:
                            valid = False
                            break
                        if (host_v, assignment[nb]) in used_edges:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Tentatively assign
                assignment[tree_v] = host_v
                # Mark edges as temporarily used
                temp_edges = set()
                for nb in tree_adj[tree_v]:
                    if assignment[nb] is not None:
                        e = (assignment[nb], host_v)
                        temp_edges.add(e)
                        temp_edges.add((host_v, assignment[nb]))
                
                if backtrack(pos + 1):
                    # Commit the edges
                    for e in temp_edges:
                        used_edges.add(e)
                    return True
                
                # Undo
                assignment[tree_v] = None
        
        if backtrack(0):
            result[idx] = assignment[:]
        else:
            result[idx] = None
    
    return result
# EVOLVE-BLOCK-END
