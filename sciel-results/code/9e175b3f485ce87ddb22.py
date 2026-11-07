from random import Random
import time
import sys

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    trees = instance_trees(20, 611)
    
    # Trees indexed 0..19, tree k has k vertices (k+1 actually? No)
    # Tree index 0: T1, 1 vertex, 0 edges
    # Tree index 1: T2, 2 vertices, 1 edge
    # Tree index k: T_{k+1}, k+1 vertices, k edges
    
    # Order: process largest first
    # Tree indices: 19 (T20, 20 verts, 19 edges), 18 (T19, 19 verts, 18 edges), ..., 0 (T1, 1 vert, 0 edges)
    
    n = 20
    # used_edges: set of (u,v) with u<v
    used_edges = set()
    embeddings = [None] * n  # embeddings[i] for tree i
    
    # Process trees in order from largest to smallest
    order = list(range(n - 1, -1, -1))  # 19, 18, ..., 0
    
    # For each tree, find an embedding
    # Tree i has (i+1) vertices, edges = trees[i]
    
    def get_tree_vertices(i):
        """Return list of vertices in tree i"""
        if i == 0:
            return [0]
        return list(range(i + 1))
    
    def get_tree_edges(i):
        return trees[i]
    
    # Backtracking for a single tree embedding
    def embed_tree(tree_idx, used_edges, deadline):
        """Find an embedding of tree tree_idx that is edge-disjoint with used_edges.
        Returns list of length (tree_idx+1) where position i is host vertex for tree vertex i.
        Or None if not found within deadline.
        """
        k = tree_idx + 1  # number of vertices
        edges = trees[tree_idx]
        
        if k == 1:
            return [0]  # single vertex, any host works
        
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Order vertices for assignment: use BFS order from vertex 0
        # This helps with constraint propagation
        visited = [False] * k
        order = []
        queue = [0]
        visited[0] = True
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        
        # assignment: tree vertex -> host vertex
        assignment = [-1] * k
        used_hosts = [False] * n
        
        def backtrack(pos, used_edges_local):
            if time.time() > deadline:
                return None
            
            if pos == len(order):
                return assignment[:]
            
            tree_v = order[pos]
            
            # Try each unused host vertex
            for host in range(n):
                if used_hosts[host]:
                    continue
                
                # Check if edges from tree_v to already-assigned tree vertices conflict
                conflict = False
                for neighbor in adj[tree_v]:
                    if assignment[neighbor] != -1:
                        edge_key = (min(host, assignment[neighbor]), max(host, assignment[neighbor]))
                        if edge_key in used_edges_local:
                            conflict = True
                            break
                
                if conflict:
                    continue
                
                # Forward checking: for unassigned neighbors, check if they still have options
                # (optional, can add later)
                
                assignment[tree_v] = host
                used_hosts[host] = True
                
                result = backtrack(pos + 1, used_edges_local)
                if result is not None:
                    return result
                
                assignment[tree_v] = -1
                used_hosts[host] = False
            
            return None
        
        result = backtrack(0, used_edges)
        return result
    
    # Process trees
    for tree_idx in order:
        if time.time() > deadline:
            break
        
        k = tree_idx + 1
        if k == 1:
            embeddings[tree_idx] = [0]
            continue
        
        # Try to embed
        emb = embed_tree(tree_idx, used_edges, deadline)
        if emb is None:
            # Try with randomization/restart? Or just skip
            # For now, skip this tree
            continue
        
        embeddings[tree_idx] = emb
        
        # Add edges to used_edges
        for (u, v) in trees[tree_idx]:
            host_u = emb[u]
            host_v = emb[v]
            edge_key = (min(host_u, host_v), max(host_u, host_v))
            used_edges.add(edge_key)
    
    # Count total edges
    total_edges = 0
    for i in range(n):
        if embeddings[i] is not None:
            total_edges += len(trees[i])
    
    # Verify type correctness
    result = []
    for i in range(n):
        if embeddings[i] is None:
            result.append(None)
        else:
            # Convert to list of Python ints
            result.append([int(x) for x in embeddings[i]])
    
    return result
