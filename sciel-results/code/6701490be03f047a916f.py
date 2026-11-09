import time
from random import Random

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
    deadline = start_time + 130  # safety margin
    
    trees = instance_trees(20, 611)
    n = 20
    
    # trees[k-1] is the tree with k vertices (k-1 edges)
    # We need to embed each tree into K20, edge-disjoint
    
    # Total edges: sum(k-1 for k=1..20) = 0+1+2+...+19 = 190
    # K20 has exactly 190 edges, so we need a perfect decomposition
    
    # Strategy: embed trees in order of decreasing number of vertices
    # T20 first (20 vertices, 19 edges), then T19, etc.
    
    # For each tree, find all possible embeddings (injective maps from tree vertices to 0..19)
    # such that no edge collides with already-used edges
    
    # Since T20 is a spanning tree, its embedding is a permutation of 0..19
    # We can fix vertex 0 of T20 to host vertex 0 (symmetry breaking)
    
    # Build adjacency for each tree
    tree_adj = []
    for k in range(1, n + 1):
        edges = trees[k-1]
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    # Order: embed from largest to smallest
    # T20 (idx 19), T19 (idx 18), ..., T2 (idx 1), T1 (idx 0)
    order = list(range(n - 1, -1, -1))  # [19, 18, ..., 0]
    
    used_edges = set()
    embeddings = [None] * n  # embeddings[k-1] = list of host vertices for tree k
    
    # For T1 (k=1), any single vertex works
    # For T2 (k=2), any edge not used
    
    def get_tree_edges(k, mapping):
        """Get the set of host edges for tree k under mapping."""
        edges = set()
        for u, v in trees[k-1]:
            a, b = mapping[u], mapping[v]
            if a > b:
                a, b = b, a
            edges.add((a, b))
        return edges
    
    def try_embed(k, mapping, depth):
        """Try to embed tree k with given partial mapping."""
        if time.time() > deadline:
            return False
        
        if k == 1:
            # Single vertex, any unused... actually no edge constraint
            # Just pick any vertex
            embeddings[0] = [0]  # will be adjusted
            return True
        
        # Check if mapping is complete
        if len(mapping) == k:
            # Check all edges are valid (not in used_edges)
            for u, v in trees[k-1]:
                a, b = mapping[u], mapping[v]
                if a > b:
                    a, b = b, a
                if (a, b) in used_edges:
                    return False
            # All good, record
            embeddings[k-1] = [mapping[i] for i in range(k)]
            # Add to used_edges
            for u, v in trees[k-1]:
                a, b = mapping[u], mapping[v]
                if a > b:
                    a, b = b, a
                used_edges.add((a, b))
            return True
        
        # Pick next tree vertex to assign
        # Use the vertex with highest degree in the tree among unassigned
        unassigned = [v for v in range(k) if v not in mapping]
        if not unassigned:
            return False
        
        # Sort by degree descending for better pruning
        unassigned.sort(key=lambda v: len(tree_adj[k-1][v]), reverse=True)
        next_v = unassigned[0]
        
        # Try all possible host vertices
        for host in range(n):
            if host in mapping.values():
                continue
            
            # Check edges to already-assigned neighbors
            valid = True
            for nb in tree_adj[k-1][next_v]:
                if nb in mapping:
                    a, b = host, mapping[nb]
                    if a > b:
                        a, b = b, a
                    if (a, b) in used_edges:
                        valid = False
                        break
            if not valid:
                continue
            
            mapping[next_v] = host
            if try_embed(k, mapping, depth + 1):
                return True
            del mapping[next_v]
        
        return False
    
    # Main backtracking over trees in order
    def backtrack_tree(idx):
        if time.time() > deadline:
            return False
        
        if idx == len(order):
            return True
        
        k = order[idx]  # tree index (0-based, so tree k has k+1 vertices)
        num_vertices = k + 1
        
        if num_vertices == 1:
            # T1: single vertex, pick any
            # But we need to make sure it doesn't conflict... no edges so always ok
            embeddings[0] = [0]
            return backtrack_tree(idx + 1)
        
        mapping = {}
        if not try_embed(k, mapping, 0):
            return False
        
        return backtrack_tree(idx + 1)
    
    # Special handling: T20 is a spanning tree, so its embedding is a bijection
    # Fix vertex 0 of T20 to host 0 for symmetry breaking
    # Actually, let's just let the backtracking handle it
    
    if not backtrack_tree(0):
        # Fallback: just embed T20 as identity
        return [None] * 19 + [list(range(20))]
    
    # Build result: entry k-1 is None or list of k host vertices
    result = []
    for k in range(1, n + 1):
        if embeddings[k-1] is not None:
            # Convert to list of Python ints
            result.append([int(x) for x in embeddings[k-1]])
        else:
            result.append(None)
    
    return result
