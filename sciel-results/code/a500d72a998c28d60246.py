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
    deadline = start_time + 140  # Safety margin before 150s limit
    
    trees = instance_trees(20, 611)
    n = 20
    
    # Total edges in K20
    total_edges = n * (n - 1) // 2  # 190
    
    # Sum of edges in all trees
    tree_edge_counts = [len(t) for t in trees]
    
    # We need to pack all trees edge-disjointly
    # Process largest trees first for better constraint propagation
    # Order: tree indices sorted by size descending
    order = sorted(range(n), key=lambda i: -len(trees[i]))
    
    # used_edges: set of (u, v) with u < v that are already used
    used_edges = set()
    # embedding[i] = list of n host vertices for tree i, or None if not embedded
    embeddings = [None] * n
    
    # For symmetry breaking, we can fix vertex 0 of each tree to a specific host vertex
    # But trees have different structures, so let's just try all possibilities with pruning
    
    def edge_key(a, b):
        return (min(a, b), max(a, b))
    
    def find_embedding(tree_idx, depth, assignment, used_edges_set):
        """
        Find an embedding for tree tree_idx given already-used edges.
        assignment: dict mapping tree-vertex -> host-vertex (partial)
        Returns list of host vertices (length k) or None.
        """
        tree = trees[tree_idx]
        k = len(tree) + 1 if tree else 1
        
        if k == 1:
            # Single vertex, no edges - any unused vertex works, but we need distinct vertices
            # Actually for k=1, we just need one vertex, no edge constraint
            # But we need it to be distinct from other vertices in this tree (trivially true)
            # And no edge conflicts since no edges
            # We can assign any vertex from 0..19
            # But we need to ensure it's not conflicting... actually for single vertex, 
            # the only constraint is that it's a valid host vertex
            # Since no edges, any vertex works. Let's pick the smallest available.
            for v in range(n):
                return [v]
            return None
        
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for u, v in tree:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find a root - pick vertex 0
        root = 0
        
        # Do DFS to assign host vertices
        # assignment is a list of length k, -1 means unassigned
        if assignment is None:
            assignment = [-1] * k
        else:
            # Copy
            assignment = assignment[:]
        
        # For symmetry breaking, fix root to a specific vertex
        # Try each possible host vertex for root
        # But to reduce search, we can try a subset first
        
        def dfs_tree(node, parent_host):
            """Assign host vertex to tree node, given parent's host vertex."""
            if assignment[node] != -1:
                return True
            
            # Try each host vertex for this node
            for h in range(n):
                if h == parent_host and node != root:
                    continue  # Can't map to same host as parent (would create self-loop)
                
                # Check edge conflicts with already assigned neighbors
                conflict = False
                for nb in adj[node]:
                    if assignment[nb] != -1:
                        e = edge_key(h, assignment[nb])
                        if e in used_edges_set:
                            conflict = True
                            break
                    # Also check if this would create a duplicate edge within this tree
                    # (shouldn't happen in a tree, but just in case)
                
                if conflict:
                    continue
                
                # Also check: if nb is assigned and the edge (h, assignment[nb]) is already used
                # We already checked above
                
                # Assign
                assignment[node] = h
                
                # Recurse on children
                success = True
                for nb in adj[node]:
                    if nb == parent_host:
                        continue
                    if assignment[nb] == -1:
                        if not dfs_tree(nb, h):
                            success = False
                            break
                
                if success:
                    return True
                else:
                    assignment[node] = -1
            
            return False
        
        # Try different host vertices for root
        # Use symmetry breaking: try vertices in a specific order
        # To reduce search, we can try all 20 vertices for root
        for root_host in range(n):
            assignment[root] = root_host
            if dfs_tree(root, -1):
                return assignment[:]
            assignment[root] = -1
        
        return None
    
    def backtrack(idx_in_order):
        """Try to embed trees in the given order."""
        if time.time() > deadline:
            return False
        
        if idx_in_order == n:
            return True
        
        tree_idx = order[idx_in_order]
        k = len(trees[tree_idx]) + 1 if trees[tree_idx] else 1
        
        # Find an embedding for this tree
        emb = find_embedding(tree_idx, idx_in_order, None, used_edges)
        
        if emb is None:
            return False
        
        # Check: all vertices in emb must be distinct
        if len(set(emb)) != len(emb):
            return False
        
        # Check all edges of this tree are not in used_edges
        for u, v in trees[tree_idx]:
            e = edge_key(emb[u], emb[v])
            if e in used_edges:
                return False
        
        # Commit
        embeddings[tree_idx] = emb
        for u, v in trees[tree_idx]:
            used_edges.add(edge_key(emb[u], emb[v]))
        
        # Recurse
        if backtrack(idx_in_order + 1):
            return True
        
        # Undo
        for u, v in trees[tree_idx]:
            used_edges.remove(edge_key(emb[u], emb[v]))
        embeddings[tree_idx] = None
        
        return False
    
    # Try the backtracking
    success = backtrack(0)
    
    if success:
        result = []
        for i in range(n):
            if trees[i] is None or len(trees[i]) == 0:
                # T1: single vertex
                result.append([0])  # Any single vertex
            else:
                result.append(embeddings[i])
        return result
    
    # If backtracking failed, try a simpler approach or return best effort
    # As a fallback, just embed T20 with identity
    result = [None] * n
    result[19] = list(range(n))
    return result
