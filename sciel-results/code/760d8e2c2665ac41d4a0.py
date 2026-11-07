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
    deadline = start_time + 145  # safety margin before 150s
    
    trees = instance_trees(20, 611)
    
    # Process trees from largest to smallest for better packing
    # Tree index k-1 has k vertices, k-1 edges
    # Order: try larger trees first
    order = list(range(19, 0, -1))  # indices 19 down to 1 (skip tree 0 which is trivial)
    
    used_edges = set()
    result = [None] * 20
    result[0] = None  # T1 is trivial (single vertex, no edges)
    
    # For each tree in order, try to find an embedding
    # An embedding maps tree vertices 0..k-1 to distinct host vertices 0..19
    # such that all tree edges map to unused edges in K20
    
    def try_embed(tree_idx, used_edges, deadline):
        """Try to find an embedding for tree tree_idx."""
        if time.time() > deadline:
            return None
        
        tree = trees[tree_idx]
        k = tree_idx + 1  # number of vertices in this tree
        
        if k == 1:
            return [0]  # single vertex, any host vertex works
        if k == 2:
            # Need to find an unused edge
            for u in range(20):
                for v in range(u + 1, 20):
                    if (u, v) not in used_edges:
                        return [u, v]
            return None
        
        # For larger trees, use greedy vertex assignment with backtracking
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for u, v in tree:
            adj[u].append(v)
            adj[v].append(u)
            
        # Find a root (vertex with max degree for better branching)
        root = max(range(k), key=lambda v: len(adj[v]))
        
        # BFS/DFS order from root for vertex assignment
        # We'll assign vertices in BFS order
        from collections import deque
        order_bfs = [root]
        visited = {root}
        dq = deque([root])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    order_bfs.append(v)
                    dq.append(v)
        
        # Now greedily assign host vertices in BFS order
        # For each tree vertex, try all available host vertices
        # such that all edges to already-assigned neighbors are valid and unused
        
        assigned = {}  # tree vertex -> host vertex
        assigned[root] = None  # placeholder, we'll try all 20 options
        
        def backtrack(pos, assigned):
            if time.time() > deadline:
                return None
            if pos == k:
                # Verify all edges are used
                for u, v in tree:
                    hu, hv = assigned[u], assigned[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge in used_edges:
                        return None
                return dict(assigned)
            
            tv = order_bfs[pos]
            # Get constraints from already-assigned neighbors
            constraints = set()
            for nb in adj[tv]:
                if nb in assigned:
                    constraints.add(assigned[nb])
            
            # Try all host vertices not used and not in constraints (for edge validity)
            used_hosts = set(assigned.values())
            for h in range(20):
                if h in used_hosts:
                    continue
                # Check that edges to already-assigned neighbors don't use used_edges
                valid = True
                for nb in adj[tv]:
                    if nb in assigned:
                        edge = (min(h, assigned[nb]), max(h, assigned[nb]))
                        if edge in used_edges:
                            valid = False
                            break
                if not valid:
                    continue
                
                assigned[tv] = h
                result_bt = backtrack(pos + 1, assigned)
                if result_bt is not None:
                    return result_bt
                del assigned[tv]
            
            return None
        
        # Try each possible root host vertex
        for root_host in range(20):
            if time.time() > deadline:
                return None
            assigned = {root: root_host}
            result_bt = backtrack(1, assigned)
            if result_bt is not None:
                # Convert to list format: position i is image of vertex i
                return [result_bt[i] for i in range(k)]
        
        return None
    
    # Process trees
    for tree_idx in order:
        if time.time() > deadline:
            break
        embedding = try_embed(tree_idx, used_edges, deadline)
        if embedding is not None:
            result[tree_idx] = embedding
            # Mark edges as used
            tree = trees[tree_idx]
            for u, v in tree:
                hu, hv = embedding[u], embedding[v]
                edge = (min(hu, hv), max(hu, hv))
                used_edges.add(edge)
        else:
            # Can't embed this tree, leave as None
            pass
    
    # T1 (index 0) is trivial
    result[0] = None
    
    # Verify types
    for i in range(20):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]
    
    return result
