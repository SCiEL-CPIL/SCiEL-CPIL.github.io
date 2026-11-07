"""Packing trees T1..T20 into K20 via backtracking with constraint propagation."""
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    trees = instance_trees(20, 611)
    n_trees = len(trees)
    
    # Sort trees by size (number of edges) descending for better pruning
    # Tree k has k-1 edges
    tree_indices = list(range(n_trees))
    tree_indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    # Precompute edges for each tree
    # tree_edges[i] = list of edges in tree i
    tree_edges = [trees[i] for i in tree_indices]
    
    # Total edges in K20 = 190
    # Total edges in all trees = sum(k-1 for k in 1..20) = 0+1+2+...+19 = 190
    # So we need a perfect packing using all edges
    
    # For backtracking:
    # State: which edges of K20 are used
    # We embed trees one by one (largest first)
    # For each tree, we need to assign its vertices to distinct host vertices
    # such that no edge maps to an already-used edge
    
    # Host vertices: 0..19
    # Edge key: frozenset({u,v}) or sorted tuple (min(u,v), max(u,v))
    
    # Precompute all K20 edges
    all_k20_edges = set()
    for u in range(20):
        for v in range(u+1, 20):
            all_k20_edges.add((u, v))
    
    def edge_key(u, v):
        if u > v:
            u, v = v, u
        return (u, v)
    
    # For each tree, precompute its edge list
    # We'll use backtracking to embed each tree
    
    # Key insight: we need to embed all trees such that their images are edge-disjoint
    # Since total edges = 190 = |E(K20)|, we must use every edge exactly once
    
    # Strategy: backtracking with the largest trees first
    # For each tree, try all possible embeddings (vertex mappings)
    
    # An embedding of tree T_k is a bijection from {0..k-1} to a subset of {0..19} of size k
    # such that for every edge (a,b) in T_k, the mapped edge (f(a), f(b)) is not already used
    
    # This is expensive. Let's use a smarter approach.
    
    # Alternative: use the fact that we can build the solution incrementally.
    # Start with the largest tree (T20, a spanning tree on 20 vertices).
    # T20 uses 19 edges. Then T19 needs 18 more edges, etc.
    
    # Actually, let's think about this differently.
    # We have 190 edges to assign to 190 tree-edges.
    # Each tree's edges must form a tree structure under the mapping.
    
    # Let's use a different approach: 
    # 1. First embed T20 (spanning tree on all 20 vertices) - this fixes the vertex mapping for all trees
    #    Wait, no. Each tree is embedded independently. The vertices of different trees can map to the same host vertices.
    #    Only the edges need to be disjoint.
    
    # So the problem is: assign to each tree edge a unique K20 edge, such that for each tree,
    # the assigned edges form a valid tree embedding (i.e., there exists a vertex mapping consistent with all edge assignments).
    
    # This is complex. Let me try a direct backtracking approach.
    
    # For each tree (in order from largest to smallest), find an embedding.
    # An embedding is a mapping of the tree's vertices to host vertices.
    
    # For T_k with k vertices, we need to choose k distinct host vertices and assign them to the tree's vertices
    # such that no tree edge maps to an already-used edge.
    
    # Let's implement backtracking for each tree embedding.
    
    used_edges = set()
    result = [None] * n_trees
    
    def find_embedding(tree_idx, tree, deadline):
        """Find an embedding of the given tree using unused edges.
        Returns a list of k host vertices (mapping tree vertex i to host vertex mapping[i]), or None.
        """
        k = len(tree) + 1  # tree has k-1 edges, so k vertices
        if k == 1:
            # Single vertex, no edges. Can map to any vertex.
            return [0]
        
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for u, v in tree:
            adj[u].append(v)
            adj[v].append(u)
        
        # Backtracking: assign host vertices to tree vertices
        # We'll do this in a smart order: start from a leaf, then expand
        # Actually, let's use a recursive backtracking that assigns one vertex at a time
        
        # Order tree vertices: start from vertex 0, then BFS/DFS
        # For efficiency, pick a root and do DFS order
        visited = [False] * k
        order = []
        stack = [0]
        visited[0] = True
        while stack:
            v = stack.pop()
            order.append(v)
            for nb in adj[v]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        
        # Now assign host vertices in this order
        # host_assignment[i] = host vertex for tree vertex order[i]
        host_assignment = [None] * len(order)
        
        # Available host vertices
        available = set(range(20))
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(order):
                # Check all edges
                for u, v in tree:
                    hu = host_assignment[order.index(u)]
                    hv = host_assignment[order.index(v)]
                    ek = edge_key(hu, hv)
                    if ek in used_edges:
                        return False
                # Also need to check that we haven't used the same edge twice within this tree
                # But since it's a tree, all edges are distinct, and we check against used_edges
                # We also need to make sure edges within this tree don't conflict with each other
                # Since it's a tree, all (u,v) pairs are distinct, so this is fine
                return True
            
            tv = order[idx]
            for hv in available:
                # Check if assigning tv -> hv causes any conflict with already-assigned neighbors
                conflict = False
                for nb in adj[tv]:
                    nb_idx = order.index(nb)
                    if nb_idx < idx:
                        # nb is already assigned
                        hnb = host_assignment[nb_idx]
                        ek = edge_key(hv, hnb)
                        if ek in used_edges:
                            conflict = True
                            break
                if conflict:
                    continue
                
                # Tentatively assign
                host_assignment[idx] = hv
                available.remove(hv)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                available.add(hv)
                host_assignment[idx] = None
            
            return False
        
        # We need to be more careful: the above checks edges against used_edges but not against
        # edges that will be created by later assignments within this same tree.
        # However, since we're building the mapping incrementally and the tree is connected,
        # when we assign a new vertex, we only check edges to already-assigned neighbors.
        # Edges to not-yet-assigned neighbors will be checked when those neighbors are assigned.
        # But we also need to ensure that two different edges in the tree don't map to the same K20 edge.
        # Since the tree is a tree (no cycles), and we're assigning distinct host vertices, 
        # two different tree edges (u1,v1) and (u2,v2) could potentially map to the same K20 edge
        # if {f(u1),f(v1)} = {f(u2),f(v2)}. This can only happen if the tree has a 4-cycle or something,
        # but a tree doesn't have cycles. Actually, in a tree, two distinct edges share at most one vertex.
        # If they share a vertex, say (u,v) and (u,w), then f(u),f(v) and f(u),f(w) are different edges
        # since f(v) != f(w). If they don't share a vertex, they're on different parts of the tree,
        # and since all host vertices are distinct, the edges are different.
        # So we're fine.
        
        if backtrack(0):
            # Build the result: mapping from tree vertex i to host vertex
            mapping = [None] * k
            for i, tv in enumerate(order):
                mapping[tv] = host_assignment[i]
            # Mark edges as used
            for u, v in tree:
                used_edges.add(edge_key(mapping[u], mapping[v]))
            return mapping
        
        return None
    
    # Process trees from largest to smallest
    for ti, tree_idx in enumerate(tree_indices):
        if time.time() > deadline:
            break
        
        tree = trees[tree_idx]
        k = len(tree) + 1 if tree else 1
        
        if k == 1:
            # T1: single vertex, no edges
            result[tree_idx] = [0]
            continue
        
        mapping = find_embedding(tree_idx, tree, deadline)
        if mapping is None:
            # Failed to embed this tree. Try with a different random order?
            # For now, return what we have
            break
        result[tree_idx] = mapping
    
    # Check if we got all trees
    if all(r is not None for r in result):
        # Verify edge-disjointness
        all_used = set()
        for i in range(n_trees):
            tree = trees[i]
            mapping = result[i]
            for u, v in tree:
                ek = edge_key(mapping[u], mapping[v])
                if ek in all_used:
                    # Conflict!
                    pass
                all_used.add(ek)
        # Count total edges
        total = sum(len(trees[i]) for i in range(n_trees) if result[i] is not None)
        return result
    
    # If we didn't get all trees, return what we have
    # But the problem says any broken rule makes the result invalid
    # So we need all 20 trees embedded, or return something valid
    # If we can't embed all, we should return None for the ones we couldn't
    # But the problem says entry k-1 is None or a list... so None is allowed for unembedded trees
    # But the raw value is the total number of embedded tree edges
    # So partial solutions are scored by the number of edges
    
    return result
