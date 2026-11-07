"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    deadline = start_time + 130.0  # safety margin before 150s kill
    N = 20
    trees = instance_trees(N, 611)
    
    # Pre-compute edge sets for each tree
    tree_edges = []
    for t in trees:
        tree_edges.append(set((min(u,v), max(u,v)) for u,v in t))
    
    # We need to embed trees T1..T20 into K20 with edge-disjoint embeddings.
    # T1 is a single vertex (no edges), T2 is a single edge.
    # For tree k, we need to assign its k vertices to distinct host vertices in 0..19.
    # The image of vertex i of T_k is at position i in the result list.
    
    # Strategy: Use randomized DFS with backtracking, trying to embed trees in a 
    # good order. Start with larger trees (more constrained) and work down, or 
    # try multiple orderings.
    
    # Actually, let's think about this more carefully. We have K20 with 190 edges.
    # The total number of edges in all trees is sum(k-1 for k in 1..20) = 0+1+2+...+19 = 190.
    # So we need a perfect packing - every edge of K20 must be used exactly once.
    
    # This is a very tight constraint. Let's use a backtracking approach.
    
    # Represent the state as:
    # - assigned: for each tree, a list of length k with the host vertex for each tree vertex, or -1 if unassigned
    # - used_edges: a set of (u,v) with u<v that are already used
    
    # To make this efficient, we'll embed trees one at a time, and within each tree,
    # we'll assign vertices using DFS with backtracking.
    
    # Order of embedding: try largest trees first (most constrained)
    # Trees are indexed 0..19 corresponding to T1..T20
    # T_k has k vertices, so tree index i has (i+1) vertices
    
    # Let's try embedding in order of decreasing size: T20, T19, ..., T1
    # But T1 is just a vertex, T2 is an edge - these are easy.
    
    # Actually, let's try a different approach. Since we need to use all 190 edges,
    # this is equivalent to decomposing K20 into these specific trees.
    
    # Let me implement a backtracking solver that embeds trees one by one.
    
    def solve(order, rng, max_time):
        """Try to embed trees in the given order. Returns list of embeddings or None."""
        used_edges = set()
        embeddings = [None] * N
        
        def backtrack(idx):
            if time.time() > max_time:
                return False
            if idx == N:
                return True
            
            tree_idx = order[idx]
            k = tree_idx + 1  # tree_idx 0 -> T1 (1 vertex), tree_idx 19 -> T20 (20 vertices)
            
            if k == 1:
                # Single vertex, no edges. Any host vertex works.
                # But we need to pick one that doesn't conflict... actually no edges, so any vertex.
                # However, for consistency, let's just pick the first available vertex.
                # Actually, since there are no edges, any vertex is fine. Let's pick 0.
                embeddings[tree_idx] = [0]
                return backtrack(idx + 1)
            
            # For trees with edges, we need to assign k distinct host vertices
            # such that no edge conflicts with used_edges.
            
            # Build adjacency for this tree
            adj = [[] for _ in range(k)]
            for u, v in trees[tree_idx]:
                adj[u].append(v)
                adj[v].append(u)
            
            # We'll use a DFS to assign vertices. Start with vertex 0, try all host vertices.
            assignment = [-1] * k
            tree_used_edges = set()
            
            def dfs_vertex(v):
                if time.time() > max_time:
                    return False
                if v == k:
                    return True
                
                # Try each possible host vertex
                for host in range(N):
                    if host in assignment:
                        continue
                    # Check if assigning v to host creates any edge conflict
                    conflict = False
                    for nb in adj[v]:
                        if assignment[nb] != -1:
                            a, b = min(host, assignment[nb]), max(host, assignment[nb])
                            if (a, b) in used_edges:
                                conflict = True
                                break
                    if conflict:
                        continue
                    
                    assignment[v] = host
                    # Add edges to tree_used_edges temporarily
                    new_edges = set()
                    for nb in adj[v]:
                        if assignment[nb] != -1:
                            a, b = min(host, assignment[nb]), max(host, assignment[nb])
                            new_edges.add((a, b))
                    
                    tree_used_edges.update(new_edges)
                    
                    if dfs_vertex(v + 1):
                        return True
                    
                    tree_used_edges.difference_update(new_edges)
                    assignment[v] = -1
                
                return False
            
            if dfs_vertex(0):
                # Successfully embedded this tree
                embeddings[tree_idx] = list(assignment)
                used_edges.update(tree_used_edges)
                return backtrack(idx + 1)
            
            return False
        
        return backtrack(0)
    
    # Try multiple orderings and random seeds
    best_result = None
    
    # First try descending order (largest trees first)
    order_desc = list(range(N-1, -1, -1))  # T20, T19, ..., T1
    order_asc = list(range(N))  # T1, T2, ..., T20
    
    rng = Random(42)
    
    # Try descending order first
    result = solve(order_desc, rng, deadline - 5.0)
    if result is not None:
        best_result = result
    
    if best_result is None:
        # Try ascending order
        result = solve(order_asc, rng, deadline - 5.0)
        if result is not None:
            best_result = result
    
    if best_result is None:
        # Try random orderings
        for trial in range(20):
            if time.time() > deadline - 5.0:
                break
            order = list(range(N))
            rng.shuffle(order)
            result = solve(order, rng, deadline - 5.0)
            if result is not None:
                best_result = result
                break
    
    if best_result is None:
        # Fallback: just embed T20 with identity
        return [None] * 19 + [list(range(N))]
    
    # Convert to proper format
    final = []
    for i in range(N):
        if best_result[i] is None:
            final.append(None)
        else:
            final.append([int(x) for x in best_result[i]])
    
    return final

# EVOLVE-BLOCK-END
