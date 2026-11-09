"""Packing trees T1..T20 into K20 using greedy edge-disjoint embedding."""
import time
import random
from random import Random
from collections import deque


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


def _embed_tree(tree_edges, n=20, used_edges=None, rng=None, deadline=None):
    """
    Try to embed tree_edges (list of (u,v) pairs) into K_n.
    Returns a mapping dict from tree vertex -> host vertex,
    or None if fails.
    """
    if not tree_edges:
        return {}
    
    # Collect all tree vertices
    tree_verts = set()
    for u, v in tree_edges:
        tree_verts.add(u)
        tree_verts.add(v)
    tree_verts = sorted(tree_verts)
    k = len(tree_verts)
    
    # Build adjacency for the tree
    adj = {v: [] for v in tree_verts}
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Compute degree of each tree vertex
    tree_degree = {v: len(adj[v]) for v in tree_verts}
    
    # Find roots: prefer high-degree vertices, then leaves
    # Try different root choices across attempts
    roots = sorted(tree_verts, key=lambda v: (-tree_degree[v], v))
    # Also include some leaf roots for diversity
    leaves = [v for v in tree_verts if tree_degree[v] == 1]
    if leaves:
        roots = roots[:3] + leaves[:3]
    else:
        roots = roots[:5]
    
    # Try multiple random attempts with backtracking
    max_attempts = 800
    for attempt in range(max_attempts):
        if deadline and time.time() > deadline:
            return None
        
        # Choose root based on attempt number for diversity
        root_idx = attempt % len(roots)
        root = roots[root_idx]
        
        # BFS to get an ordering from this root
        order = [root]
        parent = {root: -1}
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for w in adj[u]:
                if w not in parent:
                    parent[w] = u
                    queue.append(w)
                    order.append(w)
        
        # Reorder: place high-degree non-root vertices earlier for better pruning
        # Keep parent before child, but sort siblings by degree descending
        # We'll use a modified order: process by (depth, -degree)
        depth = {root: 0}
        for v in order[1:]:
            depth[v] = depth[parent[v]] + 1
        
        # Create ordering: sort by (depth, -tree_degree[v], v)
        # This ensures parents come before children (same or lower depth)
        # and within same depth, higher degree first
        sorted_order = sorted(order, key=lambda v: (depth[v], -tree_degree[v], v))
        
        # Verify parent comes before child in sorted_order
        pos_in_order = {v: i for i, v in enumerate(sorted_order)}
        valid = all(pos_in_order[parent[v]] < pos_in_order[v] for v in order[1:])
        if not valid:
            # Fall back to BFS order if sorting broke parent-before-child
            sorted_order = order
        
        mapping = {}
        host_used = set()
        
        # Assign root
        available = [h for h in range(n)]
        rng.shuffle(available)
        host_root = available[0]
        mapping[root] = host_root
        host_used.add(host_root)
        
        # Assign remaining vertices in order with backtracking
        def backtrack(idx):
            if idx == len(sorted_order):
                return True
            if deadline and time.time() > deadline:
                return False
            
            v = sorted_order[idx]
            p = parent[v]
            if p == -1:
                # This is the root, already assigned
                return backtrack(idx + 1)
            
            host_p = mapping[p]
            
            # Need a host h such that (host_p, h) not in used_edges
            # and h not in host_used
            candidates = [h for h in range(n) if h not in host_used]
            
            # Sort candidates: prefer hosts with more remaining free edges
            # (i.e., fewer used edges incident to them)
            # This is a heuristic to leave more options for later vertices
            def candidate_score(h):
                # Count how many used edges are incident to h
                count = 0
                for e in used_edges:
                    if h in e:
                        count += 1
                return count
            
            # Shuffle for randomness, but bias toward lower score (more free)
            rng.shuffle(candidates)
            # Partial sort: try lower-score candidates first with some randomness
            candidates.sort(key=lambda h: (candidate_score(h), rng.random()))
            
            for h in candidates:
                e = (min(host_p, h), max(host_p, h))
                if e not in used_edges:
                    mapping[v] = h
                    host_used.add(h)
                    if backtrack(idx + 1):
                        return True
                    del mapping[v]
                    host_used.remove(h)
            
            return False
        
        if backtrack(0):
            return mapping
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    trees = instance_trees(20, 611)
    n = 20
    used_edges = set()
    result = [None] * 20
    
    # Embed trees in order of decreasing size (more edges first)
    # But we must return in original order
    order = list(range(20))
    # Sort by number of edges descending
    order.sort(key=lambda i: len(trees[i]), reverse=True)
    
    rng = Random(42)
    
    for idx in order:
        if time.time() > deadline - 5.0:
            break
        k = idx + 1
        tree_edges = trees[idx]
        
        if not tree_edges:
            # Tree with 1 vertex (k=1)
            result[idx] = [0]
            continue
        
        # Try to embed
        mapping = _embed_tree(tree_edges, n, used_edges, rng, deadline)
        if mapping is not None:
            # Build the result list: position i is image of vertex i
            # Tree vertices are 0..k-1
            result_list = []
            for v in range(k):
                result_list.append(int(mapping[v]))
            result[idx] = result_list
            
            # Add edges to used_edges
            for u, v in tree_edges:
                hu, hv = mapping[u], mapping[v]
                e = (min(hu, hv), max(hu, hv))
                used_edges.add(e)
        
        # If we failed on a large tree, try smaller ones first
        # and come back to it later with more used edges info
        # (This is handled by the order already)
    
    # Ensure types are correct
    for i in range(20):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]
    
    # Second pass: try to embed any failed trees with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        for idx in order:
            if result[idx] is None and trees[idx]:
                if time.time() > deadline - 3.0:
                    break
                tree_edges = trees[idx]
                mapping = _embed_tree(tree_edges, n, used_edges, rng, deadline)
                if mapping is not None:
                    k = idx + 1
                    result_list = []
                    for v in range(k):
                        result_list.append(int(mapping[v]))
                    result[idx] = result_list
                    for u, v in tree_edges:
                        hu, hv = mapping[u], mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        used_edges.add(e)
    
    return result
# EVOLVE-BLOCK-END
