"""Packing trees T1..T20 into K20 using edge-coloring with backtracking."""
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


def _build_matchings(n=20):
    """Build 19 perfect matchings (1-factors) of K20.
    Returns list of 19 sets, each containing 10 edges (frozensets).
    Also returns a dict: edge -> set of matching indices containing it.
    """
    # Standard 1-factorization of K_{2m} using round-robin
    # Vertices 0..19, with vertex 19 as "infinity"
    matchings = []
    for r in range(19):
        matching = set()
        # Vertex 19 pairs with vertex r
        matching.add(frozenset({19, r}))
        # Remaining 18 vertices: 0..18 except r
        # They are paired in a specific pattern
        for i in range(9):
            a = (r + 1 + i) % 19
            b = (r - 1 - i) % 19
            matching.add(frozenset({a, b}))
        matchings.append(matching)
    
    # Build edge -> matching indices map
    edge_to_matchings = {}
    for mi, matching in enumerate(matchings):
        for edge in matching:
            if edge not in edge_to_matchings:
                edge_to_matchings[edge] = set()
            edge_to_matchings[edge].add(mi)
    
    return matchings, edge_to_matchings


def _embed_tree_color_constrained(tree_edges, n, allowed_matchings, edge_to_matchings,
                                   used_edges, deadline, rng):
    """
    Embed tree_edges into K_n such that each edge uses only allowed matchings.
    Returns mapping dict or None.
    """
    if not tree_edges:
        return {}
    
    tree_verts = set()
    for u, v in tree_edges:
        tree_verts.add(u)
        tree_verts.add(v)
    tree_verts = sorted(tree_verts)
    k = len(tree_verts)
    
    # Build adjacency
    adj = {v: [] for v in tree_verts}
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find leaves for root selection
    leaves = [v for v in tree_verts if len(adj[v]) == 1]
    if not leaves:
        leaves = tree_verts
    
    # Try a few root choices
    roots_to_try = leaves[:3] if len(leaves) >= 3 else leaves
    if not roots_to_try:
        roots_to_try = [tree_verts[0]]
    
    for root in roots_to_try:
        # BFS ordering
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
        
        # Backtracking search
        def backtrack(idx):
            if idx == len(order):
                return True
            if deadline and time.time() > deadline:
                return False
            
            v = order[idx]
            p = parent[v]
            host_p = mapping[p]
            
            # Find candidate host vertices h such that:
            # 1. h not already used
            # 2. edge (host_p, h) not in used_edges
            # 3. edge (host_p, h) is in one of the allowed matchings
            candidates = []
            for h in range(n):
                if h in host_used:
                    continue
                e = frozenset({host_p, h})
                if e in used_edges:
                    continue
                # Check if this edge is in an allowed matching
                mset = edge_to_matchings.get(e, set())
                if mset & allowed_matchings:
                    candidates.append(h)
            
            if not candidates:
                return False
            
            rng.shuffle(candidates)
            
            for h in candidates:
                mapping[v] = h
                host_used.add(h)
                if backtrack(idx + 1):
                    return True
                del mapping[v]
                host_used.remove(h)
            
            return False
        
        # Try multiple random attempts
        for attempt in range(50):
            if deadline and time.time() > deadline:
                return None
            
            mapping = {}
            host_used = set()
            
            # Assign root
            available = [h for h in range(n) if h not in host_used]
            if not available:
                return None
            rng.shuffle(available)
            host_root = available[0]
            mapping[root] = host_root
            host_used.add(host_root)
            
            if backtrack(1):
                return mapping
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    trees = instance_trees(20, 611)
    n = 20
    used_edges = set()
    result = [None] * 20
    
    # Build 1-factorization
    matchings, edge_to_matchings = _build_matchings(n)
    num_matchings = len(matchings)  # 19
    
    # Assign matchings to trees
    # Trees with more edges need more matching capacity
    # Tree k has k-1 edges, needs k-1 distinct edges from assigned matchings
    # Each matching has 10 edges, so a tree with e edges needs at least ceil(e/10) matchings
    # But we also need the specific edges to be available (not used by other trees)
    
    # Strategy: assign matchings greedily by tree size
    # Sort trees by edge count descending
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    # Available matchings (indices 0..18)
    available_matchings = set(range(num_matchings))
    
    # Assign matchings to each tree
    tree_matchings = {}
    for idx in tree_indices:
        k = idx + 1
        num_edges = k - 1
        if num_edges == 0:
            tree_matchings[idx] = set()
            continue
        
        # Need enough matchings to provide num_edges distinct edges
        # Each matching provides at most 10 edges, but some may be used
        # Be generous: assign enough matchings
        # Minimum: ceil(num_edges / 10) matchings, but let's be more generous
        # For a tree with e edges, we need e distinct edges from the union of assigned matchings
        # Union of m matchings has at most 10*m edges (but they overlap in vertices)
        # Actually each matching is a perfect matching, so union of m matchings has m*10 edges
        # but they share vertices. The key constraint is that we need e edges that are
        # all in different matchings (since each edge belongs to exactly one matching).
        # So we need at least e matchings? No, that's too many.
        # Actually, each edge belongs to exactly one matching. So for a tree with e edges,
        # we need those e edges to be in the assigned matchings. Since each edge is in exactly
        # one matching, we need the assigned matchings to collectively contain all e edges.
        # But we don't know which edges the tree will use until we embed it.
        # 
        # Better approach: don't pre-assign specific matchings. Instead, for each tree,
        # allow ALL matchings but track which edges are used. The constraint is just
        # edge-disjointness. The matching structure helps with search ordering.
        
        # Revised strategy: Use all matchings for all trees, but use the matching
        # structure to guide the search. For each tree, when choosing a host vertex,
        # prefer edges that are in matchings with more remaining capacity.
        
        tree_matchings[idx] = set(range(num_matchings))  # All matchings available
    
    # Actually, let's use a simpler but effective approach:
    # For each tree, use all matchings. The backtracking with edge-disjointness
    # is the main constraint. The matching structure can help with ordering.
    
    rng = Random(42)
    
    # Process trees in order of decreasing size
    for idx in tree_indices:
        if time.time() > deadline - 3.0:
            break
        
        k = idx + 1
        tree_edges = trees[idx]
        
        if not tree_edges:
            result[idx] = [0]
            continue
        
        # Try to embed with all matchings allowed
        mapping = _embed_tree_color_constrained(
            tree_edges, n, set(range(num_matchings)), edge_to_matchings,
            used_edges, deadline, rng
        )
        
        if mapping is not None:
            result_list = [int(mapping[v]) for v in range(k)]
            result[idx] = result_list
            
            for u, v in tree_edges:
                hu, hv = mapping[u], mapping[v]
                e = frozenset({hu, hv})
                used_edges.add(e)
    
    # Ensure types
    for i in range(20):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]
    
    return result
