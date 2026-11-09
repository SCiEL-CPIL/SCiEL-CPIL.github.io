import time
import random
from itertools import permutations

_DEADLINE = 145.0  # seconds, safety margin before 150s kill
_start_time = time.time()

def _remaining_time():
    return _DEADLINE - (time.time() - _start_time)

def _generate_trees(n=20, seed=611):
    from random import Random
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

def _tree_edges_set(tree):
    """Return set of frozenset edge pairs for a tree (as vertex indices 0..k-1)."""
    return [frozenset(e) for e in tree]

def _find_embedding(tree_edges, k, n, used_edge_set):
    """
    Find a mapping from tree vertices 0..k-1 to host vertices 0..n-1
    such that no mapped edge is in used_edge_set.
    Returns list of length k where result[i] = host vertex for tree vertex i, or None.
    """
    if k == 1:
        # Single vertex, no edges; pick any unused... actually just pick 0
        # But we need distinct vertices. For k=1, any single vertex works.
        # We'll pick the first available host vertex.
        # Actually for k=1 there are no edges so any vertex works.
        # Let's just return [0] as a placeholder; caller handles.
        return list(range(1))  # [0]
    
    if k == 2:
        # One edge (0,1). Find any pair of host vertices not in used_edge_set.
        for a in range(n):
            for b in range(a + 1, n):
                if frozenset((a, b)) not in used_edge_set:
                    return [a, b]
        return None
    
    # For k >= 3, use backtracking
    # Build adjacency for the tree
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try to find an embedding using backtracking
    # We'll assign tree vertices in a fixed order (0, 1, ..., k-1)
    # But we can be smarter: start with vertex 0 mapped to some host vertex,
    # then do DFS.
    
    # To reduce symmetry, fix tree vertex 0 to host vertex 0.
    # Wait, that's too restrictive. Let's try all host vertices for tree vertex 0.
    
    # Actually, let's use a different approach: try random permutations with pruning.
    # Given time constraints, use a randomized backtracking.
    
    # Precompute: for each tree vertex, its neighbors
    # Use backtracking with MRV (minimum remaining values)
    
    # Simpler: iterate over all possible images of vertex 0 (n choices),
    # then for each, do DFS assigning the rest.
    
    best = None
    for host0 in range(n):
        mapping = [-1] * k
        mapping[0] = host0
        used_hosts = {host0}
        
        # DFS to assign remaining tree vertices
        def dfs(idx):
            if _remaining_time() < 1.0:
                return None
            if idx == k:
                return mapping[:]
            
            # Try to assign tree vertex idx
            # Constraints: 
            # 1. mapping[idx] not in used_hosts
            # 2. For each neighbor j of idx that is already mapped, 
            #    frozenset((mapping[idx], mapping[j])) not in used_edge_set
            #    AND the edge (idx, j) must exist in the tree (it does if j is neighbor)
            
            # Get candidates: host vertices not yet used
            candidates = [h for h in range(n) if h not in used_hosts]
            
            # Prune: for each candidate, check edges to already-mapped neighbors
            valid_candidates = []
            for h in candidates:
                ok = True
                for nb in adj[idx]:
                    if mapping[nb] != -1:
                        if frozenset((h, mapping[nb])) in used_edge_set:
                            ok = False
                            break
                if ok:
                    valid_candidates.append(h)
            
            if not valid_candidates:
                return None
            
            # Try each valid candidate
            for h in valid_candidates:
                mapping[idx] = h
                used_hosts.add(h)
                result = dfs(idx + 1)
                if result is not None:
                    return result
                mapping[idx] = -1
                used_hosts.remove(h)
            
            return None
        
        result = dfs(1)
        if result is not None:
            return result
    
    return None

def construct():
    global _start_time
    _start_time = time.time()
    
    trees = _generate_trees(20, 611)
    n = 20
    total_edges = sum(len(t) for t in trees)  # should be 190
    
    # We need to embed all 20 trees edge-disjointly into K_20.
    # K_20 has 190 edges, and sum of tree edges = 190, so we need perfect packing.
    
    # Strategy: process trees in order of decreasing complexity (larger first)
    # to fail early.
    
    # Order: process trees from largest k to smallest, but T1 and T2 are trivial.
    # Let's process in order: T20, T19, ..., T3, T2, T1
    # Actually, let's think about which order minimizes backtracking.
    
    # For now, try: largest first
    order = list(range(20))  # index 0 = T1, index 19 = T20
    # Sort by decreasing number of edges
    order.sort(key=lambda i: len(trees[i]), reverse=True)
    
    used_edge_set = set()
    result = [None] * 20
    rng = random.Random(42)
    
    # For each tree in order, find an embedding
    success = True
    for idx in order:
        k = idx + 1  # tree size
        tree = trees[idx]
        
        if k == 1:
            # No edges, any vertex works. Pick first available.
            # Actually we just need a valid vertex; no edge constraint.
            # Pick 0.
            result[idx] = [0]
            continue
        
        if k == 2:
            # One edge. Find any unused edge.
            found = False
            for a in range(n):
                if _remaining_time() < 0.5:
                    break
                for b in range(a + 1, n):
                    if frozenset((a, b)) not in used_edge_set:
                        result[idx] = [a, b]
                        used_edge_set.add(frozenset((a, b)))
                        found = True
                        break
                if found:
                    break
            if not found:
                success = False
                break
            continue
        
        # For k >= 3, use backtracking
        emb = _find_embedding(tree, k, n, used_edge_set)
        if emb is None:
            success = False
            break
        
        # Verify and add edges
        for u, v in tree:
            eu, ev = emb[u], emb[v]
            edge = frozenset((eu, ev))
            if edge in used_edge_set:
                # Shouldn't happen if _find_embedding worked correctly
                success = False
                break
            used_edge_set.add(edge)
        if not success:
            break
        
        result[idx] = emb
    
    if not success:
        # Try a different order or randomized approach
        # Try multiple random restarts with different orderings
        for attempt in range(50):
            if _remaining_time() < 5.0:
                break
            
            # Shuffle the order of trees (keeping T1, T2 last since they're easy)
            order2 = list(range(20))
            # Keep T1 (index 0) and T2 (index 1) at the end
            main_order = list(range(2, 20))
            rng.shuffle(main_order)
            order2 = main_order + [1, 0]  # T2, T1 last
            
            used_edge_set2 = set()
            result2 = [None] * 20
            ok = True
            
            for idx in order2:
                k = idx + 1
                tree = trees[idx]
                
                if k == 1:
                    result2[idx] = [0]
                    continue
                
                if k == 2:
                    found = False
                    for a in range(n):
                        if _remaining_time() < 0.5:
                            break
                        for b in range(a + 1, n):
                            if frozenset((a, b)) not in used_edge_set2:
                                result2[idx] = [a, b]
                                used_edge_set2.add(frozenset((a, b)))
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        ok = False
                        break
                    continue
                
                emb = _find_embedding(tree, k, n, used_edge_set2)
                if emb is None:
                    ok = False
                    break
                
                for u, v in tree:
                    eu, ev = emb[u], emb[v]
                    edge = frozenset((eu, ev))
                    if edge in used_edge_set2:
                        ok = False
                        break
                    used_edge_set2.add(edge)
                if not ok:
                    break
                result2[idx] = emb
            
            if ok:
                # Verify all edges are used exactly once
                if len(used_edge_set2) == total_edges:
                    # Convert to proper format
                    final_result = []
                    for i in range(20):
                        if result2[i] is None:
                            final_result.append(None)
                        else:
                            final_result.append([int(x) for x in result2[i]])
                    return final_result
    
    # If all failed, return what we have (may be incomplete)
    # But we need a valid result. Return the first attempt's result if partially valid.
    # Actually, let's just return the best partial result.
    # For safety, return the identity-based result for T20 at least.
    fallback = [None] * 20
    fallback[19] = list(range(20))
    return fallback
