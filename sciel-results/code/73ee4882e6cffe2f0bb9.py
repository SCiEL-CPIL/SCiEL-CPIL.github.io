"""Packing trees T1..T20 into K20 via randomized DFS backtracking."""
import time
import random
from collections import deque

def instance_trees(n=20, seed=611):
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

def _build_adj(edges, k):
    """Build adjacency list for tree with k vertices and given edges."""
    adj = [[] for _ in range(k)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _embed_tree_dfs(edges, k, used_edges, rng, deadline, node_limit=200000):
    """
    Try to embed a tree with k vertices and given edges into K20.
    used_edges is a set of frozenset({u,v}) for edges already used.
    Returns a list of k ints (the image of vertex 0..k-1) or None.
    """
    if k == 1:
        # Just need any free vertex
        for v in range(20):
            return [v]
    if k == 2:
        for u in range(20):
            for v in range(u+1, 20):
                e = frozenset((u, v))
                if e not in used_edges:
                    return [u, v]
        return None
    
    adj = _build_adj(edges, k)
    
    # Find a leaf to start
    leaves = [i for i in range(k) if len(adj[i]) == 1]
    if not leaves:
        leaves = [0]
    
    best = None
    nodes_visited = 0
    
    def dfs(pos, mapping, used_local):
        nonlocal best, nodes_visited
        if best is not None:
            return True
        if time.time() > deadline:
            return False
        nodes_visited += 1
        if nodes_visited > node_limit:
            return False
        
        if pos == k:
            best = list(mapping)
            return True
        
        # Choose which tree vertex to place next
        # Heuristic: place vertices in BFS order from root for better pruning
        # Actually, let's just place in order 0..k-1 but with backtracking
        # Better: use the tree structure. Place root first, then expand.
        
        # Let's use a different approach: build the embedding incrementally
        # mapping is a dict tree_vertex -> host_vertex
        
        # Find an unplaced tree vertex adjacent to a placed one
        candidates = []
        for tv in range(k):
            if tv not in mapping:
                # Check if any neighbor is already placed
                for nb in adj[tv]:
                    if nb in mapping:
                        candidates.append(tv)
                        break
                if not candidates:
                    # No placed neighbor, this can only be the first vertex
                    if len(mapping) == 0:
                        candidates.append(tv)
        
        if not candidates:
            if len(mapping) == 0:
                candidates = [0]
            else:
                return False  # disconnected, shouldn't happen for a tree
        
        # Order candidates: prefer those with more placed neighbors (more constrained)
        def constraint_score(tv):
            return sum(1 for nb in adj[tv] if nb in mapping)
        
        candidates.sort(key=constraint_score, reverse=True)
        
        # Shuffle candidates with some randomness for diversity
        rng.shuffle(candidates)
        
        for tv in candidates:
            if best is not None:
                return True
            
            # Determine which host vertices are available for tv
            # Must be adjacent in K20 to all already-placed neighbors
            available = set(range(20))
            for nb in adj[tv]:
                if nb in mapping:
                    hv = mapping[nb]
                    # In K20, tv's image must be != hv and the edge (hv, img) must be free
                    # Actually in K20 all pairs are connected, so just need edge not used
                    # and vertex not already used by another tree vertex
                    available = {x for x in available if x != hv}
            
            # Also, the image vertex must not already be used by another tree vertex
            used_hosts = set(mapping.values())
            available = available - used_hosts
            
            # Shuffle available for randomness
            avail_list = list(available)
            rng.shuffle(avail_list)
            
            for hv in avail_list:
                if best is not None:
                    return True
                if time.time() > deadline:
                    return False
                
                # Check all edges from tv to already-placed neighbors are free
                ok = True
                for nb in adj[tv]:
                    if nb in mapping:
                        e = frozenset((hv, mapping[nb]))
                        if e in used_local or e in used_edges:
                            ok = False
                            break
                
                if not ok:
                    continue
                
                mapping[tv] = hv
                new_used = used_local.copy()
                for nb in adj[tv]:
                    if nb in mapping:
                        new_used.add(frozenset((hv, mapping[nb])))
                
                if dfs(pos + 1, mapping, new_used):
                    return True
                
                del mapping[tv]
        
        return False
    
    # Start with a root
    for root in leaves:
        if best is not None:
            break
        if time.time() > deadline:
            break
        mapping = {root: 0}  # Start by trying host 0 for root
        # Actually, we should try multiple host starts
        # Let's restart: try each possible host for the root
        mapping = {}
        used_local = set()
        
        # Try placing root at various hosts
        for hv in range(20):
            if best is not None:
                break
            if time.time() > deadline:
                break
            mapping = {root: hv}
            used_local = set()
            if dfs(1, mapping, used_local):
                break
    
    if best is not None:
        # Update used_edges
        for i in range(k):
            for j in adj[i]:
                if i < j:
                    used_edges.add(frozenset((best[i], best[j])))
        return best
    return None

def _embed_tree_bfs(edges, k, used_edges, rng, deadline):
    """
    Greedy BFS embedding: place root, then expand level by level.
    """
    if k == 1:
        for v in range(20):
            return [v]
    if k == 2:
        for u in range(20):
            for v in range(u+1, 20):
                e = frozenset((u, v))
                if e not in used_edges:
                    return [u, v]
        return None
    
    adj = _build_adj(edges, k)
    leaves = [i for i in range(k) if len(adj[i]) == 1]
    if not leaves:
        leaves = [0]
    
    best = None
    for root in leaves:
        if best is not None:
            break
        if time.time() > deadline:
            break
        
        # Try BFS from root
        for hv_root in range(20):
            if best is not None:
                break
            if time.time() > deadline:
                break
            
            mapping = {root: hv_root}
            used_local = set()
            queue = deque([root])
            success = True
            
            while queue and success:
                if time.time() > deadline:
                    success = False
                    break
                tv = queue.popleft()
                hv = mapping[tv]
                
                for nb in adj[tv]:
                    if nb in mapping:
                        continue
                    
                    # Find a free host vertex for nb
                    # Must be adjacent to hv (in K20, all are) and edge (hv, x) free
                    # And x not already used
                    used_hosts = set(mapping.values())
                    candidates = [x for x in range(20) if x != hv and x not in used_hosts]
                    rng.shuffle(candidates)
                    
                    placed = False
                    for x in candidates:
                        e = frozenset((hv, x))
                        if e in used_local or e in used_edges:
                            continue
                        # Check if x conflicts with other already-placed neighbors of nb
                        ok = True
                        for nb2 in adj[nb]:
                            if nb2 in mapping:
                                e2 = frozenset((x, mapping[nb2]))
                                if e2 in used_local or e2 in used_edges:
                                    ok = False
                                    break
                        if not ok:
                            continue
                        
                        mapping[nb] = x
                        used_local.add(e)
                        for nb2 in adj[nb]:
                            if nb2 in mapping and nb2 != tv:
                                used_local.add(frozenset((x, mapping[nb2])))
                        queue.append(nb)
                        placed = True
                        break
                    
                    if not placed:
                        success = False
                        break
            
            if success and len(mapping) == k:
                best = [mapping[i] for i in range(k)]
                for i in range(k):
                    for j in adj[i]:
                        if i < j:
                            used_edges.add(frozenset((best[i], best[j])))
                break
    
    return best

def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill
    trees = instance_trees()
    n = 20
    
    # Order of embedding: try multiple strategies
    # Strategy 1: Ascending (small trees first)
    # Strategy 2: Descending (large trees first)
    # Strategy 3: Random order
    
    orders = [
        list(range(n)),           # ascending
        list(range(n-1, -1, -1)), # descending
    ]
    
    # Add a few random orders
    rng_master = random.Random(42)
    for _ in range(3):
        o = list(range(n))
        rng_master.shuffle(o)
        orders.append(o)
    
    best_result = None
    best_count = 0
    
    for order in orders:
        if time.time() > deadline - 5:
            break
        
        used_edges = set()
        result = [None] * n
        success = True
        
        for k_idx in order:
            if time.time() > deadline - 5:
                success = False
                break
            k = k_idx + 1
            edges = trees[k_idx]
            
            if k == 1:
                # T1 has no edges, just needs a vertex
                for v in range(20):
                    result[k_idx] = [v]
                    break
                continue
            
            if k == 2:
                found = False
                for u in range(20):
                    for v in range(u+1, 20):
                        e = frozenset((u, v))
                        if e not in used_edges:
                            result[k_idx] = [u, v]
                            used_edges.add(e)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    success = False
                    break
                continue
            
            # Use DFS backtracking
            rng = random.Random(hash((order, k_idx)) % (2**32))
            emb = _embed_tree_dfs(edges, k, used_edges, rng, deadline)
            if emb is None:
                # Try BFS as fallback
                emb = _embed_tree_bfs(edges, k, used_edges, rng, deadline)
            if emb is None:
                success = False
                break
            result[k_idx] = emb
        
        if success:
            count = sum(1 for r in result if r is not None)
            if count > best_count:
                best_count = count
                best_result = result[:]
        
        # Count partial
        count = sum(1 for r in result if r is not None)
        if count > best_count:
            best_count = count
            best_result = result[:]
    
    if best_result is None:
        # Fallback: just T20
        return [None] * 19 + [list(range(20))]
    
    return best_result
