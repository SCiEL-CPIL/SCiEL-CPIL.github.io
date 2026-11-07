"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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
    deadline = start_time + 110.0  # 10 second safety margin
    
    trees = instance_trees(20, 611)
    
    # T1 is empty, T2 is a single edge
    # We need to embed T1..T20 into K20 with edge-disjoint embeddings
    
    # Key insight: K20 has C(20,2) = 190 edges
    # Trees T1..T20 have total edges: 0 + 1 + 2 + ... + 19 = 190
    # So we need to use ALL edges exactly once - a perfect edge decomposition
    
    # Strategy: Use a greedy/backtracking approach to embed trees
    # Process trees from largest to smallest for better constraint propagation
    
    # First, let's try a systematic approach
    # Since total edges = 190 = edges in K20, we need exact packing
    
    # Try backtracking with pruning
    # Embed trees in order: T20, T19, ..., T2 (T1 is trivial)
    
    def solve(order, used_edges, embeddings):
        nonlocal deadline
        if time.time() > deadline:
            return None
        
        if not order:
            return embeddings
        
        tree_idx = order[0]
        tree = trees[tree_idx]
        k = tree_idx + 1  # number of vertices
        
        # Try to embed this tree
        # We need to assign k distinct host vertices such that all tree edges
        # map to unused edges
        
        def try_embed(tree_edges, k):
            # Backtracking search for vertex assignment
            # tree_edges is list of (u, v) where u, v are in 0..k-1
            # We need to assign each vertex 0..k-1 to a distinct host vertex
            
            # Build adjacency for the tree
            adj = [[] for _ in range(k)]
            for u, v in tree_edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Start with vertex 0 mapped to some host vertex
            # Use DFS with constraint checking
            
            def dfs(assign, used_hosts):
                if time.time() > deadline:
                    return None
                
                # Check if all vertices assigned
                if len(assign) == k:
                    # Verify all edges are available
                    for u, v in tree_edges:
                        hu = assign[u]
                        hv = assign[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            return None
                    return assign.copy()
                
                # Pick the next unassigned vertex
                next_v = min(set(range(k)) - set(assign.keys()))
                
                # Try each available host vertex
                for h in range(20):
                    if h in used_hosts:
                        continue
                    
                    # Check if edges from next_v to already-assigned neighbors are valid
                    valid = True
                    for nb in adj[next_v]:
                        if nb in assign:
                            hn = assign[nb]
                            e = (min(h, hn), max(h, hn))
                            if e in used_edges:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    assign[next_v] = h
                    used_hosts.add(h)
                    
                    result = dfs(assign, used_hosts)
                    if result is not None:
                        return result
                    
                    del assign[next_v]
                    used_hosts.remove(h)
                
                return None
            
            return dfs({}, set())
        
        # Try multiple random orderings of host vertices for better performance
        best_embedding = None
        
        # First try with a fixed ordering
        emb = try_embed(tree, k)
        if emb is not None:
            # Build new used_edges
            new_used = set(used_edges)
            for u, v in tree:
                hu = emb[u]
                hv = emb[v]
                new_used.add((min(hu, hv), max(hu, hv)))
            
            new_emb = embeddings.copy()
            new_emb[tree_idx] = [emb[i] for i in range(k)]
            
            result = solve(order[1:], new_used, new_emb)
            if result is not None:
                return result
        
        return None
    
    # Process trees from largest to smallest
    order = list(range(19, 1, -1))  # T20 down to T2
    
    embeddings = [None] * 20
    embeddings[0] = [0]  # T1: single vertex
    
    result = solve(order, set(), embeddings)
    
    if result is not None:
        # Ensure all entries are proper lists
        for i in range(20):
            if result[i] is not None:
                result[i] = list(result[i])
        return result
    
    # Fallback: return what we can
    fallback = [None] * 20
    fallback[0] = [0]
    
    # Try to embed as many trees as possible with a simpler greedy approach
    used_edges = set()
    for i in range(1, 20):
        if time.time() > deadline:
            break
        tree = trees[i]
        k = i + 1
        if not tree:
            fallback[i] = [0]
            continue
        if len(tree) == 1:
            # Single edge, find any unused edge
            found = False
            for u in range(20):
                if found:
                    break
                for v in range(u + 1, 20):
                    if (u, v) not in used_edges:
                        fallback[i] = [u, v]
                        used_edges.add((u, v))
                        found = True
                        break
            if not found:
                break
            continue
        # For larger trees, try a simple greedy
        # Map vertices in order to available host vertices
        hosts = list(range(20))
        assign = {}
        valid = True
        for tv in range(k):
            if not hosts:
                valid = False
                break
            # Pick first host that doesn't create conflict
            chosen = None
            for h in hosts:
                # Check edges to already assigned neighbors
                ok = True
                for u, v in tree:
                    if u == tv and v in assign:
                        e = (min(h, assign[v]), max(h, assign[v]))
                        if e in used_edges:
                            ok = False
                            break
                    elif v == tv and u in assign:
                        e = (min(h, assign[u]), max(h, assign[u]))
                        if e in used_edges:
                            ok = False
                            break
                if ok:
                    chosen = h
                    break
            if chosen is None:
                valid = False
                break
            assign[tv] = chosen
            hosts.remove(chosen)
        
        if valid:
            # Final check
            ok = True
            for u, v in tree:
                e = (min(assign[u], assign[v]), max(assign[u], assign[v]))
                if e in used_edges:
                    ok = False
                    break
            if ok:
                for u, v in tree:
                    e = (min(assign[u], assign[v]), max(assign[u], assign[v]))
                    used_edges.add(e)
                fallback[i] = [assign[i2] for i2 in range(k)]
            else:
                break
        else:
            break
    
    return fallback

# EVOLVE-BLOCK-END
