import time
import random
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
    deadline = start_time + 110  # 10s margin
    
    trees = instance_trees()
    
    # We want to return entries for T1..T20
    # Index 0 is T1, Index 19 is T20
    result = [None] * 20
    
    used_edges = set()
    
    # Process trees from largest to smallest to constrain the space early
    # But we need to fill result[k-1] for T_k
    # T_k has k vertices and k-1 edges
    # T20 is index 19, T1 is index 0
    
    # Sort indices by size descending
    indices = list(range(20))
    indices.sort(key=lambda i: i+1, reverse=True)
    
    # For each tree, try to find an embedding
    for idx in indices:
        k = idx + 1
        if k == 1:
            # T1: 1 vertex, 0 edges. Any unused vertex? 
            # Actually, vertex can be reused. But we need to return a list of 1 distinct int.
            # Since no edges, any vertex 0..19 is valid.
            # Let's pick a random one or just 0.
            # To be safe, pick a random one.
            result[idx] = [random.randint(0, 19)]
            continue
            
        tree_edges = trees[idx]
        
        # Find embedding for this tree
        # Backtracking search
        max_restarts = 50
        found = None
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Randomize the order of vertices to assign
            # Or just use a fixed order but randomize the choice of host vertices
            # Let's use a randomized backtracking
            
            # Pre-compute adjacency of the tree
            adj = [[] for _ in range(k)]
            for u, v in tree_edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Order vertices by degree descending (heuristic)
            # But we must return mapping for original indices 0..k-1
            # So we assign to original indices, but we can choose the order of assignment
            # Let's assign in order of decreasing degree
            order = list(range(k))
            order.sort(key=lambda x: len(adj[x]), reverse=True)
            
            mapping = [-1] * k
            used_hosts = set()
            
            def backtrack(pos):
                if pos == len(order):
                    return True
                v = order[pos]
                # Try all host vertices
                # Shuffle host vertices for randomness
                hosts = list(range(20))
                random.shuffle(hosts)
                
                for h in hosts:
                    if h in used_hosts:
                        continue
                    
                    # Check constraints with already assigned neighbors
                    valid = True
                    for u in adj[v]:
                        if mapping[u] != -1:
                            # Edge (mapping[u], h) must not be in used_edges
                            e1 = min(mapping[u], h)
                            e2 = max(mapping[u], h)
                            if (e1, e2) in used_edges:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    # Assign
                    mapping[v] = h
                    used_hosts.add(h)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    # Unassign
                    mapping[v] = -1
                    used_hosts.remove(h)
                
                return False
            
            if backtrack(0):
                found = mapping
                break
        
        if found:
            result[idx] = found
            # Add edges to used_edges
            for u, v in tree_edges:
                e1 = min(found[u], found[v])
                e2 = max(found[u], found[v])
                used_edges.add((e1, e2))
        else:
            # Failed to embed this tree
            # Try smaller trees? No, we are going largest to smallest.
            # If we fail a large tree, maybe we should skip it and try the next?
            # Yes, set result[idx] = None
            pass
            
    return result
