"""Packing trees T1..T20 into K20"""
import time
from random import Random

def _generate_trees(n=20, seed=611):
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

def _find_embedding(tree_edges, k, used_edges, rng, deadline):
    """
    Find an embedding of a tree with k vertices and given edges into K20.
    Returns a list of length k where position i is the host vertex for tree vertex i.
    Returns None if not found.
    """
    if k == 1:
        for h in range(20):
            return [h]
        return None
    
    # Build adjacency
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS order from vertex 0
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
    
    n = 20
    assignment = [-1] * k
    host_used = [False] * n
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == len(order):
            return True
        
        v = order[idx]
        candidates = list(range(n))
        rng.shuffle(candidates)
        
        for h in candidates:
            if host_used[h]:
                continue
            # Check edges to already-assigned neighbors
            valid = True
            for nb in adj[v]:
                if assignment[nb] != -1:
                    edge = frozenset((h, assignment[nb]))
                    if edge in used_edges:
                        valid = False
                        break
            if valid:
                assignment[v] = h
                host_used[h] = True
                if backtrack(idx + 1):
                    return True
                assignment[v] = -1
                host_used[h] = False
        return False
    
    if backtrack(0):
        return [assignment[i] for i in range(k)]
    return None

def construct():
    deadline = time.time() + 140
    trees = _generate_trees(20, 611)
    
    best_result = [None] * 20
    best_value = 0
    
    for attempt in range(50000):
        if time.time() > deadline:
            break
        
        rng = Random(attempt * 7919 + 31337)
        used_edges = set()
        result = [None] * 20
        success = True
        value = 0
        
        for k in range(1, 21):
            if time.time() > deadline:
                success = False
                break
            
            tree_edges = trees[k - 1]
            
            if k == 1:
                h = rng.randint(0, 19)
                result[k - 1] = [h]
                continue
            
            emb = _find_embedding(tree_edges, k, used_edges, rng, deadline)
            if emb is None:
                success = False
                break
            
            result[k - 1] = emb
            value += len(tree_edges)
            
            for u, v in tree_edges:
                hu, hv = emb[u], emb[v]
                used_edges.add(frozenset((hu, hv)))
        
        if success and value > best_value:
            best_value = value
            best_result = result[:]
        
        if best_value == 190:
            break
    
    final = []
    for r in best_result:
        if r is None:
            final.append(None)
        else:
            final.append([int(x) for x in r])
    
    return final
