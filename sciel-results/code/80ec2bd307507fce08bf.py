import time
import random
from itertools import permutations

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

def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    trees = instance_trees(20, 611)
    
    # Process trees in order of decreasing number of edges (most constrained first)
    # Tree k has k-1 edges, so process k=20, 19, ..., 2, 1
    order = list(range(19, -1, -1))  # indices 19 down to 0, i.e., T20, T19, ..., T1
    
    best_result = [None] * 20
    best_edges = -1
    
    def try_solution():
        nonlocal best_result, best_edges
        used_edges = set()
        embedding = [None] * 20
        
        for idx in order:
            k = idx + 1  # tree index k
            tree_edges = trees[idx]
            
            if k == 1:
                embedding[idx] = None
                continue
            
            # Try to find an embedding of this tree
            # Randomly try permutations of k vertices from 0..19
            attempts = 0
            max_attempts = 50000 if k <= 10 else 10000
            
            while attempts < max_attempts:
                if time.time() > deadline:
                    return False
                attempts += 1
                
                # Pick k random distinct vertices from 0..19
                verts = random.sample(range(20), k)
                # Map tree vertex i to verts[i]
                mapping = dict(zip(range(k), verts))
                
                # Check if all edges are available
                ok = True
                for (u, v) in tree_edges:
                    a, b = mapping[u], mapping[v]
                    edge = (min(a, b), max(a, b))
                    if edge in used_edges:
                        ok = False
                        break
                
                if ok:
                    # Commit this embedding
                    for (u, v) in tree_edges:
                        a, b = mapping[u], mapping[v]
                        edge = (min(a, b), max(a, b))
                        used_edges.add(edge)
                    embedding[idx] = [mapping[i] for i in range(k)]
                    break
            else:
                return False
        
        # All trees embedded successfully
        total_edges = len(used_edges)
        if total_edges > best_edges:
            best_edges = total_edges
            best_result = [list(e) if e is not None else None for e in embedding]
        return True
    
    # Try multiple random restarts
    restart = 0
    while time.time() < deadline:
        restart += 1
        random.seed(611 + restart)
        try_solution()
    
    # Return best result found
    if best_edges > 0:
        return best_result
    
    # Fallback: just T20 identity
    return [None] * 19 + [list(range(20))]
