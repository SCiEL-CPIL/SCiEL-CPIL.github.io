import time
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
    trees = instance_trees(20, 611)
    
    # Process from largest to smallest
    order = list(range(19, -1, -1))
    
    for attempt in range(200):
        if time.time() - start_time > 110:
            break
        rng = Random(attempt * 7919 + 13)
        used_edges = set()
        result = [None] * 20
        
        success = True
        for idx in order:
            k = idx + 1
            tree_edges = trees[idx]
            if k == 1:
                result[idx] = None
                continue
            
            found = False
            # Try random embeddings
            for _ in range(3000):
                perm = list(range(20))
                rng.shuffle(perm)
                # Check edges
                valid = True
                for (u, v) in tree_edges:
                    mu, mv = perm[u], perm[v]
                    if mu == mv:
                        valid = False
                        break
                    edge = (min(mu, mv), max(mu, mv))
                    if edge in used_edges:
                        valid = False
                        break
                if valid:
                    for (u, v) in tree_edges:
                        mu, mv = perm[u], perm[v]
                        edge = (min(mu, mv), max(mu, mv))
                        used_edges.add(edge)
                    result[idx] = [int(perm[i]) for i in range(k)]
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if success:
            final_result = []
            for idx in range(20):
                if result[idx] is None:
                    final_result.append(None)
                else:
                    final_result.append([int(x) for x in result[idx]])
            return final_result
    
    # Fallback
    return [None] * 19 + [list(range(20))]
