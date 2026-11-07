from random import Random
import time

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: 
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    trees = instance_trees(20, seed=611)
    n = 20
    start_time = time.time()
    
    for seed in range(50000):
        if time.time() - start_time > 110:
            break
        rng = Random(seed)
        used_edges = set()
        assignment = [None] * n
        success = True
        
        for k in range(n, 1, -1):
            idx = k - 1
            tree_edges = trees[idx]
            found = False
            
            # More attempts for larger trees
            max_attempts = 200000 if k >= 18 else 100000 if k >= 15 else 50000
            
            for attempt in range(max_attempts):
                perm = list(range(n))
                rng.shuffle(perm)
                perm = perm[:k]
                
                valid = True
                for (u, v) in tree_edges:
                    e = (min(perm[u], perm[v]), max(perm[u], perm[v]))
                    if e in used_edges:
                        valid = False
                        break
                
                if valid:
                    assignment[idx] = list(perm)
                    for (u, v) in tree_edges:
                        e = (min(perm[u], perm[v]), max(perm[u], perm[v]))
                        used_edges.add(e)
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if success:
            assignment[0] = []
            result = [None if i == 0 else assignment[i-1] for i in range(n)]
            return result
    
    # Fallback
    result = [None] * n
    result[19] = list(range(n))
    return result
