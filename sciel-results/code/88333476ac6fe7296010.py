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
    deadline = start_time + 140
    
    n = 20
    trees = instance_trees(n)
    
    def validate(result):
        if result is None or len(result) != n:
            return False
        all_used = set()
        for i in range(n):
            if i == 0:
                if result[i] is not None:
                    return False
                continue
            if result[i] is None:
                return False
            mapping = result[i]
            if not isinstance(mapping, (list, tuple)):
                return False
            if len(mapping) != i + 1:
                return False
            if len(set(mapping)) != len(mapping):
                return False
            for v in mapping:
                if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v >= n:
                    return False
            for (u, v) in trees[i]:
                hu, hv = mapping[u], mapping[v]
                edge = (min(hu, hv), max(hu, hv))
                if edge in all_used:
                    return False
                all_used.add(edge)
        return True
    
    def greedy_pack(seed_offset):
        rng = Random(611 + seed_offset)
        used_edges = set()
        embeddings = [None] * n
        embeddings[0] = None
        embeddings[1] = [0, 1]
        used_edges.add((0, 1))
        
        # Largest trees first
        order = list(range(19, 2, -1))
        
        for tree_idx in order:
            k = tree_idx + 1
            tree = trees[tree_idx]
            found = False
            
            for _ in range(20000):
                if time.time() > deadline:
                    return None
                hosts = rng.sample(range(n), k)
                mapping = [hosts[i] for i in range(k)]
                
                valid = True
                new_edges = set()
                for (u, v) in tree:
                    hu, hv = mapping[u], mapping[v]
                    if hu == hv:
                        valid = False
                        break
                    edge = (min(hu, hv), max(hu, hv))
                    if edge in used_edges:
                        valid = False
                        break
                    new_edges.add(edge)
                
                if valid:
                    embeddings[tree_idx] = mapping
                    used_edges.update(new_edges)
                    found = True
                    break
            
            if not found:
                return None
        
        return embeddings
    
    # Try multiple random seeds
    for seed_offset in range(2000):
        if time.time() > deadline - 5:
            break
        result = greedy_pack(seed_offset)
        if result is not None and validate(result):
            return result
    
    # Fallback: just T20
    return [None] * 19 + [list(range(20))]
