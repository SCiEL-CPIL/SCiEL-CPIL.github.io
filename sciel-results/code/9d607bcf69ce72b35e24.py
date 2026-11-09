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
    trees = instance_trees(20, 611)
    n = 20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    def solve(order):
        available_edges = set(all_edges)
        result = [None] * n
        embedded_count = 0
        
        def embed_tree(tree_edges, k, available_edges_set):
            if k == 1:
                return [0]
            
            adj = {i: [] for i in range(k)}
            for u, v in tree_edges:
                adj[u].append(v)
                adj[v].append(u)
                
            map_ = [-1] * k
            map_[0] = 0
            
            def backtrack(idx):
                if idx == k:
                    return True
                
                candidates = []
                for i in range(k):
                    if map_[i] == -1:
                        assigned_neighbors = sum(1 for nb in adj[i] if map_[nb] != -1)
                        candidates.append((assigned_neighbors, i))
                
                candidates.sort(reverse=True)
                
                for _, i in candidates:
                    possible_hosts = set(range(n))
                    
                    for nb in adj[i]:
                        if map_[nb] != -1:
                            host_nb = map_[nb]
                            new_possible = set()
                            for h in possible_hosts:
                                if (min(host_nb, h), max(host_nb, h)) in available_edges_set:
                                    new_possible.add(h)
                            possible_hosts = new_possible
                            if not possible_hosts:
                                break
                    
                    if not possible_hosts:
                        continue
                    
                    host_list = list(possible_hosts)
                    # Shuffle to help with search? 
                    # Random(seed) might be needed for reproducibility if we shuffle
                    # But for now, let's just use the list.
                    
                    for h in host_list:
                        if h in map_:
                            continue
                        
                        valid = True
                        for nb in adj[i]:
                            if map_[nb] != -1:
                                host_nb = map_[nb]
                                edge = (min(host_nb, h), max(host_nb, h))
                                if edge not in available_edges_set:
                                    valid = False
                                    break
                        if not valid:
                            continue
                        
                        map_[i] = h
                        if backtrack(idx + 1):
                            return True
                        map_[i] = -1
                    return False
                return False
            
            if backtrack(1):
                return map_
            return None

        for k in order:
            tree_edges = trees[k - 1]
            mapping = embed_tree(tree_edges, k, available_edges)
            if mapping is None:
                return None, 0
            else:
                result[k - 1] = mapping
                for u, v in tree_edges:
                    host_u = mapping[u]
                    host_v = mapping[v]
                    edge = (min(host_u, host_v), max(host_u, host_v))
                    available_edges.remove(edge)
                embedded_count += len(tree_edges)
        
        return result, embedded_count

    # Try increasing order
    result_inc, count_inc = solve(range(1, n + 1))
    if result_inc is not None and count_inc == 190:
        return [ [int(x) for x in r] if r is not None else None for r in result_inc ]
        
    # Try decreasing order
    result_dec, count_dec = solve(range(n, 0, -1))
    if result_dec is not None and count_dec == 190:
        return [ [int(x) for x in r] if r is not None else None for r in result_dec ]
        
    # If neither got 190, return the best one found
    if count_inc >= count_dec:
        return [ [int(x) for x in r] if r is not None else None for r in result_inc ]
    else:
        return [ [int(x) for x in r] if r is not None else None for r in result_dec ]

