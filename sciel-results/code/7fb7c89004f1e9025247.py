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
    used_edges = set()
    result = [None] * 20
    
    # Pre-compute adjacency lists for each tree
    tree_adjs = []
    for t in trees:
        k = len(t) + 1 if t else 1
        adj = [[] for _ in range(k)]
        for u, v in t:
            adj[u].append(v)
            adj[v].append(u)
        tree_adjs.append(adj)
        
    # Sort tree indices by complexity (larger trees first)
    indices = list(range(20))
    indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    for idx in indices:
        k = idx + 1
        if k == 1:
            result[idx] = None
            continue
            
        adj = tree_adjs[idx]
        host_assign = [-1] * k
        
        def get_available_hosts(tv):
            candidates = set(range(n))
            for u in range(k):
                if u != tv and host_assign[u] != -1:
                    candidates.discard(host_assign[u])
            
            final_candidates = []
            for h in candidates:
                valid = True
                for nu in adj[tv]:
                    if host_assign[nu] != -1:
                        hu = host_assign[nu]
                        if h == hu:
                            valid = False
                            break
                        edge = (min(h, hu), max(h, hu))
                        if edge in used_edges:
                            valid = False
                            break
                if valid:
                    final_candidates.append(h)
            return final_candidates

        def solve(assigned_count):
            if assigned_count == k:
                for u in range(k):
                    for v in adj[u]:
                        if u < v:
                            edge = (min(host_assign[u], host_assign[v]), max(host_assign[u], host_assign[v]))
                            if edge in used_edges:
                                return False
                            used_edges.add(edge)
                return True
                
            best_tv = -1
            best_cands = None
            min_count = float('inf')
            
            for tv in range(k):
                if host_assign[tv] == -1:
                    cands = get_available_hosts(tv)
                    if len(cands) < min_count:
                        min_count = len(cands)
                        best_tv = tv
                        best_cands = cands
                        if min_count == 0:
                            return False
                        if min_count == 1:
                            break
                
            if best_tv == -1:
                return False
                
            for h in best_cands:
                host_assign[best_tv] = h
                if solve(assigned_count + 1):
                    return True
                host_assign[best_tv] = -1
                
            return False

        # Try fixing root to 0 first (symmetry breaking)
        host_assign[0] = 0
        if solve(1):
            result[idx] = list(host_assign)
        else:
            host_assign[0] = -1
            success = False
            for r in range(1, n):
                host_assign[0] = r
                if solve(1):
                    success = True
                    break
                host_assign[0] = -1
            if success:
                result[idx] = list(host_assign)
            else:
                result[idx] = None

    return result
