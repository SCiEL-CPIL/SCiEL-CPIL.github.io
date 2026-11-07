import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    trees = instance_trees(20, 611)
    n = 20
    
    tree_info = []
    for i in range(n):
        k = i + 1
        if k == 1:
            tree_info.append({'k': 1, 'edges': [], 'adj': [[]], 'deg': [0]})
            continue
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for (a, b) in trees[i]:
            adj[a].append(b)
            adj[b].append(a)
            deg[a] += 1
            deg[b] += 1
        tree_info.append({'k': k, 'edges': trees[i], 'adj': adj, 'deg': deg})
    
    def try_packing(seed_offset):
        rng = Random(seed_offset)
        used_edges = set()
        result = [None] * n
        
        order = list(range(n - 1, -1, -1))
        
        for tree_idx in order:
            info = tree_info[tree_idx]
            k = info['k']
            
            if k == 1:
                result[tree_idx] = [0]
                continue
            
            if k == 2:
                placed = False
                for u in range(n):
                    for v in range(u + 1, n):
                        if (u, v) not in used_edges:
                            used_edges.add((u, v))
                            result[tree_idx] = [u, v]
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    return None
                continue
            
            adj = info['adj']
            deg = info['deg']
            root_candidates = sorted(range(k), key=lambda v: -deg[v])
            
            found = False
            for root in root_candidates:
                parent = [-1] * k
                visited = [False] * k
                visited[root] = True
                queue = [root]
                assign_order = [root]
                while queue:
                    u = queue.pop(0)
                    for w in adj[u]:
                        if not visited[w]:
                            visited[w] = True
                            parent[w] = u
                            assign_order.append(w)
                            queue.append(w)
                
                if time.time() > deadline:
                    return None
                
                host_map = [-1] * k
                local_used_hosts = set()
                
                def backtrack(pos):
                    if time.time() > deadline:
                        return False
                    if pos == k:
                        for (a, b) in info['edges']:
                            ha, hb = host_map[a], host_map[b]
                            key = (min(ha, hb), max(ha, hb))
                            if key in used_edges:
                                return False
                        for (a, b) in info['edges']:
                            ha, hb = host_map[a], host_map[b]
                            key = (min(ha, hb), max(ha, hb))
                            used_edges.add(key)
                        return True
                        
                    v = assign_order[pos]
                    if pos == 0:
                        candidates = list(range(n))
                    else:
                        pv = parent[v]
                        ph = host_map[pv]
                        candidates = []
                        for h in range(n):
                            if h in local_used_hosts:
                                continue
                            key = (min(ph, h), max(ph, h))
                            if key not in used_edges:
                                candidates.append(h)
                    
                    rng.shuffle(candidates)
                    
                    for h in candidates:
                        host_map[pos] = h
                        local_used_hosts.add(h)
                        
                        if backtrack(pos + 1):
                            return True
                        
                        local_used_hosts.discard(h)
                        host_map[pos] = -1
                        
                    return False
                
                if backtrack(0):
                    result[tree_idx] = [host_map[v] for v in range(k)]
                    found = True
                    break
                else:
                    local_used_hosts.clear()
                    host_map = [-1] * k
                    
            if not found:
                return None
                
        return result
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        res = try_packing(attempt)
        if res is not None:
            total_edges = 0
            all_edges = set()
            valid = True
            for i in range(n):
                if res[i] is None:
                    valid = False
                    break
                k = i + 1
                if len(res[i]) != k:
                    valid = False
                    break
                if len(set(res[i])) != k:
                    valid = False
                    break
                for h in res[i]:
                    if not (0 <= h < n) or not isinstance(h, int):
                        valid = False
                        break
                if not valid:
                    break
                for (a, b) in trees[i]:
                    ha, hb = res[i][a], res[i][b]
                    key = (min(ha, hb), max(ha, hb))
                    if key in all_edges:
                        valid = False
                        break
                    all_edges.add(key)
                    total_edges += 1
                if not valid:
                    break
            if valid and total_edges == 190:
                return res
    
    # Fallback: return whatever we can
    used_edges = set()
    result = [None] * n
    order = list(range(n - 1, -1, -1))
    
    for tree_idx in order:
        info = tree_info[tree_idx]
        k = info['k']
        
        if k == 1:
            result[tree_idx] = [0]
            continue
            
        if k == 2:
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in used_edges:
                        used_edges.add((u, v))
                        result[tree_idx] = [u, v]
                        break
                if result[tree_idx] is not None:
                    break
            continue
            
        adj = info['adj']
        deg = info['deg']
        root_candidates = sorted(range(k), key=lambda v: -deg[v])
        
        found = False
        for root in root_candidates:
            parent = [-1] * k
            visited = [False] * k
            visited[root] = True
            queue = [root]
            assign_order = [root]
            while queue:
                u = queue.pop(0)
                for w in adj[u]:
                    if not visited[w]:
                        visited[w] = True
                        parent[w] = u
                        assign_order.append(w)
                        queue.append(w)
                        
            host_map = [-1] * k
            local_used_hosts = set()
            
            def backtrack(pos):
                if time.time() > deadline:
                    return False
                if pos == k:
                    for (a, b) in info['edges']:
                        ha, hb = host_map[a], host_map[b]
                        key = (min(ha, hb), max(ha, hb))
                        if key in used_edges:
                            return False
                    for (a, b) in info['edges']:
                        ha, hb = host_map[a], host_map[b]
                        key = (min(ha, hb), max(ha, hb))
                        used_edges.add(key)
                    return True
                    
                v = assign_order[pos]
                if pos == 0:
                    candidates = list(range(n))
                else:
                    pv = parent[v]
                    ph = host_map[pv]
                    candidates = []
                    for h in range(n):
                        if h in local_used_hosts:
                            continue
                        key = (min(ph, h), max(ph, h))
                        if key not in used_edges:
                            candidates.append(h)
                
                for h in candidates:
                    host_map[pos] = h
                    local_used_hosts.add(h)
                    if backtrack(pos + 1):
                        return True
                    local_used_hosts.discard(h)
                    host_map[pos] = -1
                return False
            
            if backtrack(0):
                result[tree_idx] = [host_map[v] for v in range(k)]
                found = True
                break
            else:
                local_used_hosts.clear()
                host_map = [-1] * k
                
        if not found:
            result[tree_idx] = None
            
    return result
