import time
import random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = random.Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
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

def _edge_key(a, b):
    return (a, b) if a < b else (b, a)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin under 120s
    
    n = 20
    trees = instance_trees(n, 611)
    
    tree_info = []
    for i in range(n):
        k = i + 1
        edges = trees[i]
        if k <= 1:
            max_deg = 0
        else:
            degs = [0] * k
            for u, v in edges:
                degs[u] += 1
                degs[v] += 1
            max_deg = max(degs)
        tree_info.append((i, k, max_deg, edges))
        
    # Sort by size descending, then max degree descending
    tree_info.sort(key=lambda x: (-x[1], -x[2]))
    
    best_result = [None] * n
    best_score = -1
    
    max_attempts = 5000
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        rng = random.Random(attempt)
        used_edges = set()
        result = [None] * n
        success = True
        
        for idx, k, max_deg, edges in tree_info:
            if time.time() > deadline:
                success = False
                break
                
            if k == 1:
                h = rng.randint(0, n - 1)
                result[idx] = [h]
                continue
                
            if k == 2:
                found = False
                for _ in range(200):
                    u = rng.randint(0, n - 1)
                    v = rng.randint(0, n - 1)
                    if u == v:
                        continue
                    ek = _edge_key(u, v)
                    if ek not in used_edges:
                        result[idx] = [u, v]
                        used_edges.add(ek)
                        found = True
                        break
                if not found:
                    for u in range(n):
                        for v in range(u+1, n):
                            ek = (u, v)
                            if ek not in used_edges:
                                result[idx] = [u, v]
                                used_edges.add(ek)
                                found = True
                                break
                        if found:
                            break
                if not found:
                    success = False
                    break
                continue
            
            embedding = _embed_tree_backtrack(edges, k, used_edges, n, rng)
            if embedding is None:
                success = False
                break
                
            result[idx] = embedding
            for u, v in edges:
                hu = embedding[u]
                hv = embedding[v]
                ek = _edge_key(hu, hv)
                if ek in used_edges:
                    success = False
                    break
                used_edges.add(ek)
            
            if not success:
                break
        
        if success:
            score = sum(len(t) for t in trees)
            if score > best_score:
                best_score = score
                best_result = [list(r) if r is not None else None for r in result]
                if best_score == 190:
                    break
    
    out = []
    for i in range(n):
        if best_result[i] is None:
            out.append(None)
        else:
            out.append([int(x) for x in best_result[i]])
            
    return out

def _embed_tree_backtrack(tree_edges, k, used_edges, host_n, rng):
    if k == 1:
        return [rng.randint(0, host_n - 1)]
        
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Start from highest degree node for better constraint propagation
    start_node = max(range(k), key=lambda v: len(adj[v]))
    
    # BFS order from start_node
    order = [start_node]
    visited = {start_node}
    queue = deque([start_node])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                order.append(v)
                queue.append(v)
                
    mapping = [-1] * k
    used_host_verts = [False] * host_n
    
    # Pre-compute neighbor indices in order
    neighbor_indices = [[] for _ in range(k)]
    for i in range(k):
        tv = order[i]
        for j in range(i):
            if order[j] in adj[tv]:
                neighbor_indices[i].append(j)
                
    def backtrack(idx):
        if idx == k:
            return True
            
        tv = order[idx]
        
        valid_hosts = []
        for hv in range(host_n):
            if used_host_verts[hv]:
                continue
            
            valid = True
            for j in neighbor_indices[idx]:
                nhv = mapping[order[j]]
                ek = _edge_key(hv, nhv)
                if ek in used_edges:
                    valid = False
                    break
            if valid:
                valid_hosts.append(hv)
                
        if not valid_hosts:
            return False
            
        # Sort by degree constraint: prefer hosts with fewer available neighbors
        # This is a simple heuristic to fail early
        def host_constraint(hv):
            count = 0
            for nh in range(host_n):
                if not used_host_verts[nh] and _edge_key(hv, nh) not in used_edges:
                    count += 1
            return count
            
        valid_hosts.sort(key=host_constraint)
        # Add some randomness
        if len(valid_hosts) > 1:
            for i in range(len(valid_hosts) - 1, 0, -1):
                j = rng.randint(0, i)
                valid_hosts[i], valid_hosts[j] = valid_hosts[j], valid_hosts[i]
                
        for hv in valid_hosts:
            mapping[tv] = hv
            used_host_verts[hv] = True
            
            if backtrack(idx + 1):
                return True
                
            mapping[tv] = -1
            used_host_verts[hv] = False
            
        return False
        
    if backtrack(0):
        return mapping
        
    return None
