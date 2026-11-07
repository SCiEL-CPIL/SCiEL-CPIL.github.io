from random import Random
import time
from collections import deque

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
    deadline = start_time + 115
    trees = instance_trees(20, 611)
    
    best_result = [None] * 20
    best_value = 0
    
    def embed_all(order, restart_seed):
        rng = Random(restart_seed)
        used_edges = set()
        result = [None] * 20
        
        for idx in order:
            if time.time() > deadline - 3:
                return result, False
            k = idx + 1
            edges = trees[idx]
            
            if k == 1:
                result[idx] = [rng.randint(0, 19)]
                continue
            
            if k == 2:
                found = False
                for _ in range(3000):
                    a = rng.randint(0, 19)
                    b = rng.randint(0, 19)
                    if a == b:
                        continue
                    edge = frozenset((a, b))
                    if edge not in used_edges:
                        result[idx] = [a, b]
                        used_edges.add(edge)
                        found = True
                        break
                if not found:
                    for a in range(20):
                        for b in range(a + 1, 20):
                            edge = frozenset((a, b))
                            if edge not in used_edges:
                                result[idx] = [a, b]
                                used_edges.add(edge)
                                found = True
                                break
                        if found:
                            break
                if not found:
                    return result, False
                continue
            
            adj = [[] for _ in range(k)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            visited = [False] * k
            bfs_order = [0]
            visited[0] = True
            queue = deque([0])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        bfs_order.append(v)
                        queue.append(v)
            
            host = [-1] * k
            used_hosts = set()
            node_count = [0]
            
            def bt(pos):
                if time.time() > deadline - 2 or node_count[0] > 50000:
                    return False
                node_count[0] += 1
                if pos == k:
                    return True
                v = bfs_order[pos]
                candidates = list(range(20))
                rng.shuffle(candidates)
                for h in candidates:
                    if h in used_hosts:
                        continue
                    valid = True
                    for u in adj[v]:
                        if host[u] != -1:
                            if frozenset((h, host[u])) in used_edges:
                                valid = False
                                break
                    if not valid:
                        continue
                    host[v] = h
                    used_hosts.add(h)
                    if bt(pos + 1):
                        return True
                    host[v] = -1
                    used_hosts.remove(h)
                return False
            
            found = False
            for attempt in range(100):
                if time.time() > deadline - 2:
                    break
                for i in range(k):
                    host[i] = -1
                used_hosts.clear()
                node_count[0] = 0
                if bt(0):
                    found = True
                    break
            
            if not found:
                return result, False
            
            result[idx] = list(host)
            for u, v in edges:
                used_edges.add(frozenset((host[u], host[v])))
        
        return result, True
    
    orders = [
        list(range(20)),
        list(range(19, -1, -1)),
    ]
    
    for oi, order in enumerate(orders):
        for restart in range(100):
            if time.time() > deadline - 5:
                break
            result, success = embed_all(order, 1000 + oi * 10000 + restart)
            value = sum(len(trees[idx]) for idx in range(20) if result[idx] is not None)
            if value > best_value:
                best_value = value
                best_result = [list(r) if r is not None else None for r in result]
            if best_value >= 190:
                return best_result
    
    return best_result
