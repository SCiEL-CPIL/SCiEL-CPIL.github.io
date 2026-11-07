import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 15
    total_edges = n * (n - 1) // 2
    
    # Edge indexing
    edge_map = {}
    edge_index = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[idx] = (u, v)
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            idx += 1
    
    all_edge_ids = list(range(total_edges))
    
    def build_adj_deg(edges):
        adj = [set() for _ in range(n)]
        for e in edges:
            u, v = edge_map[e]
            adj[u].add(v)
            adj[v].add(u)
        deg = [len(adj[i]) for i in range(n)]
        return adj, deg
    
    def count_violations(adj, deg):
        violations = 0
        for u in range(n):
            for v in adj[u]:
                if u >= v:
                    continue
                su = adj[u] - {v}
                tv = adj[v] - {u}
                for s in su:
                    ds = deg[s]
                    for t in tv:
                        if t != s and deg[t] == ds:
                            violations += 1
        return violations

    def is_valid(edges):
        adj, deg = build_adj_deg(edges)
        return count_violations(adj, deg) == 0

    # K_{7,8} base
    A = list(range(7))
    B = list(range(7, 15))
    base_edges = set()
    for u in A:
        for v in B:
            base_edges.add(edge_index[(u, v)])
    
    best_edges = base_edges.copy()
    best_count = len(base_edges)
    
    # Greedy: try to add edges one by one in random order
    current = base_edges.copy()
    missing = [e for e in all_edge_ids if e not in current]
    random.shuffle(missing)
    
    for e in missing:
        if time.time() > deadline - 10:
            break
        test_edges = current | {e}
        if is_valid(test_edges):
            current = test_edges
            if len(current) > best_count:
                best_count = len(current)
                best_edges = current.copy()
    
    # Local search from best
    current = best_edges.copy()
    adj, deg = build_adj_deg(current)
    cur_viol = count_violations(adj, deg)
    
    T = 5.0
    min_T = 0.01
    for _ in range(30000):
        if time.time() > deadline - 3:
            break
        r = random.random()
        new_edges = current.copy()
        
        if r < 0.5:
            missing = [e for e in all_edge_ids if e not in current]
            if missing:
                e_add = random.choice(missing)
                new_edges.add(e_add)
        elif r < 0.7:
            if current:
                e_rem = random.choice(list(current))
                new_edges.remove(e_rem)
        else:
            if current:
                missing = [e for e in all_edge_ids if e not in current]
                if missing:
                    e_rem = random.choice(list(current))
                    e_add = random.choice(missing)
                    new_edges.remove(e_rem)
                    new_edges.add(e_add)
        
        new_adj, new_deg = build_adj_deg(new_edges)
        new_viol = count_violations(new_adj, new_deg)
        
        d_edges = len(new_edges) - len(current)
        d_viol = new_viol - cur_viol
        d_score = d_edges - 100 * d_viol
        
        accept = False
        if d_score >= 0:
            accept = True
        else:
            if T > min_T:
                prob = math.exp(d_score / T)
                if random.random() < prob:
                    accept = True
        
        if accept:
            current = new_edges
            adj = new_adj
            deg = new_deg
            cur_viol = new_viol
            
            if cur_viol == 0 and len(current) > best_count:
                best_count = len(current)
                best_edges = current.copy()
        
        T *= 0.999
    
    if not is_valid(best_edges):
        best_edges = base_edges.copy()
    
    result = []
    for e in best_edges:
        u, v = edge_map[e]
        result.append([u, v])
    
    return result
