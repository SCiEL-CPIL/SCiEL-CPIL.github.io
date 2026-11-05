import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def is_c4_free(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for i in range(n):
            for j in range(i+1, n):
                if len(adj[i] & adj[j]) >= 2:
                    return False
        return True

    def min_degree(n, edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return min(deg) if n > 0 else 0

    def validate(n, edges):
        if not is_c4_free(n, edges):
            return False
        return min_degree(n, edges) >= n - 39

    def find_c4(n, adj):
        """Find a C4 in the graph represented by adj. Returns (u, v, w, x) or None."""
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    common = adj[u] & adj[v]
                    if len(common) >= 2:
                        common_list = list(common)
                        w = common_list[0]
                        x = common_list[1]
                        return (u, w, v, x)
        return None

    def build_c4_free_graph_deletion(n, min_deg_target, max_time=10.0):
        try_start = time.time()
        adj = [set() for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.5:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
        
        def would_create_c4(u, v):
            for a in adj[v]:
                if a == u: continue
                for b in adj[u]:
                    if b == v: continue
                    if a == b: continue
                    if b in adj[a]:
                        return True
            return False

        while time.time() < try_start + max_time:
            c4 = find_c4(n, adj)
            if c4 is None:
                break
            u, w, v, x = c4
            candidates = [(u, w), (w, v), (v, x), (x, u)]
            valid_candidates = []
            for e in candidates:
                if e[1] in adj[e[0]]:
                    valid_candidates.append(e)
            if not valid_candidates:
                break
            e = random.choice(valid_candidates)
            a, b = e
            adj[a].remove(b)
            adj[b].remove(a)
            if (a, b) in edges:
                edges.remove((a, b))
            elif (b, a) in edges:
                edges.remove((b, a))
            
        degs = [len(adj[i]) for i in range(n)]
        if min(degs) >= min_deg_target:
            return edges
        
        for _ in range(1000):
            if time.time() > try_start + max_time:
                break
            degs = [len(adj[i]) for i in range(n)]
            if min(degs) >= min_deg_target:
                break
            
            min_v = min(range(n), key=lambda i: degs[i])
            potential_u = [u for u in range(n) if u != min_v and u not in adj[min_v]]
            random.shuffle(potential_u)
            added = False
            for u in potential_u:
                if not would_create_c4(min_v, u):
                    adj[min_v].add(u)
                    adj[u].add(min_v)
                    edges.append((min_v, u))
                    added = True
                    break
            if not added:
                break
                
        degs = [len(adj[i]) for i in range(n)]
        if min(degs) >= min_deg_target:
            return edges
            
        return None

    for n in [47, 46, 45, 44]:
        if time.time() > deadline:
            break
        min_deg_target = n - 39
        found = False
        attempts = 0
        while time.time() < deadline - 10 and attempts < 100:
            attempts += 1
            random.seed(attempts * 12345 + n)
            edges = build_c4_free_graph_deletion(n, min_deg_target, max_time=2.0)
            if edges is not None:
                if is_c4_free(n, edges) and min_degree(n, edges) >= min_deg_target:
                    return {'n': n, 'edges': [[u, v] for u, v in edges]}
        
    return {'n': 39, 'edges': []}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
