# EVOLVE-BLOCK-START
import random
from collections import defaultdict

def construct():
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges, adj):
        m = len(edges)
        conflict = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    if (u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or
                        u2 in adj.get(v1, set()) or v2 in adj.get(v1, set())):
                        conflict[i].add(j)
                        conflict[j].add(i)
        return conflict

    def greedy_mis(conflict, available, rng):
        active = set(available)
        mis = set()
        while active:
            min_deg = float('inf')
            chosen = None
            for v in active:
                deg = len(conflict[v] & active)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
                elif deg == min_deg:
                    if chosen is None or rng.random() < 0.5:
                        chosen = v
            if chosen is None:
                break
            mis.add(chosen)
            active.discard(chosen)
            for nb in conflict[chosen]:
                active.discard(nb)
        return mis

    def local_search_add(conflict, current, rng):
        improved = True
        while improved:
            improved = False
            for v in range(len(conflict)):
                if v not in current and not (conflict[v] & current):
                    current.add(v)
                    improved = True
        return current

    def find_max_induced_matching(conflict, remaining, rng, restarts=50):
        best_set = set()
        best_size = 0
        rem_list = list(remaining)
        
        for _ in range(restarts):
            init_set = greedy_mis(conflict, remaining, rng)
            improved_set = local_search_add(conflict, init_set, rng)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set
        return best_set

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        rng = random.Random(seed + 12345)
        
        edges = instance(n, d, seed)
        m = len(edges)
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict = build_conflict_graph(edges, adj)
        
        color = [-1] * m
        remaining = set(range(m))
        color_idx = 0
        
        while remaining:
            selected = find_max_induced_matching(conflict, remaining, rng, restarts=50)
            for idx in selected:
                color[idx] = color_idx
                remaining.discard(idx)
            color_idx += 1
        
        out.append(color)
    
    return out
# EVOLVE-BLOCK-END
