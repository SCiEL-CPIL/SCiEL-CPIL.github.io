import time
import random
import sys

def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_sa(edges, n, deadline):
        all_edges = set()
        for u, v in edges:
            all_edges.add((min(u, v), max(u, v)))
        
        rem_edges = set(all_edges)
        cycles = []
        
        def find_cycle_in(rem):
            if not rem:
                return None
            adj = {i: [] for i in range(n)}
            for u, v in rem:
                adj[u].append(v)
                adj[v].append(u)
            
            visited = set()
            parent = {}
            
            def dfs(u, start):
                visited.add(u)
                for v in adj[u]:
                    if v == start and len(visited) > 2:
                        cycle = [v]
                        cur = u
                        while cur != v:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.reverse()
                        return cycle
                    if v not in visited:
                        parent[v] = u
                        res = dfs(v, start)
                        if res:
                            return res
                return None
            
            for u in range(n):
                if adj[u]:
                    visited = set()
                    parent = {}
                    res = dfs(u, u)
                    if res:
                        return res
            return None

        while True:
            if not rem_edges:
                break
            cyc = find_cycle_in(rem_edges)
            if not cyc:
                break
            k = len(cyc)
            for i in range(k):
                u = cyc[i]
                v = cyc[(i+1)%k]
                e = (min(u,v), max(u,v))
                rem_edges.discard(e)
            cycles.append(cyc)
            
        current_cost = len(cycles) + len(rem_edges)
        rng_sa = Random(42)
        temp = 1.0
        cooling = 0.9995
        
        def get_rem_adj():
            adj = {i: set() for i in range(n)}
            for u, v in rem_edges:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        def is_valid_cycle(cyc):
            k = len(cyc)
            if k < 3:
                return False
            if len(set(cyc)) != k:
                return False
            for i in range(k):
                u = cyc[i]
                v = cyc[(i+1)%k]
                e = (min(u,v), max(u,v))
                if e not in rem_edges:
                    return False
            return True

        def find_random_cycle_in_rem():
            if not rem_edges:
                return None
            adj = get_rem_adj()
            u, v = rng_sa.choice(list(rem_edges))
            
            visited = {u: True}
            parent = {u: None}
            
            def dfs_cycle(u, start):
                for v in adj[u]:
                    if v == start:
                        if len(visited) > 2:
                            cycle = [start]
                            cur = u
                            while cur != start:
                                cycle.append(cur)
                                cur = parent[cur]
                            cycle.reverse()
                            return cycle
                    elif v not in visited:
                        visited[v] = True
                        parent[v] = u
                        res = dfs_cycle(v, start)
                        if res:
                            return res
                return None
            
            res = dfs_cycle(u, u)
            return res

        best_cost = current_cost
        best_cycles = [c[:] for c in cycles]
        best_rem = set(rem_edges)
        
        while time.time() < deadline:
            if rng_sa.random() < 0.5:
                if not cycles:
                    continue
                idx = rng_sa.randint(0, len(cycles)-1)
                cyc = cycles[idx]
                cycles.pop(idx)
                for i in range(len(cyc)):
                    u = cyc[i]
                    v = cyc[(i+1)%len(cyc)]
                    rem_edges.add((min(u,v), max(u,v)))
                
                new_cost = len(cycles) + len(rem_edges)
                delta = new_cost - current_cost
                
                if delta < 0 or rng_sa.expovariate(1.0/temp) > delta:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_cycles = [c[:] for c in cycles]
                        best_rem = set(rem_edges)
                else:
                    cycles.insert(idx, cyc)
                    for i in range(len(cyc)):
                        u = cyc[i]
                        v = cyc[(i+1)%len(cyc)]
                        rem_edges.discard((min(u,v), max(u,v)))
            else:
                if not rem_edges:
                    continue
                cyc = find_random_cycle_in_rem()
                if not cyc:
                    continue
                if not is_valid_cycle(cyc):
                    continue
                
                k = len(cyc)
                for i in range(k):
                    u = cyc[i]
                    v = cyc[(i+1)%k]
                    rem_edges.discard((min(u,v), max(u,v)))
                cycles.append(cyc)
                
                new_cost = len(cycles) + len(rem_edges)
                delta = new_cost - current_cost
                
                if delta < 0 or rng_sa.expovariate(1.0/temp) > delta:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_cycles = [c[:] for c in cycles]
                        best_rem = set(rem_edges)
                else:
                    cycles.pop()
                    for i in range(k):
                        u = cyc[i]
                        v = cyc[(i+1)%k]
                        rem_edges.add((min(u,v), max(u,v)))
            
            temp *= cooling
            if temp < 0.001:
                temp = 0.001
        
        pieces = []
        for cyc in best_cycles:
            pieces.append(list(cyc))
        for e in best_rem:
            pieces.append(list(e))
        return pieces

    t_start = time.time()
    total_budget = 115
    results = []
    for i, spec in enumerate(((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))):
        n = spec[0]
        edges = instance(*spec)
        if i < 2:
            dl = t_start + (i + 1) * (total_budget / 3.0)
        else:
            dl = t_start + total_budget
        pieces = decompose_sa(edges, n, dl)
        results.append(pieces)
        
    return results

if __name__ == "__main__":
    construct()
