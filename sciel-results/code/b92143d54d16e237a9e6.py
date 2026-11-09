import time
import sys
from random import Random
from collections import deque
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _greedy_path_cover_fast(edges, n):
        """
        Fast greedy path cover using degree-based heuristics.
        """
        rem_edges = set(tuple(sorted(e)) for e in edges)
        adj = {i: set() for i in range(n)}
        for u, v in rem_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        paths = []
        
        while rem_edges:
            min_deg = float('inf')
            start_v = -1
            for v in range(n):
                deg = len(adj[v])
                if deg > 0 and deg < min_deg:
                    min_deg = deg
                    start_v = v
            
            if start_v == -1:
                break

            path = [start_v]
            visited_in_path = {start_v}
            current_v = start_v
            
            while True:
                candidates = []
                for nb in adj[current_v]:
                    if nb not in visited_in_path:
                        e_key = tuple(sorted((current_v, nb)))
                        if e_key in rem_edges:
                            candidates.append((len(adj[nb]), nb))
                
                if not candidates:
                    break
                
                candidates.sort()
                next_v = candidates[0][1]
                
                e_key = tuple(sorted((current_v, next_v)))
                rem_edges.discard(e_key)
                
                adj[current_v].discard(next_v)
                adj[next_v].discard(current_v)
                
                path.append(next_v)
                visited_in_path.add(next_v)
                current_v = next_v
            
            if len(path) > 1:
                paths.append(path)
            else:
                break

        return paths

    def _is_valid(paths, edges, n):
        edge_set = set(tuple(sorted(e)) for e in edges)
        used = set()
        for p in paths:
            if len(p) < 2:
                return False
            seen = set()
            for v in p:
                if v in seen:
                    return False
                seen.add(v)
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i + 1])))
                if e not in edge_set:
                    return False
                if e in used:
                    return False
                used.add(e)
        return used == edge_set

    def path_cover(edges, n):
        if not edges:
            return []
        
        res = _greedy_path_cover_fast(edges, n)
        
        if res and _is_valid(res, edges, n):
            return res
        
        fallback = []
        for e in edges:
            fallback.append([e[0], e[1]])
        return fallback

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            while len(results) < 3:
                edges = instance(*spec)
                results.append([list(e) for e in edges])
            break
        
        edges = instance(*spec)
        n = spec[0]
        paths = path_cover(edges, n)
        results.append(paths)
        
    return results

if __name__ == "__main__":
    res = construct()
    print("Number of paths:", [len(r) for r in res])
    print("Total paths:", sum(len(r) for r in res))
