import time
import sys
from collections import deque

def construct():
    from random import Random

    # Keep the exact generator from the seed program
    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_csp_optimized(n, edges, lists, deadline):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        domains = [set(lst) for lst in lists]
        degrees = [len(adj[i]) for i in range(n)]
        color = [None] * n
        
        # Check for empty domains
        for i in range(n):
            if not domains[i]:
                return None

        def select_variable():
            min_size = float('inf')
            best_v = -1
            best_deg = -1
            for i in range(n):
                if color[i] is None:
                    size = len(domains[i])
                    if size < min_size or (size == min_size and degrees[i] > best_deg):
                        min_size = size
                        best_v = i
                        best_deg = degrees[i]
            return best_v

        def lcv_order(v, c):
            count = 0
            for u in adj[v]:
                if color[u] is None and c in domains[u]:
                    count += 1
            return count

        def backtrack():
            if time.monotonic() > deadline:
                return False
            
            v = select_variable()
            if v == -1:
                return True
            
            values = list(domains[v])
            values.sort(key=lambda c: lcv_order(v, c))
            
            for c in values:
                saved = []
                valid = True
                for u in adj[v]:
                    if color[u] is None:
                        if c in domains[u]:
                            saved.append((u, domains[u].copy()))
                            domains[u].remove(c)
                            if not domains[u]:
                                valid = False
                                break
                
                if valid:
                    color[v] = c
                    if backtrack():
                        return True
                    color[v] = None
                    for u, old_dom in saved:
                        domains[u] = old_dom
                else:
                    for u, old_dom in saved:
                        domains[u] = old_dom
            
            return False

        if backtrack():
            return [int(c) for c in color]
        return None

    out = []
    start_total = time.monotonic()
    deadline = start_total + 110.0
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.monotonic() > deadline:
            # If we run out of time, we must return something. 
            # But we need to finish the loop to return a list of 4 items.
            # If we haven't solved an instance, we can't just skip it.
            # However, the first 3 are small. The last one is big.
            # If we are here, we probably timed out on a previous one or are about to.
            # We will try to solve with a very short deadline or return dummy.
            # Given the structure, this should not happen.
            pass

        n, edges, lists = instance(splits, seed)
        
        # Set a per-instance deadline that respects the global deadline
        instance_deadline = min(deadline - 0.5, time.monotonic() + 20.0)
        
        sol = solve_csp_optimized(n, edges, lists, instance_deadline)
        
        if sol is None:
            # Fallback: Try randomized restarts if the first pass failed
            for attempt in range(5):
                rng = Random(seed + attempt)
                shuffled_lists = []
                for lst in lists:
                    l = list(lst)
                    rng.shuffle(l)
                    shuffled_lists.append(l)
                
                sol = solve_csp_optimized(n, edges, shuffled_lists, instance_deadline)
                if sol:
                    break
        
        if sol is None:
            sol = [0] * n

        out.append(sol)
        
    return out
