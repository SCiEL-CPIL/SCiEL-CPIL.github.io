import sys
import time
from random import Random

def construct():
    sys.setrecursionlimit(10000)
    
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

    def solve(n, edges, lists, rng):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Build reverse adjacency for efficient arc consistency
        domains = [set(lst) for lst in lists]
        
        # AC-3 preprocessing
        def arc_consistency(domains, adj):
            queue = []
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            
            while queue:
                u, v = queue.pop(0)
                # Check if domain of u needs revision w.r.t. v
                revised = False
                for c in list(domains[u]):
                    # If c has no support in v's domain, remove c from u
                    if not any(c != d for d in domains[v]):
                        domains[u].discard(c)
                        revised = True
                        if not domains[u]:
                            return False
                if revised:
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))
            return True
        
        # Try arc consistency
        if not arc_consistency(domains, adj):
            return None
        
        assignment = [None] * n
        unassigned = set(range(n))
        
        def backtrack():
            if not unassigned:
                return True
            
            # MRV: minimum domain size, tie-break by degree
            min_size = n + 1
            best_v = -1
            best_deg = -1
            for i in unassigned:
                sz = len(domains[i])
                deg = len(adj[i])
                if sz < min_size or (sz == min_size and deg > best_deg):
                    min_size = sz
                    best_v = i
                    best_deg = deg
                    if sz == 0:
                        return False
                    if sz == 1:
                        break
            
            if best_v == -1:
                return True
            
            v = best_v
            colors = list(domains[v])
            rng.shuffle(colors)
            
            for c in colors:
                assignment[v] = c
                unassigned.remove(v)
                
                # Save and propagate
                old_domains = {}
                consistent = True
                for u in adj[v]:
                    if u in unassigned:
                        old_domains[u] = domains[u].copy()
                        domains[u].discard(c)
                        if not domains[u]:
                            consistent = False
                            break
                
                if consistent and backtrack():
                    return True
                
                # Restore
                for u in old_domains:
                    domains[u] = old_domains[u]
                unassigned.add(v)
                assignment[v] = None
            
            return False
        
        # First attempt with given rng
        if backtrack():
            return [int(c) for c in assignment]
        
        # Try a few restarts with different seeds
        for attempt in range(5):
            # Reset state
            domains = [set(lst) for lst in lists]
            if not arc_consistency(domains, adj):
                continue
            assignment = [None] * n
            unassigned = set(range(n))
            rng2 = Random(rng.getstate()[1][0] + attempt * 1000)
            
            # Monkey-patch rng for this call
            old_rng = rng
            rng = rng2
            result = backtrack()
            rng = old_rng
            
            if result:
                return [int(c) for c in assignment]
        
        return None

    out = []
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 5.0:
            # Not enough time for remaining instances; use greedy fallback
            n, edges, lists = instance(splits, seed)
            # Greedy: assign first available color that doesn't conflict with already-assigned neighbors
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
            sol = [None] * n
            for v in order:
                used = set()
                for u in adj[v]:
                    if sol[u] is not None:
                        used.add(sol[u])
                for c in lists[v]:
                    if c not in used:
                        sol[v] = c
                        break
                if sol[v] is None:
                    sol[v] = lists[v][0]
            out.append([int(c) for c in sol])
            continue
        
        n, edges, lists = instance(splits, seed)
        search_rng = Random(seed + 1)
        sol = solve(n, edges, lists, search_rng)
        
        if sol is None:
            # Greedy fallback
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
            sol = [None] * n
            for v in order:
                used = set()
                for u in adj[v]:
                    if sol[u] is not None:
                        used.add(sol[u])
                for c in lists[v]:
                    if c not in used:
                        sol[v] = c
                        break
                if sol[v] is None:
                    sol[v] = lists[v][0]
        
        out.append([int(c) for c in sol])
    
    return out
