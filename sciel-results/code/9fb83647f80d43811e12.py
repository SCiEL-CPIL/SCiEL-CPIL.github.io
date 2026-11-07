"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
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

    def solve_csp(n, edges, lists, time_limit=20.0):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert to sets for faster lookup
        adj_set = [set(adj[i]) for i in range(n)]
        
        colour = [-1] * n
        # domains[v] = set of available colors for vertex v
        domains = [set(lists[v]) for v in range(n)]
        
        start_time = time.time()
        nodes_explored = 0
        
        def is_timed_out():
            return time.time() - start_time > time_limit
        
        def backtrack():
            nonlocal nodes_explored
            if is_timed_out():
                return 'timeout'
            
            # Find uncolored vertex with minimum remaining values (MRV)
            best_v = -1
            best_count = n + 1
            for v in range(n):
                if colour[v] != -1:
                    continue
                count = len(domains[v])
                if count == 0:
                    return False
                if count < best_count:
                    best_count = count
                    best_v = v
                    if count == 1:
                        break
            
            if best_v == -1:
                return True  # All colored
            
            nodes_explored += 1
            
            # Try each color for best_v
            for c in sorted(domains[best_v]):
                colour[best_v] = c
                domains[best_v].discard(c)
                
                # Forward checking: remove c from neighbors' domains
                undo = []
                valid = True
                for u in adj_set[best_v]:
                    if colour[u] != -1:
                        if colour[u] == c:
                            valid = False
                            break
                        continue
                    if c in domains[u]:
                        domains[u].discard(c)
                        undo.append((u, c))
                        if len(domains[u]) == 0:
                            valid = False
                            break
                
                if valid:
                    result = backtrack()
                    if result is True:
                        return True
                    if result == 'timeout':
                        # Undo and return timeout
                        for (u, cc) in undo:
                            domains[u].add(cc)
                        colour[best_v] = -1
                        domains[best_v].add(c)
                        return 'timeout'
                
                # Undo
                for (u, cc) in undo:
                    domains[u].add(cc)
                colour[best_v] = -1
                domains[best_v].add(c)
            
            return False
        
        result = backtrack()
        if result is True:
            # Verify the solution
            for v in range(n):
                if colour[v] not in lists[v]:
                    return None
                for u in adj_set[v]:
                    if u > v and colour[u] == colour[v]:
                        return None
            return [int(colour[v]) for v in range(n)]
        return None

    def greedy_color(n, edges, lists):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        colour = [-1] * n
        # Order by descending degree
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        
        for v in order:
            taken = set()
            for u in adj[v]:
                if colour[u] != -1:
                    taken.add(colour[u])
            free = [c for c in lists[v] if c not in taken]
            if free:
                colour[v] = int(free[0])
            else:
                colour[v] = int(lists[v][0])
        
        return colour

    out = []
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 5.0:
            # Not enough time, use greedy for remaining
            n, edges, lists = instance(splits, seed)
            colour = greedy_color(n, edges, lists)
            out.append(colour)
            continue
        
        n, edges, lists = instance(splits, seed)
        
        # Try CSP solver with time limit
        remaining = deadline - time.time() - 2.0
        time_limit = max(1.0, min(20.0, remaining / 2.0))
        
        colour = solve_csp(n, edges, lists, time_limit=time_limit)
        
        if colour is None:
            # Fallback to greedy
            colour = greedy_color(n, edges, lists)
        
        # Ensure all values are Python int
        colour = [int(c) for c in colour]
        out.append(colour)
    
    # Final type check and conversion
    result = []
    for colour in out:
        result.append([int(c) for c in colour])
    
    return result
# EVOLVE-BLOCK-END
