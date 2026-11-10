"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

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

    def solve_csp(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        adj_sets = [set(neigh) for neigh in adj]
        
        # Initial domains as sets
        domains = [set(lst) for lst in lists]
        
        # Proper AC-3: for each arc (i,j), remove values from domain[i] that have no support in domain[j]
        def ac3():
            queue = []
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            
            while queue:
                i, j = queue.pop()
                removed = False
                # For each value ci in domain[i], check if it has support in domain[j]
                # Support means there exists cj in domain[j] such that ci != cj
                # Since the constraint is "not equal", ci has support iff domain[j] has any value != ci
                # i.e., len(domain[j]) > 1 OR (len(domain[j]) == 1 and ci not in domain[j])
                for ci in list(domains[i]):
                    # ci is unsupported if domain[j] == {ci}
                    if len(domains[j]) == 1 and ci in domains[j]:
                        domains[i].discard(ci)
                        removed = True
                        if not domains[i]:
                            return False
                if removed:
                    for k in adj_sets[i]:
                        if k != j:
                            queue.append((k, i))
            return True
        
        if not ac3():
            return None
        
        # Backtracking with dynamic variable ordering (MRV) and forward checking
        def backtrack(assigned, current_domains):
            if time.time() > deadline:
                return None
            
            # Find unassigned variable with minimum remaining values (MRV)
            var = -1
            min_size = n + 1
            for v in range(n):
                if v not in assigned:
                    sz = len(current_domains[v])
                    if sz < min_size:
                        min_size = sz
                        var = v
                        if sz <= 1:
                            break
            
            if var == -1:
                return assigned  # All assigned
            
            # Try values for var
            for c in sorted(current_domains[var]):
                # Check if c is valid (no assigned neighbor has c)
                valid = True
                for u in adj[var]:
                    if u in assigned and assigned[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Forward checking: create new domains
                new_domains = [set(d) for d in current_domains]
                new_assigned = dict(assigned)
                new_assigned[var] = c
                
                ok = True
                for u in adj[var]:
                    if u not in new_assigned:
                        if c in new_domains[u]:
                            new_domains[u].discard(c)
                            if not new_domains[u]:
                                ok = False
                                break
                
                if not ok:
                    continue
                
                result = backtrack(new_assigned, new_domains)
                if result is not None:
                    return result
            
            return None
        
        result = backtrack({}, domains)
        if result is None:
            return None
        
        colour = [0] * n
        for v in range(n):
            colour[v] = int(result[v])
        return colour

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            while len(out) < 4:
                out.append([0] * 8)
            break
        
        n, edges, lists = instance(splits, seed)
        colour = solve_csp(n, edges, lists)
        
        if colour is None:
            # Fallback: greedy with random restarts
            colour = None
            for attempt in range(100):
                if time.time() > deadline:
                    break
                adj_local = [set() for _ in range(n)]
                for u, v in edges:
                    adj_local[u].add(v)
                    adj_local[v].add(u)
                
                colour_try = [None] * n
                order = list(range(n))
                Random(seed + attempt).shuffle(order)
                
                success = True
                for v in order:
                    taken = {colour_try[u] for u in adj_local[v] if colour_try[u] is not None}
                    free = [c for c in lists[v] if c not in taken]
                    if free:
                        colour_try[v] = free[0]
                    else:
                        success = False
                        break
                
                if success:
                    valid = True
                    for v in range(n):
                        if colour_try[v] not in lists[v]:
                            valid = False
                            break
                        for u in adj_local[v]:
                            if u < v and colour_try[u] == colour_try[v]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        colour = [int(c) for c in colour_try]
                        break
            
            if colour is None:
                colour = [0] * n
        
        out.append(colour)
    
    while len(out) < 4:
        out.append([0] * 8)
    
    return out
# EVOLVE-BLOCK-END
