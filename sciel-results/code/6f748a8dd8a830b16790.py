import sys
import time

def construct():
    # Increase recursion limit for backtracking
    sys.setrecursionlimit(10000)
    
    def instance(splits, seed, palette=5):
        from random import Random
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

    def solve(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial available colors
        avail = [set(lst) for lst in lists]
        
        # To speed up, we can pre-compute a static ordering or use dynamic MRV.
        # Dynamic MRV is better.
        
        # We use a recursive backtracking function.
        # To avoid deep copying of 'avail' which is a list of sets, 
        # we will modify and restore.
        
        def backtrack():
            # Find vertex with minimum remaining values
            min_size = n + 1
            v = -1
            for i in range(n):
                if len(avail[i]) < min_size:
                    min_size = len(avail[i])
                    v = i
                    if min_size == 0:
                        return False
                    if min_size == 1:
                        break # Can't get better than 1
            
            if v == -1:
                return True # All vertices processed? 
                # Wait, if n=0? No, n>=8.
                # Actually, if all are assigned, avail would be empty? 
                # No, we don't clear avail[v] when assigned in this simple model unless we track assignment.
                # Let's track assignment explicitly.
                
        # Better approach: Track assigned vertices.
        # Or, simpler: since we are coloring all vertices, we can just check if all are assigned.
        # But 'avail' shrinks. If we don't remove the assigned color from 'avail[v]' itself, 
        # we need a separate 'assigned' array.
        
        assigned = [None] * n
        # avail[i] contains colors NOT used by neighbors.
        # Initially avail[i] is the list.
        # When we assign color c to v, we remove c from avail[u] for all u in adj[v].
        # We do NOT remove c from avail[v] in the main loop logic if we check 'assigned'.
        
        def backtrack_v2():
            # Find unassigned vertex with min avail
            min_size = n + 1
            v = -1
            for i in range(n):
                if assigned[i] is None:
                    sz = len(avail[i])
                    if sz < min_size:
                        min_size = sz
                        v = i
                        if sz == 0:
                            return False
                        if sz == 1:
                            break
            
            if v == -1:
                return True # All assigned
            
            # Try each color in avail[v]
            # To make it deterministic and potentially faster, sort colors?
            # Doesn't matter much.
            for c in list(avail[v]):
                assigned[v] = c
                # Propagate
                valid = True
                changed = []
                for u in adj[v]:
                    if c in avail[u]:
                        avail[u].remove(c)
                        changed.append(u)
                        if len(avail[u]) == 0:
                            valid = False
                            break
                
                if valid and backtrack_v2():
                    return True
                
                # Undo
                for u in changed:
                    avail[u].add(c)
                assigned[v] = None
            
            return False
        
        if backtrack_v2():
            return assigned
        else:
            return None

    out = []
    start_time = time.time()
    deadline = start_time + 100 # Safety margin
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we run out of time, we can't really do much better than what we have
            # But for this small problem, it won't happen.
            break
            
        n, edges, lists = instance(splits, seed)
        sol = solve(n, edges, lists)
        
        if sol is None:
            # Should not happen per Alon-Tarsi, but if it does, return something valid-ish?
            # The task says "An instance whose colouring breaks a list or an edge scores 0".
            # If we can't solve it, we might return a partial or invalid one.
            # But we expect to solve it.
            sol = [0] * n # Fallback, will score 0 for this instance
        
        # Ensure types are Python int
        out.append([int(c) for c in sol])
        
    return out
