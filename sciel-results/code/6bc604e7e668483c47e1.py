"""Rewritten program using backtracking CSP solver for List Coloring."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Deadline guard: 120s limit, leave 10s margin
    START_TIME = time.time()
    DEADLINE = START_TIME + 110.0

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

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert to sets for faster lookup
        adj_sets = [set(a) for a in adj]
        
        # Get available colors for each vertex
        available = [list(l) for l in lists]
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        
        # Color assignment array
        color = [-1] * n
        
        # Backtracking with forward checking
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            
            # Try each available color for vertex v
            for c in available[v]:
                # Check if color c is valid (not used by any already-colored neighbor)
                valid = True
                for u in adj[v]:
                    if color[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign color
                color[v] = c
                
                # Check if this assignment makes any uncolored neighbor's domain empty
                consistent = True
                for u in adj[v]:
                    if color[u] == -1:
                        # Check if u still has at least one available color
                        has_color = False
                        for cu in available[u]:
                            if cu != c:
                                has_color = True
                                break
                        if not has_color:
                            consistent = False
                            break
                
                if consistent:
                    if backtrack(idx + 1):
                        return True
                
                # Undo assignment
                color[v] = -1
            
            return False
        
        if backtrack(0):
            return color
        else:
            # Fallback: should not happen by Alon-Tarsi, but return something
            return [l[0] for l in lists]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        # Check deadline
        if time.time() > DEADLINE:
            # Return best effort - use greedy as fallback
            n, edges, lists = instance(splits, seed)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append([int(c) for c in colour])
            continue
        
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        out.append([int(c) for c in colouring])
    
    return out
# EVOLVE-BLOCK-END
