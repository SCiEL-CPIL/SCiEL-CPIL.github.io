"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Increase recursion limit to be safe for backtracking depth
    sys.setrecursionlimit(5000)
    
    from random import Random
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for 120s limit

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

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to sets for faster lookup
        domains = [set(lst) for lst in lists]
        colour = [None] * n
        uncolored = set(range(n))

        def backtrack():
            if not uncolored:
                return True
            
            # MRV Heuristic: Pick vertex with fewest available colors
            min_count = 5
            target = -1
            for v in uncolored:
                available = len(domains[v])
                if available < min_count:
                    min_count = available
                    target = v
                    if available == 0:
                        return False
                    if available == 1:
                        break
            
            if target == -1:
                return False # Should not happen if uncolored is not empty

            v = target
            uncolored.remove(v)
            
            # Try each color in domain
            # Sort colors to try most constrained first? Or just iterate.
            # Since domains are small, order doesn't matter much.
            for c in list(domains[v]):
                # Check if color is valid (not used by neighbors)
                valid = True
                for u in adj[v]:
                    if colour[u] == c:
                        valid = False
                        break
                
                if valid:
                    colour[v] = c
                    
                    # Forward Checking: Check if any neighbor loses all colors
                    # If so, this branch is dead.
                    fc_valid = True
                    for u in adj[v]:
                        if u in uncolored and c in domains[u]:
                            # We don't actually remove from domain permanently in this simple version,
                            # we just check if it becomes empty.
                            # For efficiency, we can skip full propagation and rely on MRV.
                            pass 
                    
                    # Recursively solve
                    if backtrack():
                        return True
                    
                    # Undo
                    colour[v] = None
            
            # Restore vertex to uncolored set if all colors failed
            uncolored.add(v)
            return False

        if backtrack():
            return colour
        else:
            # Fallback: return a list of 0s (invalid coloring, scores 0 for this instance)
            # But must be valid type and length.
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we are running out of time, fill remaining with dummy solutions
            # to ensure we return a valid structure (list of 4 lists of ints)
            # though they will score 0.
            while len(out) < 4:
                 out.append([0])
            break
        
        n, edges, lists = instance(splits, seed)
        # Ensure lists are lists of ints
        sol = solve_instance(n, edges, lists)
        # Convert to Python ints just in case
        # Ensure they are plain ints, not numpy or bool
        out.append([int(c) for c in sol])
        
    # If we exited early due to deadline, ensure we have 4 entries
    while len(out) < 4:
        out.append([0])
        
    return out
# EVOLVE-BLOCK-END
