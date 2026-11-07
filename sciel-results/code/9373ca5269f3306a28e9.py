"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    _start_time = time.time()
    _deadline = _start_time + 140.0  # 140s safety margin before 150s kill

    from random import Random

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

    def solve_coloring(n, edges, lists):
        """Backtracking with MRV + forward checking for list coloring."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Convert lists to sets for fast membership
        avail = [set(lst) for lst in lists]
        
        # Color assignment: -1 means uncolored
        color = [-1] * n
        
        # We'll use a simple recursive backtracker with MRV
        # For efficiency on small graphs, this will be very fast
        
        import sys
        sys.setrecursionlimit(10000)
        
        def backtrack():
            # Find uncolored vertex with minimum remaining values (MRV)
            best_v = -1
            best_count = float('inf')
            for v in range(n):
                if color[v] == -1:
                    cnt = len(avail[v])
                    if cnt < best_count:
                        best_count = cnt
                        best_v = v
                        if cnt == 0:
                            return False  # Dead end
                        if cnt == 1:
                            break  # Can't do better than 1
            
            if best_v == -1:
                return True  # All colored
            
            v = best_v
            # Try each available color
            for c in sorted(avail[v]):
                # Assign
                color[v] = c
                avail[v].discard(c)
                
                # Forward checking: remove c from neighbors
                undo = []
                feasible = True
                for nb in adj[v]:
                    if color[nb] == -1:
                        if c in avail[nb]:
                            avail[nb].discard(c)
                            undo.append((nb, c))
                            if len(avail[nb]) == 0:
                                feasible = False
                                break
                
                if feasible:
                    if backtrack():
                        return True
                
                # Undo
                for nb, uc in undo:
                    avail[nb].add(uc)
                avail[v].add(c)
                color[v] = -1
            
            return False
        
        if backtrack():
            return [int(color[v]) for v in range(n)]
        else:
            return None  # Should not happen per Alon-Tarsi, but safety

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > _deadline:
            # Not enough time; append a dummy (will score 0 for remaining)
            # But we need valid structure; use a placeholder that will fail
            out.append([0] * 0)  # This will be invalid; better to try quickly
            continue
        
        n, edges, lists = instance(splits, seed)
        result = solve_coloring(n, edges, lists)
        
        if result is None or len(result) != n:
            # Fallback: greedy as last resort
            result = [lists[v][0] for v in range(n)]
        
        # Ensure Python int
        result = [int(c) for c in result]
        out.append(result)
    
    return out
# EVOLVE-BLOCK-END
