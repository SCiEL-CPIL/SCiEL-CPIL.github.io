"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

def construct():
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

    def solve_one(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Approach 1: Min-Conflicts Local Search
        def min_conflicts():
            rng = Random(n) # Deterministic seed based on n
            max_restarts = 20
            max_iters = 2000
            
            for _ in range(max_restarts):
                # Initialize
                coloring = [rng.choice(lst) for lst in lists]
                
                for _ in range(max_iters):
                    # Find conflicted vertices
                    conflicted = set()
                    for u, v in edges:
                        if coloring[u] == coloring[v]:
                            conflicted.add(u)
                            conflicted.add(v)
                    
                    if not conflicted:
                        return coloring
                    
                    # Pick a random conflicted vertex
                    v = rng.choice(list(conflicted))
                    
                    # Try all colors in lists[v] and pick the one with min conflicts
                    best_color = None
                    min_conflicts_count = float('inf')
                    
                    for c in lists[v]:
                        # Count conflicts if v takes color c
                        count = 0
                        for u in adj[v]:
                            if coloring[u] == c:
                                count += 1
                        if count < min_conflicts_count:
                            min_conflicts_count = count
                            best_color = c
                            if count == 0:
                                break # Can't do better than 0
                    
                    coloring[v] = best_color
            
            return None

        result = min_conflicts()
        if result is not None:
            return result

        # Approach 2: Fallback to Greedy + Backtracking (Original Logic)
        # This ensures we don't fail if Min-Conflicts gets stuck.
        
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        
        def greedy_color():
            colour_g = [-1] * n
            for v in order:
                taken = set()
                for u in adj[v]:
                    if colour_g[u] != -1:
                        taken.add(colour_g[u])
                free = [c for c in lists[v] if c not in taken]
                if free:
                    colour_g[v] = free[0]
                else:
                    return None
            return colour_g

        result = greedy_color()
        if result is not None:
            return result

        # Backtracking
        sys.setrecursionlimit(10000)
        colour = [-1] * n

        def backtrack(idx):
            if idx == n:
                return True

            v = order[idx]

            for c in lists[v]:
                valid = True
                for u in adj[v]:
                    if colour[u] == c:
                        valid = False
                        break

                if not valid:
                    continue

                colour[v] = c
                if backtrack(idx + 1):
                    return True
                colour[v] = -1

            return False

        if backtrack(0):
            return colour

        # Last resort
        return greedy_color()

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        colouring = solve_one(n, edges, lists)
        # Ensure all values are Python ints
        out.append([int(c) for c in colouring])
    return out
# EVOLVE-BLOCK-END
