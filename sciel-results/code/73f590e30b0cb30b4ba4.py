# EVOLVE-BLOCK-START
import time
import sys
from random import Random

sys.setrecursionlimit(10000)

def construct():
    t0 = time.time()
    LIMIT = 120.0          # hard wall-clock cap for the evaluator
    MARGIN = 10.0          # safety margin before the cap

    def deadline():
        return (time.time() - t0) > (LIMIT - MARGIN)

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

    def solve_coloring(n, adj, lists):
        # Bitmask representation for available colors (0-4)
        # bit i set means color i is available
        avail = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            avail[i] = mask
        
        color = [-1] * n
        colored = [False] * n
        num_colored = 0
        
        # Pre-compute neighbor lists for fast access
        # adj is already a list of lists
        
        def backtrack():
            nonlocal num_colored
            if num_colored == n:
                return True
            
            # MRV Heuristic: Pick uncolored vertex with fewest available colors
            min_avail_count = 5
            best_v = -1
            for i in range(n):
                if not colored[i]:
                    cnt = bin(avail[i]).count('1')
                    if cnt == 0:
                        return False # Dead end
                    if cnt < min_avail_count:
                        min_avail_count = cnt
                        best_v = i
            
            v = best_v
            # Try each available color for v
            mask = avail[v]
            while mask:
                # Get lowest set bit
                c_bit = mask & -mask
                c = c_bit.bit_length() - 1
                mask ^= c_bit
                
                # Assign color c to v
                color[v] = c
                colored[v] = True
                num_colored += 1
                
                # Check neighbors and update avail
                valid = True
                affected = []
                for nb in adj[v]:
                    if colored[nb]:
                        # If neighbor is already colored, check for conflict
                        if color[nb] == c:
                            valid = False
                            break
                    else:
                        # Neighbor is uncolored, remove c from its avail
                        if avail[nb] & c_bit:
                            avail[nb] ^= c_bit
                            if avail[nb] == 0:
                                valid = False
                                break
                            affected.append(nb)
                
                if valid:
                    if backtrack():
                        return True
                
                # Undo
                num_colored -= 1
                colored[v] = False
                color[v] = -1
                for nb in affected:
                    avail[nb] |= c_bit
        
        # Initial check
        for i in range(n):
            if avail[i] == 0:
                return None

        if backtrack():
            return color
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if deadline():
            break
        n, edges, lists = instance(splits, seed)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        col = solve_coloring(n, adj, lists)
        
        if col is None:
            # Fallback
            col = [lists[i][0] for i in range(n)]
            
        out.append([int(c) for c in col])

    return out
# EVOLVE-BLOCK-END
