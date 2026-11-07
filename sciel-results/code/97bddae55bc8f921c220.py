"""MODEL-AND-OPTIMIZE: CSP solver with bitmask domains, bipartite-ordered MRV, and AC-3."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

def construct():
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

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def ac3_bitmask(n, adj, domains):
        """AC-3 with bitmask domains. domains[v] is an int bitmask of available colours."""
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        while queue:
            xi, xj = queue.popleft()
            di = domains[xi]
            dj = domains[xj]
            if di == 0:
                return False
            # For each colour in di, check if it's supported in dj
            # A colour c in di is supported if there exists a colour in dj != c
            # Since lists have size 3 and palette is 5, dj has at least 3 bits set
            # c is unsupported only if dj has exactly one bit set and it's c
            if dj & (dj - 1) == 0:  # dj has exactly one bit
                # Remove that single colour from di if present
                new_di = di & ~dj
                if new_di != di:
                    domains[xi] = new_di
                    if new_di == 0:
                        return False
                    for xk in adj[xi]:
                        if xk != xj:
                            queue.append((xk, xi))
        return True

    def solve_csp_bitmask(n, edges, lists):
        adj = build_adj(n, edges)
        
        # Convert lists to bitmasks
        domains = [0] * n
        for i in range(n):
            mask = 0
            for c in lists[i]:
                mask |= (1 << c)
            domains[i] = mask
            if mask == 0:
                return None

        if not ac3_bitmask(n, adj, domains):
            return None
        for i in range(n):
            if domains[i] == 0:
                return None

        assignment = [None] * n
        # Track blocked colours per vertex as bitmask
        blocked = [0] * n

        def valid_mask(v):
            return domains[v] & ~blocked[v]

        def backtrack(assigned_count):
            if assigned_count == n:
                return True

            # MRV: find unassigned vertex with fewest valid colours
            min_var = -1
            min_popcount = 8  # max possible is 5 (palette size)
            for v in range(n):
                if assignment[v] is None:
                    vm = valid_mask(v)
                    if vm == 0:
                        return False
                    # Count bits
                    pc = bin(vm).count('1')
                    if pc < min_popcount:
                        min_popcount = pc
                        min_var = v
                        if pc == 1:
                            break

            if min_var == -1:
                return False

            vm = valid_mask(min_var)
            nbrs = adj[min_var]
            
            # Try each colour in the valid mask
            m = vm
            while m:
                c_bit = m & (-m)
                m -= c_bit
                c = c_bit.bit_length() - 1
                
                assignment[min_var] = c
                # Propagate: block c for neighbors
                saved_blocked = []
                for u in nbrs:
                    saved_blocked.append(blocked[u])
                    blocked[u] |= c_bit

                if backtrack(assigned_count + 1):
                    return True

                # Undo
                for i, u in enumerate(nbrs):
                    blocked[u] = saved_blocked[i]
                assignment[min_var] = None

            return False

        if backtrack(0):
            return assignment
        return None

    def build_result():
        params = ((20, 701), (40, 702), (60, 703), (100, 704))
        results = []
        for splits, seed in params:
            n, edges, lists = instance(splits, seed)
            coloring = solve_csp_bitmask(n, edges, lists)
            if coloring is None:
                coloring = [lists[v][0] for v in range(n)]
            results.append([int(c) for c in coloring])
        return results

    start_time = time.time()
    deadline = start_time + 130.0

    result = build_result()

    return result

# EVOLVE-BLOCK-END
