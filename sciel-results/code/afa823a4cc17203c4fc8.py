"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    sys.setrecursionlimit(10000)
    
    _start = time.time()
    _deadline = _start + 130  # Safety margin before 150s kill

    from random import Random

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        # Cube faces
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
        """Solve list coloring for a single instance."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Convert to lists for speed
        adj_list = [list(s) for s in adj]
        
        # Precompute list masks
        list_masks = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            list_masks[v] = mask
        
        # Try greedy first (fast path)
        colour = [-1] * n
        order_greedy = sorted(range(n), key=lambda u: -len(adj_list[u]))
        greedy_ok = True
        for v in order_greedy:
            taken = 0
            for u in adj_list[v]:
                if colour[u] >= 0:
                    taken |= (1 << colour[u])
            avail = list_masks[v] & (~taken) & 0x1F
            if avail:
                c = (avail & -avail).bit_length() - 1
                colour[v] = c
            else:
                greedy_ok = False
                break
        
        if greedy_ok:
            # Verify
            for v in range(n):
                if colour[v] not in lists[v]:
                    greedy_ok = False
                    break
                for u in adj_list[v]:
                    if colour[u] == colour[v]:
                        greedy_ok = False
                        break
                if not greedy_ok:
                    break
            if greedy_ok:
                return colour
        
        # Backtracking with MRV + degree heuristic
        # Use iterative deepening or just straight backtracking with good ordering
        # For these small graphs, straight backtracking should work
        
        # Order: sort by (list_size, -degree) for MRV-like behavior
        order = sorted(range(n), key=lambda u: (len(lists[u]), -len(adj_list[u])))
        
        # Try multiple orderings
        orderings = [
            order,
            sorted(range(n), key=lambda u: -len(adj_list[u])),
            sorted(range(n), key=lambda u: (len(lists[u]), -len(adj_list[u]), u)),
        ]
        
        for ord in orderings:
            colour_bt = [-1] * n
            
            def backtrack(idx):
                if time.time() > _deadline:
                    return False
                if idx == n:
                    return True
                v = ord[idx]
                used = 0
                for u in adj_list[v]:
                    if colour_bt[u] >= 0:
                        used |= (1 << colour_bt[u])
                avail = list_masks[v] & (~used) & 0x1F
                if not avail:
                    return False
                # Try colors in order of least constraining
                while avail:
                    bit = avail & (-avail)
                    c = bit.bit_length() - 1
                    colour_bt[v] = c
                    if backtrack(idx + 1):
                        return True
                    colour_bt[v] = -1
                    avail ^= bit
                return False
            
            if backtrack(0):
                # Verify
                valid = True
                for v in range(n):
                    if colour_bt[v] not in lists[v]:
                        valid = False
                        break
                    for u in adj_list[v]:
                        if colour_bt[u] == colour_bt[v]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return colour_bt
        
        # Fallback: return greedy attempt or first available colors
        if not greedy_ok:
            for v in order_greedy:
                if colour[v] < 0:
                    colour[v] = lists[v][0]
        return colour

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > _deadline - 2:
            n, edges, lists = instance(splits, seed)
            out.append([lists[v][0] for v in range(n)])
            continue
        
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists)
        out.append(result)
    
    # Ensure all are Python ints
    for i in range(len(out)):
        out[i] = [int(c) for c in out[i]]
    return out
# EVOLVE-BLOCK-END
