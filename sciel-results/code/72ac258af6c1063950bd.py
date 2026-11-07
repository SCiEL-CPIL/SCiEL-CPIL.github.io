"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Set recursion limit high enough for the largest instance (n=408)
    sys.setrecursionlimit(10000)
    
    _start = time.time()
    _deadline = _start + 140  # Safety margin before 150s kill

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

    def is_valid(colour, n, adj, lists):
        for v in range(n):
            c = colour[v]
            if c is None or c not in lists[v]:
                return False
            for u in adj[v]:
                if colour[u] == c:
                    return False
        return True

    def backtrack_dynamic(n, adj, lists, colour, unassigned, list_masks, deadline):
        if time.time() > deadline:
            return False
        if not unassigned:
            return True
        
        # Find vertex with Minimum Remaining Values (MRV)
        min_count = 5
        min_v = -1
        for v in unassigned:
            used_mask = 0
            for u in adj[v]:
                if colour[u] is not None:
                    used_mask |= (1 << colour[u])
            avail = list_masks[v] & (~used_mask) & 0x1F
            count = bin(avail).count('1')
            if count < min_count:
                min_count = count
                min_v = v
                if count == 0:
                    return False
                if count == 1:
                    break
        
        if min_v == -1:
            return False
            
        used_mask = 0
        for u in adj[min_v]:
            if colour[u] is not None:
                used_mask |= (1 << colour[u])
        avail = list_masks[min_v] & (~used_mask) & 0x1F
        
        unassigned.remove(min_v)
        
        # Try colors
        for c in range(5):
            if avail & (1 << c):
                colour[min_v] = c
                if backtrack_dynamic(n, adj, lists, colour, unassigned, list_masks, deadline):
                    return True
                colour[min_v] = None
        
        unassigned.add(min_v)
        return False

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        # Check time
        if time.time() > _deadline - 2:
            # Fallback: generate instance and return a dummy valid-looking structure
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue

        n, edges, lists = instance(splits, seed)
        adj = [list(set()) for _ in range(n)]
        for u, v in edges:
            if v not in adj[u]:
                adj[u].append(v)
            if u not in adj[v]:
                adj[v].append(u)

        # Precompute list masks
        list_masks = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            list_masks[v] = mask

        # 1. Try Greedy (Degree Descending)
        colour = [None] * n
        order_greedy = sorted(range(n), key=lambda u: -len(adj[u]))
        greedy_success = True
        for v in order_greedy:
            taken_mask = 0
            for u in adj[v]:
                cu = colour[u]
                if cu is not None:
                    taken_mask |= (1 << cu)
            
            avail_mask = list_masks[v] & (~taken_mask) & 0x1F
            if avail_mask:
                # Pick the first available color
                c = (avail_mask & -avail_mask).bit_length() - 1
                colour[v] = c
            else:
                greedy_success = False
                break
        
        if greedy_success and is_valid(colour, n, adj, lists):
            out.append(colour)
            continue

        # 2. Backtracking with Dynamic MRV
        colour_bt = [None] * n
        unassigned = set(range(n))
        
        bt_deadline = _deadline - 0.5
        if backtrack_dynamic(n, adj, lists, colour_bt, unassigned, list_masks, bt_deadline):
            out.append(colour_bt)
            continue

        # Fallback: Return the greedy attempt (might be invalid, scores 0 for this instance)
        if not greedy_success:
            for v in order_greedy:
                if colour[v] is None:
                    colour[v] = lists[v][0]
        out.append(colour)

        # 5. Fallback: Return the greedy attempt (might be invalid, scores 0 for this instance)
        if not greedy_success:
            for v in order_greedy:
                if colour[v] is None:
                    colour[v] = lists[v][0]
        out.append(colour)

    # Ensure all are Python ints
    for i in range(len(out)):
        out[i] = [int(c) for c in out[i]]
    return out
# EVOLVE-BLOCK-END
