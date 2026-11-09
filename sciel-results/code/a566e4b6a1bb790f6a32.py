"""Rewritten program using optimized MAC (Maintaining Arc Consistency) and Trail-Based Backtracking with Bitmasks."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Deadline guard: 120s limit, leave 5s margin
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0
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

    def solve_instance(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Bitmask domains: bit c is 1 if color c is in domain
        doms = []
        for l in lists:
            mask = 0
            for c in l:
                mask |= (1 << c)
            doms.append(mask)
        
        if any(d == 0 for d in doms):
            return None

        # Trail for undoing: stores (var, old_mask)
        trail = []

        def ac3(initial_queue=None):
            if initial_queue is None:
                queue = [(u, v) for u in range(n) for v in adj[u]]
            else:
                queue = list(initial_queue)
            
            head = 0
            while head < len(queue):
                i, j = queue[head]
                head += 1
                
                dj = doms[j]
                # If dom[j] is a singleton, remove that color from dom[i]
                if dj and (dj & (dj - 1)) == 0:
                    if doms[i] & dj:
                        old_mask = doms[i]
                        new_mask = doms[i] & ~dj
                        if new_mask != old_mask:
                            doms[i] = new_mask
                            trail.append((i, old_mask))
                            if new_mask == 0:
                                return False
                            # Propagate on any domain reduction for stronger pruning
                            for k in adj[i]:
                                if k != j:
                                    queue.append((k, i))
            return True

        # Initial AC-3
        mark = len(trail)
        if not ac3():
            return None
        
        assignment = [-1] * n
        # Precompute degrees for tie-breaking in MRV
        degrees = [len(adj[i]) for i in range(n)]

        def backtrack():
            if time.time() > DEADLINE:
                raise TimeoutError()
            
            # Find MRV node with Degree Heuristic tie-breaker
            min_size = 6 # Max possible size is 5
            max_degree = -1
            min_node = -1
            
            for i in range(n):
                if assignment[i] == -1:
                    size = doms[i].bit_count()
                    if size == 0:
                        return False
                    if size < min_size or (size == min_size and degrees[i] > max_degree):
                        min_size = size
                        max_degree = degrees[i]
                        min_node = i
                        if min_size == 1:
                            break

            if min_node == -1:
                return True # Success

            # Get values from domain in natural order (sufficient for 3-color lists)
            mask = doms[min_node]
            vals = []
            c = 0
            while mask:
                if mask & 1:
                    vals.append(c)
                mask >>= 1
                c += 1

            for c in vals:
                bit = 1 << c
                
                # Check consistency with assigned neighbors
                valid = True
                for nb in adj[min_node]:
                    if assignment[nb] != -1:
                        if assignment[nb] == c:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Save state
                mark = len(trail)
                
                # Assign c to min_node
                assignment[min_node] = c
                # Restrict doms[min_node] to {c}
                old_mask = doms[min_node]
                doms[min_node] = bit
                trail.append((min_node, old_mask))
                
                # Propagate: Run AC-3 on neighbors
                queue = [(nb, min_node) for nb in adj[min_node] if assignment[nb] == -1]
                
                if not ac3(queue):
                    # Conflict, undo and try next value
                    assignment[min_node] = -1
                    while len(trail) > mark:
                        var, old_mask = trail.pop()
                        doms[var] = old_mask
                    continue
                
                # Recurse
                if backtrack():
                    return True
                
                # Undo
                assignment[min_node] = -1
                while len(trail) > mark:
                    var, old_mask = trail.pop()
                    doms[var] = old_mask
            
            return False

        try:
            if backtrack():
                return [int(assignment[i]) for i in range(n)]
        except TimeoutError:
            pass
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            n, _, _ = instance(splits, seed)
            out.append([0] * n)
            continue
            
        n, edges, lists = instance(splits, seed)
        
        if time.time() > DEADLINE:
            out.append([0] * n)
            continue
            
        sol = solve_instance(n, edges, lists)
        
        if sol is None:
            out.append([0] * n)
        else:
            out.append([int(c) for c in sol])
            
    return out
# EVOLVE-BLOCK-END
