"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
from random import Random
from collections import deque


def construct():
    sys.setrecursionlimit(100000)
    start = time.time()
    TOTAL_BUDGET = 145.0  # explicit guard, safety margin below the 150s kill

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

    def _solve(n, edges, lists, time_limit):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        degree = [len(adj[i]) for i in range(n)]

        # Initialize domains: list of available colours per vertex
        domains = [list(lists[i]) for i in range(n)]
        color = [-1] * n
        uncolored = set(range(n))
        deadline = time.time() + time_limit
        state = {'count': 0, 'ok': False}

        def is_deadline():
            return time.time() > deadline

        def revise(i, j):
            """Check if domain[i] can be reduced given domain[j]."""
            revised = False
            di = domains[i]
            dj = domains[j]
            new_di = []
            for c in di:
                # Check if c has support in dj (i.e., exists c' in dj such that c != c')
                # Since constraint is c != c', c is supported if there is ANY c' in dj where c' != c.
                # This is equivalent to: if len(dj) > 1, any c is supported (unless dj is singleton {c}).
                # If len(dj) == 1, c is supported only if c != dj[0].
                if len(dj) == 1:
                    if c == dj[0]:
                        continue
                # If len(dj) > 1, c is always supported because there is at least one color in dj different from c.
                # Wait, if dj = {1, 2} and c = 1, support is 2. If c = 3, support is 1 or 2.
                # So yes, if len(dj) > 1, all c in di are supported.
                # Optimization: only check if len(dj) == 1.
                new_di.append(c)
            
            if len(new_di) < len(di):
                domains[i] = new_di
                revised = True
            return revised

        def ac3(initial_queue):
            """Run AC-3. Returns False if a domain becomes empty."""
            queue = deque(initial_queue)
            while queue:
                i, j = queue.popleft()
                if revise(i, j):
                    if not domains[i]:
                        return False
                    for k in adj[i]:
                        if k != j and k in uncolored:
                            queue.append((k, i))
            return True

        def backtrack():
            if not uncolored:
                state['ok'] = True
                return True

            state['count'] += 1
            if (state['count'] & 255) == 0 and is_deadline():
                return False

            # Select variable with smallest domain (MRV), tie-break by degree
            best_v = -1
            best_len = 999
            best_deg = -1
            for v in uncolored:
                ln = len(domains[v])
                if ln == 0:
                    return False
                if ln < best_len or (ln == best_len and degree[v] > best_deg):
                    best_len = ln
                    best_v = v
                    best_deg = degree[v]
                    if ln == 1:
                        break

            if best_v == -1:
                return False

            # Order values: Least Constraining Value (LCV)
            # Prefer values that rule out the fewest choices in neighbors
            def value_score(c):
                cnt = 0
                for u in adj[best_v]:
                    if u in uncolored and c in domains[u]:
                        cnt += 1
                return cnt

            values = sorted(domains[best_v], key=value_score)

            uncolored.discard(best_v)

            # Save state for undo
            old_domains = {v: domains[v][:] for v in uncolored if v in adj[best_v]}
            old_domain_v = domains[best_v][:]

            for c in values:
                # Assign c to best_v
                domains[best_v] = [c]
                color[best_v] = c
                
                # Propagate constraints
                # Create initial queue of arcs (u, best_v) for all neighbors u
                init_queue = [(u, best_v) for u in adj[best_v] if u in uncolored]
                
                # Run AC-3
                if ac3(init_queue):
                    if backtrack():
                        return True
                else:
                    # Conflict, undo and try next value
                    pass

                # Undo
                color[best_v] = -1
                domains[best_v] = old_domain_v
                for v, old_d in old_domains.items():
                    domains[v] = old_d

            uncolored.add(best_v)
            return False

        # Initial check: any empty domain?
        for i in range(n):
            if not domains[i]:
                return None

        # Initial AC-3 pass to prune domains before backtracking
        init_queue = []
        for u in range(n):
            for v in adj[u]:
                init_queue.append((u, v))
        
        if not ac3(init_queue):
            return None

        if backtrack() and state['ok']:
            return [int(c) for c in color]
        return None

    def _greedy(n, adj, lists):
        order = sorted(range(n), key=lambda v: (-len(adj[v]), v))
        color = [-1] * n
        for v in order:
            taken = set()
            for u in adj[v]:
                if color[u] != -1:
                    taken.add(color[u])
            c = -1
            for cc in lists[v]:
                if cc not in taken:
                    c = cc
                    break
            if c == -1:
                for cc in lists[v]:
                    c = cc
                    break
            color[v] = c
        return [int(c) for c in color]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        remaining = TOTAL_BUDGET - (time.time() - start)
        n, edges, lists = instance(splits, seed)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        solved = False
        if remaining > 2.0:
            # Allocate time proportionally, but ensure at least 0.5s for small instances
            # Larger instances get more time.
            # Splits: 20, 40, 60, 100 -> n: 88, 168, 248, 408
            # We'll just give a fixed chunk, say 20s each if available, else scale.
            tl = max(0.5, min(25.0, remaining * 0.4))
            res = _solve(n, edges, lists, tl)
            if res is not None:
                out.append(res)
                solved = True

        if not solved:
            # Fallback: best-effort greedy (may score 0 for a broken instance, but is structurally valid).
            out.append(_greedy(n, adj, lists))

    # Hardening: ensure top-level is a list of 4 lists of Python ints with correct lengths.
    # (A wrong length/type makes the whole result invalid, so guard it.)
    lengths = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, _, _ = instance(splits, seed)
        lengths.append(n)
    for i in range(len(out)):
        if not isinstance(out[i], list):
            out[i] = list(out[i])
        if len(out[i]) != lengths[i]:
            out[i] = (out[i] + [0] * lengths[i])[:lengths[i]]
        out[i] = [int(x) for x in out[i]]
    return out
# EVOLVE-BLOCK-END
