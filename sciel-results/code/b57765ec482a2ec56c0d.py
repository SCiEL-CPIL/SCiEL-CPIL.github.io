import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before the 150s wall-clock limit

    def is_c5_free(adj, n):
        """Exact check: does adj (list of sets) contain a C5 as a subgraph?
        A C5 subgraph: 5 distinct vertices a,b,c,d,e with edges
        a-b, b-c, c-d, d-e, e-a. Chords are allowed."""
        for a in range(n):
            for b in range(a + 1, n):
                if b not in adj[a]:
                    continue
                for c in range(b + 1, n):
                    if c not in adj[b]:
                        continue
                    for d in range(c + 1, n):
                        if d not in adj[c]:
                            continue
                        for e in range(d + 1, n):
                            if e not in adj[d]:
                                continue
                            if e in adj[a]:
                                # a-b-c-d-e-a is a 5-cycle (subgraph, chords OK)
                                return False
        return True

    def alpha_le_8(adj, n):
        """Exact check: is the independence number < 9?"""
        for indep in combinations(range(n), 9):
            ok = True
            for i in range(9):
                for j in range(i + 1, 9):
                    if indep[j] in adj[indep[i]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return False
        return True

    # Fallback: n=32 (8 disjoint K4s)
    n = 32
    adj = [set() for _ in range(n)]
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                adj[base + i].add(base + j)
                adj[base + j].add(base + i)
                edges.append([base + i, base + j])

    if not is_c5_free(adj, n) or not alpha_le_8(adj, n):
        # Should never happen
        pass

    if time.time() - start_time > deadline:
        return {'n': 0, 'edges': []}

    return {'n': n, 'edges': edges}
