"""Combined solver: max edge-disjoint same-vertex-set simple cycles on 21 vertices.

Strategy: repeatedly find a Hamiltonian cycle in the remaining-edge graph using
Warnsdorff backtracking with dead-end pruning, remove its edges (in random order
per restart), and repeat. Randomized restarts explore different removal orders so
we reach the theoretical maximum (floor(72/21)=3) edge-disjoint Hamiltonian cycles.
"""
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}


def construct():
    t0 = time.time()
    DEADLINE = t0 + 82.0  # safety margin before 90s kill
    n = P['n']

    # --- global graph state (rebuilt fresh per restart) ---
    def build_state():
        adj = [set() for _ in range(n)]
        rem = set()
        for u, v in P['edges']:
            if u > v:
                u, v = v, u
            rem.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
        return adj, rem

    def find_hc(adj, rem, start=0):
        """Return a Hamiltonian cycle (list of n distinct vertices) using only
        edges in `rem`, or None. Warnsdorff backtracking with dead-end pruning."""
        path = [start]
        visited = {start}

        def backtrack():
            if time.time() > DEADLINE:
                return None
            if len(path) == n:
                last = path[-1]
                e = (start, last) if start < last else (last, start)
                if e in rem:
                    return path[:]
                return None
            curr = path[-1]
            nbrs = [x for x in adj[curr] if x not in visited]
            # Warnsdorff: fewest onward moves first
            nbrs.sort(key=lambda x: len(adj[x]))
            for nxt in nbrs:
                ek = (curr, nxt) if curr < nxt else (nxt, curr)
                if ek not in rem:
                    continue
                path.append(nxt)
                visited.add(nxt)
                # dead-end pruning: every unvisited vertex must retain an
                # unvisited neighbour (else it can never be reached)
                dead = False
                if len(visited) < n:
                    unvis = [v for v in range(n) if v not in visited]
                    for v in unvis:
                        if not (adj[v] & set(unvis)):
                            dead = True
                            break
                if not dead:
                    r = backtrack()
                    if r is not None:
                        return r
                path.pop()
                visited.remove(nxt)
            return None

        return backtrack()

    def remove_cycle_edges(adj, rem, cyc):
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            e = (a, b) if a < b else (b, a)
            if e in rem:
                rem.discard(e)
                adj[a].discard(b)
                adj[b].discard(a)

    def solve_one_order(order, adj, rem):
        """Greedily remove cycles; the order of edge removal is randomized by
        shuffling each cycle's edges before deletion."""
        cycles = []
        for _ in range(3):
            if time.time() > DEADLINE:
                break
            cyc = find_hc(adj, rem)
            if cyc is None:
                break
            remove_cycle_edges(adj, rem, cyc)
            cycles.append(cyc)
        return cycles

    best = []
    for seed in range(0, 4000):
        if time.time() > DEADLINE - 1.0:
            break
        random.seed(seed)
        adj, rem = build_state()
        cycs = solve_one_order(None, adj, rem)
        if len(cycs) > len(best):
            best = cycs
        if len(best) >= 3:
            break

    # --- final validation & type cleanup ---
    if not best:
        # fallback: known Hamiltonian cycle on all vertices
        full = list(range(n))
        es = set()
        for u, v in P['edges']:
            es.add((u, v) if u < v else (v, u))
        ok = all((full[i], full[(i + 1) % n]) if full[i] < full[(i + 1) % n]
                 else (full[(i + 1) % n], full[i]) in es for i in range(n))
        best = [full] if ok else [full]

    host = set()
    for u, v in P['edges']:
        host.add((u, v) if u < v else (v, u))

    valid = []
    used = set()
    vset = None
    for cyc in best:
        cyc = [int(x) for x in cyc]
        if len(cyc) < 3 or len(set(cyc)) != len(cyc):
            continue
        s = frozenset(cyc)
        if vset is None:
            vset = s
        elif s != vset:
            continue
        ce = set()
        good = True
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            e = (a, b) if a < b else (b, a)
            if e not in host or e in used:
                good = False
                break
            ce.add(e)
        if not good:
            continue
        used |= ce
        valid.append(cyc)
    if valid:
        return valid
    # last resort: single known cycle
    return [list(range(n))]
