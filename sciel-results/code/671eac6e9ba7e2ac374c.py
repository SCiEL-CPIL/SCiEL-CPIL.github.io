"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

_ADJ = None
_EDGE_SET = None

def _init():
    global _ADJ, _EDGE_SET
    if _ADJ is not None:
        return
    n = P['n']
    adj = defaultdict(list)
    eset = set()
    for a, b in P['edges']:
        adj[a].append(b)
        adj[b].append(a)
        eset.add((min(a,b), max(a,b)))
    _ADJ = adj
    _EDGE_SET = eset

def _find_hamiltonian_cycle(used_edges, deadline, start=None, order_hint=None):
    """Find a Hamiltonian cycle avoiding used_edges."""
    n = P['n']
    adj = _ADJ
    if start is None:
        start = 0
    
    path = [start]
    path_set = {start}
    
    def dfs():
        if time.time() > deadline:
            return None
        if len(path) == n:
            last = path[-1]
            first = path[0]
            e = (min(last, first), max(last, first))
            if e not in used_edges:
                return list(path)
            return None
        cur = path[-1]
        cands = []
        for nb in adj[cur]:
            if nb not in path_set:
                e = (min(cur, nb), max(cur, nb))
                if e not in used_edges:
                    cands.append(nb)
        if not cands:
            return None
        # Heuristic: prefer vertices with fewer remaining unvisited neighbors
        def score(nb):
            cnt = 0
            for nn in adj[nb]:
                if nn not in path_set:
                    e = (min(nb, nn), max(nb, nn))
                    if e not in used_edges:
                        cnt += 1
            return cnt
        cands.sort(key=score)
        for nb in cands:
            path.append(nb)
            path_set.add(nb)
            res = dfs()
            if res is not None:
                return res
            path.pop()
            path_set.discard(nb)
        return None
    
    return dfs()

def construct():
    _init()
    deadline = time.time() + 75
    used_edges = set()
    cycles = []
    
    # Try to find up to 3 Hamiltonian cycles
    for attempt in range(3):
        if time.time() > deadline:
            break
        cyc = _find_hamiltonian_cycle(used_edges, deadline)
        if cyc is None:
            break
        # Verify
        valid = True
        seen = set()
        for i in range(len(cyc)):
            a = cyc[i]
            b = cyc[(i+1) % len(cyc)]
            e = (min(a,b), max(a,b))
            if e not in _EDGE_SET or e in used_edges:
                valid = False
                break
            seen.add(a)
        if valid and len(seen) == P['n']:
            cycles.append(cyc)
            for i in range(len(cyc)):
                a = cyc[i]
                b = cyc[(i+1) % len(cyc)]
                used_edges.add((min(a,b), max(a,b)))
        else:
            break
    
    if not cycles:
        # Fallback
        full = list(range(P['n']))
        valid = True
        for i in range(len(full)):
            a = full[i]
            b = full[(i+1) % len(full)]
            e = (min(a,b), max(a,b))
            if e not in _EDGE_SET:
                valid = False
                break
        if valid:
            cycles.append(full)
    
    result = [[int(v) for v in c] for c in cycles]
    return result
# EVOLVE-BLOCK-END
