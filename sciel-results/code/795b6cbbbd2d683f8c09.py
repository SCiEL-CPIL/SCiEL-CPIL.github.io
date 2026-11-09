"""Improved solver: Exact search for minimum cochromatic partition.

Uses a backtracking search with symmetry breaking to find the minimum number of parts.
For a fixed number of parts k, it attempts to assign all vertices to k parts such that
each part is a clique in G or a clique in G_bar (independent set in G).
Symmetry is broken by requiring that the first vertex assigned to part i is smaller than
the first vertex assigned to part j for i < j. This is achieved by only allowing a
vertex to start a new part if it is the next available part index.
"""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
DEADLINE = 85.0
_T0 = None

def _setup():
    global _T0
    _T0 = time.time()
    edges = P['edges']
    adj = [0] * N
    for a, b in edges:
        adj[a] |= 1 << b
        adj[b] |= 1 << a
    comp = [((1 << N) - 1) ^ (1 << v) ^ adj[v] for v in range(N)]
    return adj, comp

def _chromatic_number(adj):
    # Simple greedy for lower bound
    n = N
    order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
    color = [0] * n
    used = 0
    for v in order:
        banned = 0
        for u in range(n):
            if color[u] and (adj[v] >> u) & 1:
                banned |= 1 << color[u]
        c = 0
        while (banned >> c) & 1:
            c += 1
        color[v] = c
        if c >= used:
            used = c + 1
    return used

def _solve_exact(adj, comp):
    lb = max(_chromatic_number(adj), _chromatic_number(comp))
    
    for k in range(lb, N + 1):
        if time.time() > DEADLINE:
            break
        parts = [0] * k
        part_types = [-1] * k
        used_parts = 0
        
        def can_add(v, p):
            mask = parts[p]
            if mask == 0:
                return True
            if part_types[p] == -1:
                return True
            t = part_types[p]
            a = adj if t == 0 else comp
            for u in range(N):
                if (mask >> u) & 1:
                    if not ((a[v] >> u) & 1):
                        return False
            return True

        def dfs(v, used):
            if time.time() > DEADLINE:
                return False
            if v == N:
                return True
            
            for p in range(used):
                if can_add(v, p):
                    parts[p] |= (1 << v)
                    if part_types[p] == -1:
                        old_mask = parts[p] ^ (1 << v)
                        u = (old_mask & -old_mask).bit_length() - 1
                        if (adj[v] >> u) & 1:
                            part_types[p] = 0
                        else:
                            part_types[p] = 1
                        
                    if dfs(v + 1, used):
                        return True
                    
                    parts[p] ^= (1 << v)
                    if part_types[p] != -1 and bin(parts[p]).count('1') == 1:
                        part_types[p] = -1

            if used < k:
                p = used
                parts[p] = 1 << v
                part_types[p] = -1
                if dfs(v + 1, used + 1):
                    return True
                parts[p] = 0
                part_types[p] = -1

            return False

        if dfs(0, 0):
            result = []
            for p in range(k):
                if parts[p]:
                    lst = [u for u in range(N) if (parts[p] >> u) & 1]
                    result.append(lst)
            return result

    return _reconstruct_greedy(adj, comp)

def _reconstruct_greedy(adj, comp):
    order = sorted(range(N), key=lambda v: bin(adj[v]).count('1') + bin(comp[v]).count('1'), reverse=True)
    parts = []
    for v in order:
        for it, (typ, mask) in enumerate(parts):
            a = adj if typ == 0 else comp
            if all((a[v] >> u) & 1 for u in range(N) if (mask >> u) & 1):
                parts[it] = (typ, mask | (1 << v))
                break
        else:
            parts.append((0, 1 << v))
    return [[u for u in range(N) if (mask >> u) & 1] for (typ, mask) in parts]

def _validate(parts, adj):
    seen = [False] * N
    for part in parts:
        for v in part:
            if seen[v]:
                return False
            seen[v] = True
        m = 0
        for v in part:
            m |= 1 << v
        ms = [u for u in range(N) if (m >> u) & 1]
        clique = all((adj[u] >> w) & 1 for u in ms for w in ms if u < w)
        indep = all(not ((adj[u] >> w) & 1) for u in ms for w in ms if u < w)
        if not (clique or indep):
            return False
    return all(seen)

def construct():
    adj, comp = _setup()
    parts = _solve_exact(adj, comp)
    if not _validate(parts, adj):
        parts = _reconstruct_greedy(adj, comp)
    out = []
    for part in parts:
        out.append([int(v) for v in part])
    return out
# EVOLVE-BLOCK-END
