import time
import sys
from itertools import combinations
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = P['n']

def _build_adj():
    adj = [set() for _ in range(N)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_max_clique(adj):
    n = N
    best = []
    def bronk(R, Pset, Xset):
        nonlocal best
        if not Pset and not Xset:
            if len(R) > len(best):
                best = list(R)
            return
        U = Pset | Xset
        if U:
            pivot = max(U, key=lambda v: len(adj[v] & Pset))
            cands = [v for v in Pset if v not in adj[pivot]]
        else:
            cands = list(Pset)
        for v in cands:
            bronk(R | {v}, Pset & adj[v], Xset & adj[v])
            Pset.discard(v)
            Xset.add(v)
    deg_order = sorted(range(n), key=lambda v: len(adj[v]))
    for i, v in enumerate(deg_order):
        bronk({v}, set(adj[v]) & set(deg_order[i+1:]), set())
    if not best:
        bronk(set(), set(range(n)), set())
    return best

class _Flow:
    __slots__ = ('adj', 'cap', 'to', 'rev', 'n')
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]
        self.cap = [[] for _ in range(n)]
        self.to = [[] for _ in range(n)]
        self.rev = [[] for _ in range(n)]
    def add_edge(self, u, v, c):
        self.to[u].append(v); self.cap[u].append(c); self.rev[u].append(len(self.to[v]))
        self.to[v].append(u); self.cap[v].append(0); self.rev[v].append(len(self.to[u]))
    def max_flow(self, s, t):
        n = self.n
        flow = 0
        INF = 10**9
        while True:
            level = [-1]*n
            q = deque([s]); level[s] = 0
            while q:
                u = q.popleft()
                for i in range(len(self.to[u])):
                    v = self.to[u][i]
                    if self.cap[u][i] > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        q.append(v)
            if level[t] < 0:
                return flow
            it = [0]*n
            def dfs(u, f):
                if u == t:
                    return f
                for i in range(it[u], len(self.to[u])):
                    it[u] = i
                    v = self.to[u][i]
                    if self.cap[u][i] > 0 and level[v] == level[u] + 1:
                        ret = dfs(v, min(f, self.cap[u][i]))
                        if ret:
                            self.cap[u][i] -= ret
                            self.cap[v][self.rev[u][i]] += ret
                            return ret
                it[u] = len(self.to[u])
                return 0
            while True:
                pushed = dfs(s, INF)
                if not pushed:
                    break
                flow += pushed
        return flow

def _flow_paths(adj, branches):
    """Return list of internally-disjoint simple paths (one per branch pair) or None."""
    bs = set(branches)
    k = len(branches)
    if k < 2:
        return None
    M = 2*N
    def in_(x): return 2*x
    def out_(x): return 2*x + 1
    flow = _Flow(M)
    for x in range(N):
        if x in bs:
            flow.add_edge(in_(x), out_(x), 10**9)
        else:
            flow.add_edge(in_(x), out_(x), 1)
    for u in range(N):
        for v in adj[u]:
            if v > u:
                flow.add_edge(out_(u), in_(v), 1)
                flow.add_edge(out_(v), in_(u), 1)
    S = M - 2
    T = M - 1
    flow.add_edge(S, T, 10**9)
    pairs = list(combinations(branches, 2))
    for (u, v) in pairs:
        flow.add_edge(S, out_(u), 1)
        flow.add_edge(in_(v), T, 1)
    need = k*(k-1)//2
    if flow.max_flow(S, T) < need:
        return None
    # collect used directed edges
    used = set()
    for a in range(M):
        for i in range(len(flow.to[a])):
            if flow.cap[a][i] < 0:
                used.add((a, flow.to[a][i]))
    paths = []
    for (u, v) in pairs:
        path = [u]
        cur = out_(u)
        seen = {cur}
        while cur != in_(v):
            nxt = None
            for b in range(len(flow.to[cur])):
                if (cur, flow.to[cur][b]) in used:
                    nxt = flow.to[cur][b]
                    break
            if nxt is None or nxt in seen:
                return None
            seen.add(nxt)
            cur = nxt
            if cur % 2 == 0:
                path.append(cur//2)
        if path[-1] != v:
            return None
        paths.append(path)
    return paths

def _verify(adj, branches):
    paths = _flow_paths(adj, branches)
    if paths is None:
        return False
    bs = set(branches)
    interiors = set()
    for p in paths:
        if len(p) < 2 or p[0] not in bs or p[-1] not in bs:
            return False
        for x in p[1:-1]:
            if x in bs or x in interiors:
                return False
            interiors.add(x)
    return True

def construct():
    start = time.time()
    deadline = start + 55.0
    adj = _build_adj()

    clique = _find_max_clique(adj)
    best = list(clique)
    if time.time() > deadline - 15.0:
        paths = _flow_paths(adj, best)
        if paths is not None:
            return {'branches': [int(b) for b in best], 'paths': [[int(x) for x in p] for p in paths]}
        pairs = list(combinations(best, 2))
        return {'branches': [int(b) for b in best], 'paths': [[int(u), int(v)] for (u, v) in pairs]}

    # Greedy grow
    cur = set(clique)
    for _ in range(6):
        if time.time() > deadline - 15.0:
            break
        added = False
        cands = [v for v in range(N) if v not in cur]
        cands.sort(key=lambda v: -len(adj[v] & cur))
        for v in cands:
            if time.time() > deadline - 15.0:
                break
            nb = cur | {v}
            if _verify(adj, list(nb)):
                cur.add(v)
                added = True
        if len(cur) > len(best):
            best = list(cur)
        if not added:
            break

    # Local improvement: swap out a branch for a non-branch
    for _ in range(4):
        if time.time() > deadline - 12.0:
            break
        improved = False
        cur_set = set(best)
        nonb = [v for v in range(N) if v not in cur_set]
        nonb.sort(key=lambda v: -len(adj[v] & cur_set))
        for a in list(cur_set):
            if time.time() > deadline - 12.0:
                break
            for v in nonb:
                cand = cur_set - {a} | {v}
                if len(cand) < len(cur_set):
                    continue
                if _verify(adj, list(cand)):
                    best = list(cand)
                    cur_set = set(cand)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Final flow construction
    paths = _flow_paths(adj, best)
    if paths is not None:
        return {'branches': [int(b) for b in best], 'paths': [[int(x) for x in p] for p in paths]}

    # Fallback to baseline (clique edges)
    if time.time() > deadline - 5.0:
        pairs = list(combinations(best, 2))
        return {'branches': [int(b) for b in best], 'paths': [[int(u), int(v)] for (u, v) in pairs]}
    # Try to recover a valid subdivision by dropping vertices
    for drop in range(1, len(best)):
        if time.time() > deadline - 5.0:
            break
        for combo in combinations(best, drop):
            red = set(best) - set(combo)
            if len(red) < 2:
                continue
            p2 = _flow_paths(adj, list(red))
            if p2 is not None:
                return {'branches': [int(b) for b in red], 'paths': [[int(x) for x in p] for p in p2]}
    pairs = list(combinations(best, 2))
    return {'branches': [int(b) for b in best], 'paths': [[int(u), int(v)] for (u, v) in pairs]}

# EVOLVE-BLOCK-END
