"""Minimize distinct colors for degree-4 graph with bicolored-forest constraint.
Uses structured greedy + backtracking with fast bicolored-forest verification."""
import time
import random
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_N = P['n']
_EDGES = P['edges']
_ADJ = [[] for _ in range(_N)]
for _a, _b in _EDGES:
    _ADJ[_a].append(_b)
    _ADJ[_b].append(_a)


def _bfs_2color():
    """Return proper 2-coloring if bipartite, else None."""
    color = [-1] * _N
    for start in range(_N):
        if color[start] != -1:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            cu = color[u]
            for v in _ADJ[u]:
                if color[v] == -1:
                    color[v] = 1 - cu
                    q.append(v)
                elif color[v] == cu:
                    return None
    return color


def _is_valid(coloring):
    """Check proper coloring and bicolored-forest constraint."""
    for a, b in _EDGES:
        if coloring[a] == coloring[b]:
            return False
    present = sorted(set(coloring))
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            ci, cj = present[i], present[j]
            parent = list(range(_N))
            rank = [0] * _N

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True

            for a, b in _EDGES:
                ca, cb = coloring[a], coloring[b]
                if ca in (ci, cj) and cb in (ci, cj):
                    if not union(a, b):
                        return False
    return True


def _greedy_color(k, order=None):
    """Greedy coloring with k colors."""
    if order is None:
        order = sorted(range(_N), key=lambda v: (-len(_ADJ[v]), v))
    coloring = [-1] * _N
    for v in order:
        used = set()
        for nb in _ADJ[v]:
            if coloring[nb] != -1:
                used.add(coloring[nb])
        for c in range(k):
            if c not in used:
                coloring[v] = c
                break
        if coloring[v] == -1:
            return None
    return coloring


class _RollbackUF:
    """Union-Find with rollback support."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.history = []

    def find(self, x):
        # No path compression for rollback safety (or careful restoration)
        # Given N=50, tree depth is small, so linear find is fine.
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.history.append((ry, rx, self.rank[rx]))
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def rollback(self):
        if not self.history:
            return
        ry, rx, rank_rx = self.history.pop()
        self.parent[ry] = ry
        self.rank[rx] = rank_rx

    def snapshot(self):
        return len(self.history)

    def restore(self, snap):
        while len(self.history) > snap:
            self.rollback()


def _backtrack_k(k, deadline):
    """Backtracking search for k-coloring with incremental bicolored-forest check."""
    color = [-1] * _N
    # Order: static, sorted by index. 
    # Dynamic MCV might be better, but static is simpler and often sufficient for N=50.
    # Let's try sorting by degree (all 4) then index.
    order = list(range(_N))
    
    # We maintain a dict of UF for each active color pair (ci, cj) where ci < cj
    ufs = {}
    
    def get_uf(ci, cj):
        key = (min(ci, cj), max(ci, cj))
        if key not in ufs:
            ufs[key] = _RollbackUF(_N)
        return ufs[key]
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == _N:
            return True
        
        v = order[pos]
        used = set()
        for nb in _ADJ[v]:
            if color[nb] != -1:
                used.add(color[nb])
                
        # Try colors 0..k-1
        # Heuristic: try colors that are already used in neighbors first? 
        # Or just 0..k-1.
        for c in range(k):
            if c in used:
                continue
            
            color[v] = c
            
            # Check constraints with neighbors
            ok = True
            snapshots = []
            
            for u in _ADJ[v]:
                cu = color[u]
                if cu == -1 or cu == c:
                    continue
                
                ci, cj = min(c, cu), max(c, cu)
                uf = get_uf(ci, cj)
                snap = uf.snapshot()
                snapshots.append((uf, snap))
                
                if not uf.union(v, u):
                    ok = False
                    break
            
            if ok:
                if backtrack(pos + 1):
                    return True
            
            # Rollback
            for uf, snap in reversed(snapshots):
                uf.restore(snap)
            color[v] = -1
            
        return False
    
    if backtrack(0):
        return color[:]
    return None


def construct():
    deadline = time.time() + 85.0
    
    # 1) Check bipartite (2 colors)
    c2 = _bfs_2color()
    if c2 is not None and _is_valid(c2):
        return [int(x) for x in c2]
    
    # 2) Try k=3,4,5,... with greedy first, then fast backtracking
    best = None
    for k in range(3, 12):
        if time.time() > deadline - 5.0:
            break
        
        # Try greedy with multiple orderings
        rng = random.Random(42 + k)
        found = None
        order1 = sorted(range(_N), key=lambda v: (-len(_ADJ[v]), v))
        c = _greedy_color(k, order1)
        if c is not None and _is_valid(c):
            found = c
            
        if found is None:
            for attempt in range(5):
                if time.time() > deadline - 2.0:
                    break
                order = list(range(_N))
                rng.shuffle(order)
                c = _greedy_color(k, order)
                if c is not None and _is_valid(c):
                    found = c
                    break
        
        if found is None:
            # Try fast backtracking
            # Give it a tight deadline to avoid hanging
            bt_deadline = min(deadline - 1.0, time.time() + 10.0)
            found = _backtrack_k(k, bt_deadline)
            
        if found is not None:
            best = found
            break
    
    # 3) Fallback: greedy with more colors
    if best is None:
        for k in range(5, 15):
            if time.time() > deadline - 5.0:
                break
            c = _greedy_color(k)
            if c is not None and _is_valid(c):
                best = c
                break
    
    # 4) Last resort: unique colors
    if best is None:
        best = list(range(_N))
    
    return [int(x) for x in best]
