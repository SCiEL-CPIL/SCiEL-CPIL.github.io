"""Combined solver: structural candidate sweep + degeneracy-ordered backtracking
with incremental bicolored-cycle detection, plus local repair and cycle-aware pruning."""
import time
import random
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_valid(coloring):
    """Check adjacent differ and no bicolored cycles. Returns True if valid."""
    n = len(coloring)
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    color_pairs = set()
    for u, v in P['edges']:
        cu, cv = coloring[u], coloring[v]
        color_pairs.add((min(cu, cv), max(cu, cv)))
    for c1, c2 in color_pairs:
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True
        for u, v in P['edges']:
            cu, cv = coloring[u], coloring[v]
            if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                if not union(u, v):
                    return False
    return True


def _precompute_cycles():
    """Precompute all simple cycles of length <= 8 in the graph.
    Returns a list of cycles, each cycle is a tuple of vertices in order."""
    n = P['n']
    adj = _build_adj()
    cycles = []
    seen = set()
    
    for start in range(n):
        # DFS to find cycles back to start
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) > 8:
                continue
            for nb in adj[node]:
                if nb == start and len(path) >= 3:
                    # Found a cycle
                    cycle = tuple(path)
                    # Normalize: rotate so min vertex is first
                    min_idx = cycle.index(min(cycle))
                    norm = cycle[min_idx:] + cycle[:min_idx]
                    if norm not in seen:
                        seen.add(norm)
                        cycles.append(norm)
                elif nb not in path and nb > start:
                    # Only explore vertices > start to avoid duplicates
                    stack.append((nb, path + [nb]))
    
    return cycles


def _num_colors(coloring):
    return len(set(coloring))


def _generate_candidates(n):
    """Yield candidate colorings in order of increasing expected color count."""
    for k in range(2, 15):
        for a in range(1, k):
            for b in range(k):
                yield [(a * v + b) % k for v in range(n)]
    for k in range(2, 15):
        yield [v % k for v in range(n)]
    for k in range(2, 10):
        yield [(v // 10 + v % 10) % k for v in range(n)]
    for k in range(2, 10):
        yield [(v // 5) % k for v in range(n)]
    yield [(v // 10) * 2 + (v % 2) for v in range(n)]
    yield [(v % 4) + 4 * ((v // 10) % 2) for v in range(n)]
    yield [(v % 5) + 5 * ((v // 10) % 2) for v in range(n)]


def _try_candidates(n, deadline):
    """Try candidate colorings, return the best valid one found."""
    best = None
    best_count = n + 1
    for coloring in _generate_candidates(n):
        if time.time() > deadline:
            break
        ok = True
        for u, v in P['edges']:
            if coloring[u] == coloring[v]:
                ok = False
                break
        if not ok:
            continue
        if _check_valid(coloring):
            cnt = _num_colors(coloring)
            if cnt < best_count:
                best_count = cnt
                best = [int(x) for x in coloring]
            if best_count <= 3:
                break
    return best, best_count


class _Backtracker:
    """Degeneracy-ordered backtracking with incremental bicolored-cycle detection
    and precomputed cycle constraints for stronger pruning."""
    def __init__(self, adj, num_colors, deadline, cycles=None):
        self.n = len(adj)
        self.adj = adj
        self.num_colors = num_colors
        self.deadline = deadline
        # Degeneracy ordering
        order = self._degeneracy_order(adj)
        self.order = order
        self.order_pos = [0] * self.n
        for i, v in enumerate(order):
            self.order_pos[v] = i
        self.coloring = [-1] * self.n
        # For each color pair, maintain a union-find structure
        # Key: (c1, c2) with c1 < c2
        self.uf = {}  # (c1,c2) -> {parent, rank}
        # Precompute which edges are "forward" in the degeneracy order
        # For each vertex, the edges to already-colored neighbors
        self.forward_edges = [[] for _ in range(self.n)]
        for u in range(self.n):
            for v in adj[u]:
                if self.order_pos[v] < self.order_pos[u]:
                    self.forward_edges[u].append(v)
        self.nodes = 0  # count of DFS nodes visited
        # Precomputed cycles for constraint propagation
        self.cycles = cycles if cycles is not None else []
        # For each cycle, track which vertices are colored and their colors
        self.cycle_state = [None] * len(self.cycles)

    def _degeneracy_order(self, adj):
        n = len(adj)
        deg = [len(adj[i]) for i in range(n)]
        order = []
        remaining = set(range(n))
        while remaining:
            # Find vertex with min degree in remaining
            min_v = min(remaining, key=lambda x: deg[x])
            order.append(min_v)
            remaining.remove(min_v)
            for nb in adj[min_v]:
                if nb in remaining:
                    deg[nb] -= 1
        # Reverse so that when we process in this order, neighbors come first
        return order[::-1]

    def _uf_find(self, key, x):
        if key not in self.uf:
            self.uf[key] = {'parent': {}, 'rank': {}}
        uf = self.uf[key]
        if x not in uf['parent']:
            uf['parent'][x] = x
            uf['rank'][x] = 0
        parent = uf['parent']
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _uf_union(self, key, a, b):
        """Union a and b. Returns False if cycle detected."""
        if key not in self.uf:
            self.uf[key] = {'parent': {}, 'rank': {}}
        uf = self.uf[key]
        # Initialize if needed
        if a not in uf['parent']:
            uf['parent'][a] = a
            uf['rank'][a] = 0
        if b not in uf['parent']:
            uf['parent'][b] = b
            uf['rank'][b] = 0
        parent = uf['parent']
        rank = uf['rank']
        # Find
        ra = a
        while parent[ra] != ra:
            parent[ra] = parent[parent[ra]]
            ra = parent[ra]
        rb = b
        while parent[rb] != rb:
            parent[rb] = parent[parent[rb]]
            rb = parent[rb]
        if ra == rb:
            return False  # cycle
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    def _uf_rollback(self, snapshot):
        """Rollback UF structures to snapshot."""
        for key, data in snapshot.items():
            if key in self.uf:
                self.uf[key]['parent'] = data['parent']
                self.uf[key]['rank'] = data['rank']
            else:
                self.uf[key] = data

    def _take_snapshot(self):
        """Take a snapshot of UF state for rollback."""
        snap = {}
        for key, data in self.uf.items():
            snap[key] = {'parent': dict(data['parent']), 'rank': dict(data['rank'])}
        return snap

    def dfs(self, idx):
        if time.time() > self.deadline:
            return None
        self.nodes += 1
        if self.nodes % 10000 == 0 and time.time() > self.deadline:
            return None
        if idx == self.n:
            return self.coloring[:]
        v = self.order[idx]
        used_by_neighbors = set()
        for nb in self.forward_edges[v]:
            if self.coloring[nb] != -1:
                used_by_neighbors.add(self.coloring[nb])
        
        # Try each color
        for c in range(self.num_colors):
            if c in used_by_neighbors:
                continue
            self.coloring[v] = c
            snapshot = self._take_snapshot()
            valid = True
            for nb in self.forward_edges[v]:
                cn = self.coloring[nb]
                if cn == -1:
                    continue
                c1, c2 = min(c, cn), max(c, cn)
                key = (c1, c2)
                if not self._uf_union(key, v, nb):
                    valid = False
                    break
            # Additional pruning: check if any precomputed cycle becomes invalid
            if valid and self.cycles:
                valid = self._check_cycle_constraints(v)
            if valid:
                result = self.dfs(idx + 1)
                if result is not None:
                    return result
            self._uf_rollback(snapshot)
            self.coloring[v] = -1
        return None
    
    def _check_cycle_constraints(self, v):
        """Check if coloring v creates a bicolored cycle in any precomputed cycle.
        A cycle is bicolored if all its edges connect vertices with the same two colors."""
        for ci, cycle in enumerate(self.cycles):
            if v not in cycle:
                continue
            # Check if this cycle could be bicolored
            colors_in_cycle = set()
            for u in cycle:
                cu = self.coloring[u]
                if cu != -1:
                    colors_in_cycle.add(cu)
            # If we have exactly 2 colors and all edges are between these two,
            # check if it forms a cycle (which it does by definition)
            if len(colors_in_cycle) == 2:
                c1, c2 = sorted(colors_in_cycle)
                # Check all edges in the cycle
                all_bicolored = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    w = cycle[(i+1) % len(cycle)]
                    cu, cw = self.coloring[u], self.coloring[w]
                    if cu == -1 or cw == -1:
                        all_bicolored = False
                        break
                    if not ((cu == c1 and cw == c2) or (cu == c2 and cw == c1)):
                        all_bicolored = False
                        break
                if all_bicolored:
                    return False
        return True


def _backtrack_coloring(adj, num_colors, deadline, cycles=None):
    """Backtracking to find valid coloring with exactly num_colors colors."""
    bt = _Backtracker(adj, num_colors, deadline, cycles=cycles)
    return bt.dfs(0)


def _local_repair(start_coloring, adj, max_iter, deadline):
    """Try to reduce colors by local moves. Returns improved coloring or None."""
    coloring = [int(x) for x in start_coloring]
    n = len(coloring)
    best = [int(x) for x in coloring]
    best_count = _num_colors(coloring)
    rng = random.Random(12345)

    def try_move():
        nonlocal best, best_count
        v = rng.randrange(n)
        used = set()
        for nb in adj[v]:
            used.add(coloring[nb])
        avail = [c for c in range(best_count) if c not in used]
        if not avail:
            return
        c_new = rng.choice(avail)
        if c_new == coloring[v]:
            return
        old = coloring[v]
        coloring[v] = c_new
        if _check_valid(coloring):
            cnt = _num_colors(coloring)
            if cnt < best_count:
                best_count = cnt
                best = [int(x) for x in coloring]
        else:
            coloring[v] = old

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        try_move()
    return best if best_count < _num_colors(start_coloring) else None


def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    adj = _build_adj()
    n = P['n']

    # Precompute cycles for constraint propagation
    cycles = _precompute_cycles()
    
    # Phase 1: systematic candidate sweep (affine + regular patterns)
    best, best_count = _try_candidates(n, deadline - 30)
    if best is not None and best_count <= 3:
        return best

    # Phase 2: Backtracking with increasing k, starting from a small k
    # Try k=3, 4, 5, 6 in order with cycle-aware pruning
    for k in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        result = _backtrack_coloring(adj, k, deadline, cycles=cycles)
        if result is not None:
            if _check_valid(result):
                return [int(x) for x in result]

    # Phase 3: local repair starting from the best candidate (or a fresh one)
    if best is None:
        best = [v % 5 for v in range(n)]
    improved = _local_repair(best, adj, max_iter=100000, deadline=deadline - 2)
    if improved is not None:
        return improved

    # Phase 4: if we have a valid result, return it; else guaranteed fallback
    if best is not None and _check_valid(best):
        return best

    # Guaranteed valid: all distinct colors
    return list(range(n))


if __name__ == "__main__":
    result = construct()
    print("colors used:", _num_colors(result))
    print("valid:", _check_valid(result))
    # Debug: show cycle count
    cycles = _precompute_cycles()
    print(f"Precomputed {len(cycles)} cycles")
