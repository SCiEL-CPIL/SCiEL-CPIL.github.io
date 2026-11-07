"""New strategy: max-flow feasibility oracle + simulated-annealing local search
over branch sets (different design family from the seed's greedy+DFS B&B)."""
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    t_start = time.time()
    DEADLINE = 85.0  # safety margin under the 90 s wall-clock kill

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return (time.time() - t_start) > DEADLINE

    # ---------- Max-flow (Dinic) ----------
    class Dinic:
        __slots__ = ('n', 'g', 'level', 'it')

        def __init__(self, n):
            self.n = n
            self.g = [[] for _ in range(n)]
            self.level = [0] * n
            self.it = [0] * n

        def add_edge(self, u, v, cap):
            self.g[u].append([v, cap, len(self.g[v])])
            self.g[v].append([u, 0, len(self.g[u]) - 1])

        def bfs(self, s, t):
            self.level = [-1] * self.n
            self.level[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for e in self.g[u]:
                    v, cap, _ = e
                    if cap > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        if v == t:
                            return True
                        q.append(v)
            return self.level[t] >= 0

        def dfs(self, u, t, f):
            if u == t:
                return f
            gu = self.g[u]
            while self.it[u] < len(gu):
                e = gu[self.it[u]]
                v, cap, rev = e
                if cap > 0 and self.level[v] == self.level[u] + 1:
                    d = self.dfs(v, t, min(f, cap))
                    if d > 0:
                        e[1] -= d
                        self.g[v][rev][1] += d
                        return d
                self.it[u] += 1
            return 0

        def max_flow(self, s, t):
            flow = 0
            INF = 1 << 30
            while self.bfs(s, t):
                self.it = [0] * self.n
                while True:
                    f = self.dfs(s, t, INF)
                    if f == 0:
                        break
                    flow += f
            return flow

    # ---------- Feasibility oracle via max-flow ----------
    # For a branch set, model each interior vertex with capacity 1 (vertex split),
    # and route one unit of flow per branch pair through a pair-specific
    # source->sink super-node. Max flow == number of pairs  <=>  feasible packing.
    def flow_feasible(branches):
        m = len(branches)
        if m < 2:
            return False
        branch_set = set(branches)
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
        num_pairs = len(pairs)

        # Node indexing:
        #   vertex v (0..n-1): in = 2v, out = 2v+1
        #   pair p (0..num_pairs-1): src = 2n + 2p, sink = 2n + 2p + 1
        N = 2 * n + 2 * num_pairs
        din = Dinic(N)

        # Vertex capacity 1 for non-branch vertices
        for v in range(n):
            if v in branch_set:
                continue
            din.add_edge(2 * v, 2 * v + 1, 1)
            din.add_edge(2 * v + 1, 2 * v, 0)  # placeholder reverse (handled by add_edge)

        # Undirected graph edges -> two directed arcs, capacity large
        BIG = num_pairs
        for u, v in edges:
            if u in branch_set and v in branch_set:
                continue  # direct branch-branch handled by pair arcs
            din.add_edge(2 * u + 1, 2 * v, BIG)
            din.add_edge(2 * v + 1, 2 * u, BIG)

        # Pair-specific source/sink connected to branch endpoints
        for p, (i, j) in enumerate(pairs):
            a, b = branches[i], branches[j]
            src = 2 * n + 2 * p
            sink = 2 * n + 2 * p + 1
            din.add_edge(src, 2 * a, 1)
            din.add_edge(2 * b + 1, sink, 1)
            # also allow the reverse direction path
            din.add_edge(src, 2 * b, 1)
            din.add_edge(2 * a + 1, sink, 1)

        # Super source/sink
        S = N
        T = N + 1
        din2 = Dinic(N + 2)
        # Copy edges by re-adding (simpler: rebuild)
        # Rebuild directly in din2 to avoid copying
        din2.g = [[] for _ in range(N + 2)]

        def add2(u, v, cap):
            din2.g[u].append([v, cap, len(din2.g[v])])
            din2.g[v].append([u, 0, len(din2.g[u]) - 1])

        for v in range(n):
            if v in branch_set:
                continue
            add2(2 * v, 2 * v + 1, 1)
        for u, v in edges:
            if u in branch_set and v in branch_set:
                continue
            add2(2 * u + 1, 2 * v, BIG)
            add2(2 * v + 1, 2 * u, BIG)
        for p, (i, j) in enumerate(pairs):
            a, b = branches[i], branches[j]
            src = 2 * n + 2 * p
            sink = 2 * n + 2 * p + 1
            add2(src, 2 * a, 1)
            add2(2 * b + 1, sink, 1)
            add2(src, 2 * b, 1)
            add2(2 * a + 1, sink, 1)
        for p in range(num_pairs):
            src = 2 * n + 2 * p
            sink = 2 * n + 2 * p + 1
            add2(S, src, 1)
            add2(sink, T, 1)

        din2.n = N + 2
        return din2.max_flow(S, T) == num_pairs

    # ---------- Extract actual paths from a feasible flow ----------
    def extract_paths(branches):
        m = len(branches)
        branch_set = set(branches)
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
        num_pairs = len(pairs)
        N = 2 * n + 2 * num_pairs
        S = N
        T = N + 1

        din = Dinic(N + 2)
        BIG = num_pairs

        def add2(u, v, cap):
            din.g[u].append([v, cap, len(din.g[v])])
            din.g[v].append([u, 0, len(din.g[u]) - 1])

        for v in range(n):
            if v in branch_set:
                continue
            add2(2 * v, 2 * v + 1, 1)
        for u, v in edges:
            if u in branch_set and v in branch_set:
                continue
            add2(2 * u + 1, 2 * v, BIG)
            add2(2 * v + 1, 2 * u, BIG)
        for p, (i, j) in enumerate(pairs):
            a, b = branches[i], branches[j]
            src = 2 * n + 2 * p
            sink = 2 * n + 2 * p + 1
            add2(src, 2 * a, 1)
            add2(2 * b + 1, sink, 1)
            add2(src, 2 * b, 1)
            add2(2 * a + 1, sink, 1)
        for p in range(num_pairs):
            src = 2 * n + 2 * p
            sink = 2 * n + 2 * p + 1
            add2(S, src, 1)
            add2(sink, T, 1)
        din.n = N + 2
        din.max_flow(S, T)

        # Walk each pair's flow to recover the vertex sequence.
        # For pair p, flow enters at src -> (2a or 2b) ... -> (2b+1 or 2a+1) -> sink.
        paths = []
        for p, (i, j) in enumerate(pairs):
            a, b = branches[i], branches[j]
            src = 2 * n + 2 * p
            sink = 2 * n + 2 * p + 1
            # Determine which branch the flow entered (a or b)
            entered = None
            for e in din.g[src]:
                if e[1] < BIG:  # residual decreased => used
                    entered = e[0]
                    break
            # entered is 2a or 2b
            start_branch = a if entered == 2 * a else b
            end_branch = b if start_branch == a else a
            # Follow flow from entered node to sink, converting node ids to vertices.
            seq = [start_branch]
            cur = entered
            seen = {cur}
            while cur != 2 * end_branch + 1:
                nxt = None
                for e in din.g[cur]:
                    v, cap, rev = e
                    if cap < BIG and (v not in seen):
                        # used edge (residual decreased from original cap)
                        nxt = v
                        break
                if nxt is None:
                    # fallback: any edge with flow
                    for e in din.g[cur]:
                        v, cap, rev = e
                        if v not in seen and cap < BIG:
                            nxt = v
                            break
                if nxt is None:
                    break
                seen.add(nxt)
                cur = nxt
                if cur >= 2 * n:
                    break
                if cur % 2 == 0:
                    seq.append(cur // 2)
            seq.append(end_branch)
            paths.append(seq)
        return paths

    # ---------- Local search with flow oracle ----------
    best_branches = None
    best_paths = None
    best_count = 0

    def update(branches, paths):
        nonlocal best_branches, best_paths, best_count
        if paths is not None and len(branches) > best_count:
            best_branches = list(branches)
            best_paths = list(paths)
            best_count = len(branches)

    # Greedy seed
    greedy = []
    for v in sorted(range(n), key=lambda x: -len(adj[x])):
        trial = greedy + [v]
        if len(trial) >= 2 and flow_feasible(trial):
            greedy = trial
            p = extract_paths(trial)
            update(trial, p)
        if time_up():
            break

    # Simulated annealing over branch sets
    if best_branches is not None and not time_up():
        cur = list(best_branches)
        cur_set = set(cur)
        T0 = 1.0
        T = T0
        cool = 0.999
        steps = 0
        while not time_up():
            steps += 1
            # Pick a random move
            r = random.random()
            if r < 0.4 and len(cur) < n:
                # add
                cand = random.choice([v for v in range(n) if v not in cur_set])
                new = cur + [cand]
            elif r < 0.7 and len(cur) > 2:
                # drop
                idx = random.randrange(len(cur))
                new = cur[:idx] + cur[idx + 1:]
            else:
                # swap
                if len(cur) < 2:
                    continue
                idx = random.randrange(len(cur))
                cand = random.choice([v for v in range(n) if v not in cur_set])
                new = cur[:idx] + [cand] + cur[idx + 1:]
            if len(new) < 2:
                T *= cool
                continue
            if flow_feasible(new):
                new_count = len(new)
                if new_count > len(cur):
                    cur = new
                    cur_set = set(cur)
                    p = extract_paths(cur)
                    update(cur, p)
                else:
                    # accept equal-size with some probability to explore
                    if random.random() < 0.3:
                        cur = new
                        cur_set = set(cur)
            T *= cool
            if steps % 500 == 0 and time_up():
                break

    # Random restarts if needed
    if best_branches is None or len(best_branches) < 2:
        for _ in range(200):
            if time_up():
                break
            size = random.randint(2, min(n, 12))
            seed = random.sample(range(n), size)
            if flow_feasible(seed):
                p = extract_paths(seed)
                update(seed, p)

    # Fallback
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]

    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(x) for x in p] for p in best_paths]
    return {'branches': branches_out, 'paths': paths_out}
