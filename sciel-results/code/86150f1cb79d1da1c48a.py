"""Cage solver: fast girth verifier + greedy random regular + targeted local search + LUW fallback."""
import time
import threading
import random
import itertools
from collections import deque

import numpy as np

_DEADLINE = 120.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


# ---------------- exact girth via non-backtracking walks (vectorized) ----------
def _girth_np(adj_lists, n, g):
    """Return the exact girth (shortest cycle length) of a simple graph, or
    g if no cycle shorter than g exists. Uses non-backtracking BFS from all
    vertices simultaneously, vectorized with NumPy."""
    deg = np.array([len(a) for a in adj_lists], dtype=np.int32)
    m = int(deg.sum())
    pos = np.zeros(m, dtype=np.int32)
    for u in range(n):
        pos[u * (k := deg[u]): (u + 1) * k] = np.arange(k)
    # neighbor lookup table: nbr[u][p] = adj[u][p]
    nbr = np.zeros((n, k), dtype=np.int32)
    for u in range(n):
        nbr[u, :] = np.array(adj_lists[u], dtype=np.int32)

    # state (v, p)  ->  index v*k + p
    # prev state: the (u, q) we came from, where u = nbr[v][p]
    # We want to detect when a non-backtracking walk returns to its origin.

    # BFS layers. Layer 1 = all directed edges (v, p) with v = nbr[u][p].
    # Represent a state by its index. We store for each state the origin vertex
    # (u) and the distance (number of edges so far).

    # For efficiency, do layered BFS with NumPy.
    # current: array of state indices
    # origin: array of origin vertex for each state
    # dist: scalar (all same within a layer)

    # Initialize layer 1: for each directed edge (u -> v) where v = nbr[u][p]
    # state index = v*k + p, origin = u, dist = 1
    u_idx = np.repeat(np.arange(n, dtype=np.int32), deg)
    p_idx = pos
    v_idx = nbr[u_idx, p_idx]
    init_state = v_idx * k + p_idx
    init_origin = u_idx

    # prev_state for state (v, p) is (u, q) where u = nbr[v][p] and q = position
    # of v in adj[u]. We need pos[u][v].
    # Build pos lookup: pos_lookup[u][v] = position of v in adj[u], or -1.
    pos_lookup = np.full((n, n), -1, dtype=np.int32)
    for u in range(n):
        for p, v in enumerate(adj_lists[u]):
            pos_lookup[u, v] = p

    # visited: for each state, store the origin vertex that first reached it,
    # and the distance. We'll use arrays.
    visited_origin = np.full(n * k, -1, dtype=np.int32)
    visited_dist = np.zeros(n * k, dtype=np.int32)

    # Mark initial layer
    visited_origin[init_state] = init_origin
    visited_dist[init_state] = 1

    cur_state = init_state
    cur_origin = init_origin
    dist = 1

    while len(cur_state) > 0:
        # Extend: from state (v, p), origin u, go to (w, r) where w != v's parent.
        # w = nbr[v][r], r in 0..k-1, r != p (non-backtracking)
        # new state index = w*k + r
        # new origin = u (same origin)
        # Check: if w == u -> cycle of length dist+1.
        # If state (w, r) already visited with a DIFFERENT origin -> cycle.
        #   cycle length = dist + 1 + visited_dist[(w,r)] ... actually need to
        #   be careful. Standard: if we reach (w, r) from (v, p) and (w, r) was
        #   first reached from some state, the cycle length = dist + 1 +
        #   visited_dist[(w, r)] only if same origin... Let me use the clean
        #   formulation: a non-backtracking walk from origin u: at each step
        #   the state is (current, prev). A cycle forms when current == origin
        #   (and prev != the neighbor we started from, but non-backtracking
        #   handles that). So we only need to check w == origin.

        v_cur = cur_state // k
        p_cur = cur_state % k

        # all next neighbors
        r_all = np.tile(np.arange(k, dtype=np.int32), len(v_cur))
        w_all = nbr[v_cur, r_all]
        # mask out r == p_cur (backtrack)
        mask = (r_all != p_cur)
        w_all = w_all[mask]
        r_all = r_all[mask]
        o_all = cur_origin[mask]

        # check w == origin
        hit = (w_all == o_all)
        if hit.any():
            return dist + 1

        new_state = w_all * k + r_all
        new_origin = o_all

        # check visited
        old_origin = visited_origin[new_state]
        is_new = (old_origin == -1)

        if not is_new.all():
            # some states were visited. If visited with a different origin,
            # that means two non-backtracking walks from different origins met
            # at the same state -> forms a cycle. Cycle length:
            #   dist + 1 + visited_dist[new_state] (the two walks join).
            # Actually the cycle is: origin_a -> ... -> (w,r) <- ... <- origin_b.
            # length = dist + 1 (from origin_a to w) + visited_dist (from w back
            # to origin_b) ... but the two paths meet at (w, r). The cycle
            # length = dist + 1 + visited_dist[new_state].
            diff = is_new == False
            cand = (dist + 1 + visited_dist[new_state[diff]])
            if cand.min() < g:
                return int(cand.min())
            # For same-origin revisit, that's just the same walk, ignore.
            # Actually if old_origin == new_origin it's a cycle too but we'd have
            # caught w==origin. So diff-origin is the only new cycle info.

        # proceed with only new states
        new_state = new_state[is_new]
        new_origin = new_origin[is_new]

        if len(new_state) == 0:
            break

        visited_origin[new_state] = new_origin
        visited_dist[new_state] = dist + 1
        cur_state = new_state
        cur_origin = new_origin
        dist += 1

        if dist >= g:
            break

    return g


def _nonback_girth(adj, n):
    """Exact girth via non-backtracking walks from every vertex."""
    if n == 0:
        return 10 ** 9
    try:
        return _girth_np(adj, n, 10 ** 9)
    except Exception:
        # fallback to pure-python
        prev = {}
        q = deque()
        for u in range(n):
            for p in adj[u]:
                key = (p, u)
                prev[key] = (u, 1)
                q.append(key)
        while q:
            (v, p), (u, dist) = q.popleft()
            for w in adj[v]:
                if w == p:
                    continue
                key = (w, v)
                if w == u:
                    return dist + 1
                if key in prev:
                    continue
                prev[key] = (v, dist + 1)
                q.append(key)
        return 10 ** 9


def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if any(len(adj[i]) != k for i in range(n)):
        return False
    for i in range(n):
        if len(set(adj[i])) != len(adj[i]):
            return False
        for j in adj[i]:
            if i <= j:
                return False
    return _nonback_girth(adj, n) >= g


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj


# ---------------- Moore bound feasibility --------------------------------------
def _moore(k, g):
    R = (g - 1) // 2
    return 1 + k * sum((k - 1) ** i for i in range(R))


def _min_n_feasible(k, g):
    """Smallest n for which a k-regular girth-g graph could exist (parity +
    Moore bound)."""
    moore = _moore(k, g)
    n = moore
    while (n * k) % 2 != 0 or n < k + 1:
        n += 1
    return n


# ---------------- construction 1: LUW (Lazebnik-Ustimenko-Woldar) ------------
def _luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# ---------------- construction 2: Greedy random regular graph ------------------
def _random_regular(n, k, rng):
    """Generate a random simple k-regular graph on n vertices using greedy
    completion. Returns adj list or None."""
    if n * k % 2 != 0:
        return None
    if n < k + 1:
        return None
    
    for attempt in range(20):
        adj = [set() for _ in range(n)]
        remaining = [k] * n
        active = list(range(n))
        rng.shuffle(active)
        
        ok = True
        for u in active:
            if remaining[u] == 0:
                continue
            # Find k - len(adj[u]) candidates to connect to
            need = remaining[u]
            candidates = [v for v in active if v != u and remaining[v] > 0 and v not in adj[u]]
            rng.shuffle(candidates)
            
            if len(candidates) < need:
                ok = False
                break
            
            for v in candidates[:need]:
                adj[u].add(v)
                adj[v].add(u)
                remaining[u] -= 1
                remaining[v] -= 1
        
        if not ok:
            continue
        
        if all(len(adj[i]) == k for i in range(n)):
            return [sorted(adj[i]) for i in range(n)]
    
    return None


# ---------------- construction 3: Random search + targeted local search --------
def _find_short_cycle(adj, n, g_limit=None):
    """Return a list of vertices forming a cycle shorter than g_limit, or None."""
    best = None
    for s in range(n):
        parent = [-1] * n
        dist = [-1] * n
        parent[s] = -2  # mark as root
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and parent[v] != u:
                    # Found a cycle
                    L = dist[u] + dist[v] + 1
                    if g_limit and L >= g_limit:
                        continue
                    if best is None or L < len(best):
                        cyc = _reconstruct_cycle(adj, n, u, v, parent, dist, s)
                        if cyc is not None:
                            best = cyc
    return best


def _reconstruct_cycle(adj, n, u, v, parent, dist, s):
    """Reconstruct a cycle from BFS tree paths to u and v."""
    # Get path from s to u
    path_u = []
    x = u
    while x != -2:  # -2 marks root
        path_u.append(x)
        x = parent[x]
    path_u.reverse()  # s -> ... -> u
    
    # Get path from s to v
    path_v = []
    x = v
    while x != -2:
        path_v.append(x)
        x = parent[x]
    path_v.reverse()  # s -> ... -> v
    
    # Find LCA (first common node from s)
    set_u = set(path_u)
    lca = None
    for node in path_v:
        if node in set_u:
            lca = node
            break
    
    if lca is None:
        return None
    
    i_u = path_u.index(lca)
    i_v = path_v.index(lca)
    
    # Cycle: lca -> ... -> u, edge u-v, v -> ... -> lca
    seg_u = path_u[i_u:]  # lca to u
    seg_v = path_v[i_v:][::-1]  # v to lca (reversed)
    
    cycle = seg_u + seg_v[:-1]  # exclude lca at end of seg_v
    return cycle


def _local_search(adj, n, k, g, rng, max_steps=500):
    """Targeted local search: repeatedly find shortest cycle and break it via
    2-opt swaps that preserve regularity."""
    adj = [set(a) for a in adj]
    
    for step in range(max_steps):
        if _elapsed() > _DEADLINE - 10.0:
            return None
        
        # Quick girth check using BFS from each vertex
        girth = _quick_girth(adj, n, g)
        if girth >= g:
            return [sorted(a) for a in adj]
        
        # Find a short cycle
        cyc = _find_short_cycle(adj, n, g)
        if cyc is None or len(cyc) >= g:
            return [sorted(a) for a in adj]
        
        # Try multiple swaps to break the cycle
        m = len(cyc)
        improved = False
        for _ in range(20):
            eidx = rng.randrange(m)
            u = cyc[eidx]
            v = cyc[(eidx + 1) % m]
            if v not in adj[u]:
                continue
            
            # Find a valid swap
            for _ in range(30):
                w = rng.randrange(n)
                if w in (u, v) or w in adj[u] or w in adj[v]:
                    continue
                
                # Pick x from adj[w]
                if not adj[w]:
                    continue
                x = rng.choice(list(adj[w]))
                if x in (u, v, w) or x in adj[v]:
                    continue
                
                # Perform swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(x)
                adj[x].remove(w)
                adj[u].add(w)
                adj[w].add(u)
                adj[v].add(x)
                adj[x].add(v)
                improved = True
                break
            
            if improved:
                break
    
    girth = _quick_girth(adj, n, g)
    if girth >= g:
        return [sorted(a) for a in adj]
    return None


def _quick_girth(adj, n, g):
    """Fast girth check: BFS from each vertex, stop if we find cycle < g."""
    for s in range(n):
        visited = [-1] * n
        parent = [-1] * n
        visited[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if visited[u] + 1 >= g:
                continue
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and parent.get(v) != u:
                    # Found a cycle
                    cycle_len = visited[u] + visited[v] + 1
                    if cycle_len < g:
                        return cycle_len
    return g


def _random_search(k, g, seed=0):
    rng = random.Random(seed)
    n0 = _min_n_feasible(k, g)
    for n in range(n0, n0 + 80):
        if (n * k) % 2 != 0:
            continue
        if n < k + 1:
            continue
        for attempt in range(30):
            if _elapsed() > _DEADLINE - 12.0:
                return None
            adj = _random_regular(n, k, rng)
            if adj is None:
                continue
            
            # Quick girth check
            if _quick_girth(adj, n, g) >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
            
            # Local search
            res = _local_search(adj, n, k, g, rng, max_steps=300)
            if res is not None:
                edges = [(i, j) for i in range(n) for j in res[i] if i < j]
                return n, edges
    return None


# ---------------- construction 4: Circulant search (small n) -------------------
def _circulant_search(k, g):
    n0 = _min_n_feasible(k, g)
    max_n = min(n0 + 40, 220)
    if k % 2 == 0:
        half = k // 2
        for n in range(n0, max_n + 1):
            if (n * k) % 2 != 0:
                continue
            if n < k + 1:
                continue
            for S in itertools.combinations(range(1, n // 2 + 1), half):
                S = list(S)
                if any((n - s) % n in S for s in S):
                    continue
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S:
                        adj[i].add((i + s) % n)
                        adj[i].add((i - s) % n)
                if all(len(adj[i]) == k for i in range(n)):
                    adj = [sorted(adj[i]) for i in range(n)]
                    if _nonback_girth(adj, n) >= g:
                        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return n, edges
    else:
        m = (k - 1) // 2
        for n in range(n0, max_n + 1, 2):
            if n < k + 1:
                continue
            for S_vals in itertools.combinations(range(1, n // 2), m):
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S_vals:
                        adj[i].add((i + s) % n)
                        adj[i].add((i - s) % n)
                    adj[i].add((i + n // 2) % n)
                if all(len(adj[i]) == k for i in range(n)):
                    adj = [sorted(adj[i]) for i in range(n)]
                    if _nonback_girth(adj, n) >= g:
                        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return n, edges
    return None


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}

    jobs = [
        ("luw", lambda: _luw(k, g)),
        ("circ", lambda: _circulant_search(k, g)),
        ("rand0", lambda: _random_search(k, g, seed=0)),
        ("rand1", lambda: _random_search(k, g, seed=1)),
        ("rand2", lambda: _random_search(k, g, seed=2)),
        ("rand3", lambda: _random_search(k, g, seed=3)),
        ("rand4", lambda: _random_search(k, g, seed=4)),
        ("rand5", lambda: _random_search(k, g, seed=5)),
    ]

    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder, k, g), daemon=True)
        t.start()
        threads.append(t)

    while _elapsed() < _DEADLINE - 5.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.03)

    best = holder.get("best")
    if best is None:
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best


def _worker(name, fn, holder, k, g):
    try:
        res = fn()
    except Exception:
        res = None
    if res is None:
        return
    n, edges = res
    if _elapsed() >= _DEADLINE - 2.0:
        return
    try:
        adj = _build_adj(edges, n)
        if n > 0 and _is_valid(adj, n, k, g):
            cur = holder.get("best")
            if cur is None or n < cur[0]:
                holder["best"] = (n, edges)
    except Exception:
        pass
