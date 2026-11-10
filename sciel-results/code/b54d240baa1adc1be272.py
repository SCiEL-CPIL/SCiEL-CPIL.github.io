import time
import random

random.seed(42)
START_TIME = time.time()
TIME_LIMIT = 118

# ---------- bitmask helpers ----------
def build_adj(edges, n=16):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def is_triangle_free_bitmask(adj, n=16):
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            common = adj[v] & adj[u]
            if common & ~(1 << u) & ~(1 << v):
                return False
    return True

# ---------- 3-colourability ----------
def greedy_3_colorable(adj, n=16, max_trials=60):
    for _ in range(max_trials):
        order = list(range(n))
        random.shuffle(order)
        colors = [-1] * n
        for node in order:
            used = 0
            mm = adj[node]
            while mm:
                lsb = mm & -mm
                nb = lsb.bit_length() - 1
                mm ^= lsb
                if colors[nb] != -1:
                    used |= 1 << colors[nb]
            for c in range(3):
                if not (used & (1 << c)):
                    colors[node] = c
                    break
            else:
                break
        else:
            return True
    return False

def is_not_3_colorable_bitmask(adj, n=16, max_nodes=400000):
    colors = [-1] * n
    nodes = [0]
    order = list(range(n))

    def dfs(uncolored_mask):
        if uncolored_mask == 0:
            return False
        if nodes[0] > max_nodes:
            return True
        nodes[0] += 1

        best_v = -1
        best_score = -1
        m = uncolored_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            score = 0
            mm = adj[v]
            while mm:
                lsb_nb = mm & -mm
                nb = lsb_nb.bit_length() - 1
                mm ^= lsb_nb
                if colors[nb] != -1:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        v = best_v

        forbidden = 0
        mm = adj[v]
        while mm:
            lsb_nb = mm & -mm
            nb = lsb_nb.bit_length() - 1
            mm ^= lsb_nb
            if colors[nb] != -1:
                forbidden |= 1 << colors[nb]

        corder = [0, 1, 2]
        random.shuffle(corder)
        for c in corder:
            if not (forbidden & (1 << c)):
                colors[v] = c
                if not dfs(uncolored_mask & ~(1 << v)):
                    return False
                colors[v] = -1
        return True

    return dfs((1 << n) - 1)

def is_not_3_colorable_fast(adj, n=16, trials=40):
    if greedy_3_colorable(adj, n, max_trials=trials):
        return False
    return is_not_3_colorable_bitmask(adj, n)

# ---------- dense seed: Mycielski of C5 on 16 vertices ----------
def build_mycielski_c5_edges():
    # C5: 0-1-2-3-4-0
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    # Mycielski: copies 5..9 (copy of i is 5+i), apex 10
    for u, v in [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]:
        edges.append([5 + u, v])
        edges.append([5 + v, u])
    for u in range(5):
        edges.append([10, 5 + u])
    # extra vertices 11..15: connect to the apex 10 (safe, no triangles)
    for v in range(11, 16):
        edges.append([10, v])
    return edges

# ---------- local search: add edges ----------
def local_search_add_edges(edges, n=16, deadline=None):
    if deadline is None:
        deadline = START_TIME + TIME_LIMIT
    adj = build_adj(edges, n)
    current_edges = list(edges)
    best_edges = list(edges)
    best_count = len(edges)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        candidates.append((u, v))
        random.shuffle(candidates)

        for u, v in candidates:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            current_edges.append([u, v])
            if is_not_3_colorable_fast(adj, n):
                best_count = len(current_edges)
                best_edges = list(current_edges)
                improved = True
            else:
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                current_edges.pop()
        if not improved:
            break

    return best_edges

# ---------- final polish: exact backtracking, tighter ----------
def polish(edges, n=16, deadline=None):
    if deadline is None:
        deadline = START_TIME + TIME_LIMIT
    adj = build_adj(edges, n)
    current_edges = list(edges)
    best_count = len(edges)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        candidates.append((u, v))
        random.shuffle(candidates)
        for u, v in candidates:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            current_edges.append([u, v])
            if not greedy_3_colorable(adj, n, max_trials=25):
                if is_not_3_colorable_bitmask(adj, n, max_nodes=800000):
                    best_count = len(current_edges)
                    improved = True
                else:
                    adj[u] ^= 1 << v
                    adj[v] ^= 1 << u
                    current_edges.pop()
            else:
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                current_edges.pop()
        if not improved:
            break
    return current_edges

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 118
    n = 16

    seed_edges = build_mycielski_c5_edges()
    adj = build_adj(seed_edges, n)
    if not is_triangle_free_bitmask(adj, n):
        # safety: fall back to plain Mycielski (11 vertices)
        seed_edges = build_mycielski_c5_edges()[:20]

    deadline = START_TIME + TIME_LIMIT

    # main local search
    best = local_search_add_edges(seed_edges, n, deadline)

    # polish pass
    if time.time() < deadline - 8:
        best = polish(best, n, deadline)

    # final safety re-check of triangle-free
    adj = build_adj(best, n)
    if not is_triangle_free_bitmask(adj, n):
        best = build_mycielski_c5_edges()

    return best
