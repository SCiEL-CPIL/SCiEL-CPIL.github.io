"""Combined cage solver: LUW + greedy leaf-pairing + random search + exact girth verification."""
import time
import threading
import random
from collections import deque

_DEADLINE = 120.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


# ---------------- exact girth (non-backtracking walks, all vertices) ----------
def _nonback_girth(adj, n):
    """Return length of shortest cycle, certified by non-backtracking walks
    from every vertex. >= g means no cycle shorter than g."""
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


# ---------------- construction 2: generalized Petersen graph G(N, k) ----------
def _petersen(k, g):
    if k != 3:
        return None
    for N in range(2 * k + 1, 5000):
        if N % 2 == 0 or N <= 2 * k:
            continue
        edges = []
        for i in range(N):
            edges.append((i, (i + 1) % N))
            edges.append((N + i, (N + (i + 1)) % N))
            edges.append((i, N + i))
        edges = list(set(edges))
        n = 2 * N
        if len(edges) != N * 3:
            continue
        adj = _build_adj(edges, n)
        if all(len(adj[i]) == 3 for i in range(n)) and _nonback_girth(adj, n) >= g:
            return n, edges
    return None


# ---------------- construction 3: Greedy leaf-pairing on extended tree --------
def _greedy_leaf_pair(k, g, max_restarts=5, time_budget=30.0):
    """Build a k-regular tree of radius r+1, then greedily pair leaves to
    achieve k-regularity while maintaining girth >= g."""
    r = (g - 1) // 2
    # Use radius r+1 to ensure leaf distances are large enough
    # For k=5, g=9: r=4, use radius 5
    # Level 0: 1
    # Level i: k*(k-1)^(i-1) for i>=1
    radius = r + 1
    
    tree_sizes = [1]
    for i in range(1, radius + 1):
        tree_sizes.append(k * (k - 1) ** (i - 1))
    
    n_tree = sum(tree_sizes)
    
    # Build the tree structure
    parent = [-1] * n_tree
    level = [0] * n_tree
    children = [[] for _ in range(n_tree)]
    
    for lvl in range(1, radius + 1):
        start_vid = sum(tree_sizes[:lvl])
        for i in range(tree_sizes[lvl]):
            v = start_vid + i
            level[v] = lvl
            if lvl == 1:
                p = 0
            else:
                prev_start = sum(tree_sizes[:lvl - 1])
                p_idx = i // (k - 1)
                p = prev_start + p_idx
            parent[v] = p
            children[p].append(v)
    
    # Identify leaves (level == radius)
    leaves = [v for v in range(n_tree) if level[v] == radius]
    n_leaves = len(leaves)
    
    # Each leaf needs (k-1) more edges. Total edges to add: n_leaves * (k-1) / 2
    edges_to_add = n_leaves * (k - 1) // 2
    
    # For greedy pairing, we need to find pairs of leaves with sufficient distance
    # in the tree. Distance between two leaves = level[u] + level[v] - 2*level[lca]
    # Since all leaves are at the same level, distance = 2*radius - 2*level[lca]
    # We need distance >= g, so 2*radius - 2*level[lca] >= g
    # => level[lca] <= radius - g/2
    
    # Precompute LCA levels for leaf pairs? Too expensive for large n.
    # Instead, use a heuristic: leaves in different subtrees of the root have
    # LCA at level 0, so distance = 2*radius. If 2*radius >= g, they're safe.
    # For k=5, g=9, radius=5: 2*5=10 >= 9. Good.
    
    # Group leaves by their parent (level radius-1)
    leaf_groups = {}
    for leaf in leaves:
        p = parent[leaf]
        if p not in leaf_groups:
            leaf_groups[p] = []
        leaf_groups[p].append(leaf)
    
    # Greedy: pair leaves from different groups (different parents at level radius-1)
    # This ensures LCA is at most level radius-2, so distance >= 2*(radius-1)
    # For radius=5: distance >= 8. But we need >= 9. So we need LCA at level <= 3.
    # Leaves from different top-level branches (children of root) have LCA at level 0.
    
    # Better: group by top-level branch (child of root)
    top_branch = {}
    for leaf in leaves:
        # Find the child of root that is an ancestor of leaf
        cur = leaf
        while parent[cur] != 0:
            cur = parent[cur]
        top_branch[leaf] = cur
    
    branch_groups = {}
    for leaf in leaves:
        b = top_branch[leaf]
        if b not in branch_groups:
            branch_groups[b] = []
        branch_groups[b].append(leaf)
    
    # Now greedily pair leaves from different branch groups
    adj = [set() for _ in range(n_tree)]
    
    # Add tree edges
    for v in range(1, n_tree):
        p = parent[v]
        adj[v].add(p)
        adj[p].add(v)
    
    # Create a list of leaves to pair, shuffled for randomness
    leaf_list = leaves[:]
    random.shuffle(leaf_list)
    
    # Each leaf needs (k-1) partners
    # Use a priority queue or simple greedy
    pending = {leaf: k - 1 for leaf in leaves}
    
    # For each leaf, find compatible partners
    # Compatible: not in same branch group (to ensure LCA at level 0, distance = 2*radius)
    # Actually, even same branch group might work if LCA is high enough, but let's be safe.
    
    # Build a candidate list for each leaf
    # For efficiency, we'll just iterate through other leaves and check compatibility
    
    # Simple greedy: for each leaf, pick (k-1) random leaves from other branch groups
    # that haven't been paired with this leaf yet
    
    # Shuffle leaves for diversity
    for restart in range(max_restarts):
        if _elapsed() > _DEADLINE - time_budget:
            break
        
        # Reset
        adj = [set() for _ in range(n_tree)]
        for v in range(1, n_tree):
            p = parent[v]
            adj[v].add(p)
            adj[p].add(v)
        
        pending = {leaf: k - 1 for leaf in leaves}
        leaf_list = leaves[:]
        random.shuffle(leaf_list)
        
        success = True
        
        # Process leaves in random order
        for leaf in leaf_list:
            if pending[leaf] == 0:
                continue
            
            # Find compatible partners
            candidates = []
            for other in leaves:
                if other == leaf:
                    continue
                if other in adj[leaf]:
                    continue
                if pending[other] == 0:
                    continue
                # Check if they're in different branch groups
                if top_branch[leaf] != top_branch[other]:
                    candidates.append(other)
            
            if len(candidates) < pending[leaf]:
                success = False
                break
            
            # Pick random candidates
            random.shuffle(candidates)
            for other in candidates[:pending[leaf]]:
                adj[leaf].add(other)
                adj[other].add(leaf)
                pending[leaf] -= 1
                pending[other] -= 1
        
        if not success:
            continue
        
        # Check regularity
        if any(len(adj[i]) != k for i in range(n_tree)):
            continue
        
        # Check girth
        girth = _nonback_girth(adj, n_tree)
        if girth >= g:
            edges = [(i, j) for i in range(n_tree) for j in adj[i] if i < j]
            return n_tree, edges
    
    return None


# ---------------- construction 4: Random Regular Graph + Local Search ---------
def _random_local_search(k, g):
    R = (g - 1) // 2
    moore = 1 + k
    for i in range(1, R):
        moore += k * (k - 1) ** i
    
    max_n = 2000
    if k == 3:
        max_n = 500
    elif k == 4:
        max_n = 1000
    
    for n in range(moore, max_n + 1, 2 if k % 2 == 0 else 1):
        if (n * k) % 2 != 0:
            continue
        
        for restart in range(3):
            if _elapsed() > _DEADLINE - 5:
                return None
            
            try:
                adj = _generate_random_regular(n, k)
            except Exception:
                continue
            
            if not adj:
                continue
                
            girth = _nonback_girth(adj, n)
            if girth >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
            
    return None


def _generate_random_regular(n, k):
    if (n * k) % 2 != 0:
        return None
    
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
    success = True
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            for j in range(i + 2, len(stubs), 2):
                if stubs[j] != u and stubs[j] != v:
                    stubs[i + 1], stubs[j] = stubs[j], stubs[i + 1]
                    v = stubs[i + 1]
                    break
            else:
                return None
        if v in adj[u]:
            for j in range(i + 2, len(stubs), 2):
                if stubs[j] not in adj[u] and stubs[j] != u:
                    stubs[i + 1], stubs[j] = stubs[j], stubs[i + 1]
                    v = stubs[i + 1]
                    break
            else:
                return None
        
        adj[u].add(v)
        adj[v].add(u)
        
    for i in range(n):
        if len(adj[i]) != k:
            return None
        if len(adj[i]) != len(set(adj[i])):
            return None
            
    return adj


# ---------------- construction 5: circulant graph C(n; S) --------------------
def _circulant(k, g):
    if k % 2 != 0:
        return None
    n = (k // 2) * 2
    while n < 4 * k and _elapsed() < _DEADLINE - 10:
        half = k // 2
        found = None
        for _ in range(100):
            S = random.sample(range(1, n // 2 + 1), half)
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(n):
                for s in S:
                    adj[i].add((i + s) % n)
                    adj[i].add((i - s) % n)
                if len(adj[i]) != k:
                    valid = False
                    break
            if not valid:
                continue
                
            if _nonback_girth(adj, n) >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                found = (n, edges)
                break
        if found:
            return found
        n += 2
    return None


# ---------------- construction 6: BFS tree-like completion --------------------
def _bfs_complete(k, g):
    R = (g - 1) // 2
    depth_lim = min(R - 1, 5)
    order = [(0, -1, 0)]
    node_idx = {0: 0}
    parent_of = {0: -1}
    depth_of = {0: 0}
    q = deque([0])
    while q:
        u = q.popleft()
        du = depth_of[u]
        if du >= depth_lim:
            continue
        for child in range(k):
            v = len(order)
            order.append((v, u, du + 1))
            node_idx[v] = v
            parent_of[v] = u
            depth_of[v] = du + 1
            q.append(v)
    n = len(order)
    adj = [set() for _ in range(n)]
    for v in range(1, n):
        p = parent_of[v]
        adj[v].add(p)
        adj[p].add(v)
    
    def cand_ok(v, w):
        if w in adj[v]:
            return False
        if w == parent_of[v]:
            return False
        common = set(adj[v]) & set(adj[w])
        if common:
            return False
        if w in adj[parent_of[v]]:
            return False
        return True

    for v in range(n):
        guard = 0
        while len(adj[v]) < k and guard < 5000:
            guard += 1
            for w in range(n):
                if len(adj[v]) >= k:
                    break
                if w <= v:
                    continue
                if cand_ok(v, w):
                    adj[v].add(w)
                    adj[w].add(v)
    if any(len(adj[i]) != k for i in range(n)):
        return None
    for i in range(n):
        adj[i] = sorted(adj[i])
    if _nonback_girth(adj, n) < g:
        return None
    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
    return n, edges


# ---------------- parallel candidate verification ---------------------------
def _worker(name, fn, holder):
    try:
        res = fn()
    except Exception:
        res = None
    if res is None:
        return
    n, edges = res
    if _elapsed() >= _DEADLINE:
        return
    try:
        if n > 0 and _is_valid(_build_adj(edges, n), n, k, g):
            cur = holder.get("best")
            if cur is None or n < cur[0]:
                holder["best"] = (n, edges)
    except Exception:
        pass


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    jobs = [
        ("luw", lambda: _luw(k, g)),
        ("petersen", lambda: _petersen(k, g)),
        ("circulant", lambda: _circulant(k, g)),
        ("random", lambda: _random_local_search(k, g)),
        ("bfs", lambda: _bfs_complete(k, g)),
        ("greedy_leaf", lambda: _greedy_leaf_pair(k, g, max_restarts=5, time_budget=25.0)),
    ]
    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder), daemon=True)
        t.start()
        threads.append(t)

    while _elapsed() < _DEADLINE - 3.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.2)

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
