"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random

# ---- Tree generation (fixed, must match task) ----
def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

_TREES = instance_trees(20, 611)
_N = 20

def _adjacency(trees):
    """Return list of adjacency lists (set of neighbors) for each tree."""
    out = []
    for edges in trees:
        adj = [set() for _ in range(len(edges) + 1 if False else len(set(v for e in edges for v in e)))]
        # simpler: size = k
        k = len(edges) + 1
        adj = [set() for _ in range(k)]
        for a, b in edges:
            adj[a].add(b); adj[b].add(a)
        out.append(adj)
    return out

def _degree_seq(adj):
    return [len(s) for s in adj]

# Precompute
_ADJ = _adjacency(_TREES)
_DEG = _degree_seq(_ADJ)

# Order: embed larger trees first (more constrained)
_ORDER = sorted(range(_N), key=lambda k: -(_DEG[k]))  # index into trees (0..19) => tree k+1

def _try_embed(rng, deadline):
    """
    Backtracking: embed trees in _ORDER.
    State: used_edges set of frozenset({u,v}) for u<v.
    For each tree, do a DFS leaf-peeling embedding.
    """
    N = _N
    used = [False] * (N * N)
    def edge_used(u, v):
        if u > v: u, v = v, u
        return used[u * N + v]
    def mark_edge(u, v):
        if u > v: u, v = v, u
        used[u * N + v] = True

    # remaining degree capacity per host vertex
    host_rem_deg = [N - 1] * N  # 19 each

    def embed_tree(t_idx, root_map, root):
        """
        t_idx: index into _TREES (0..19)
        root_map: dict tree_vertex -> host_vertex, partially filled
        root: tree vertex to start from
        Returns dict tree_vertex->host_vertex if success else None.
        Uses recursion with leaf peeling.
        """
        adj = _ADJ[t_idx]
        k = len(adj)
        # We'll do a DFS that picks an unassigned tree vertex adjacent to an assigned one,
        # and tries to map it to a free host neighbor.
        assigned = {root: root_map[root]}
        # build list of tree vertices in order of increasing "free degree" to prefer leaves
        # We'll use a stack of (tree_vertex, host_vertex) and expand.
        # Simpler: recursive backtracking over tree vertices.
        # Order tree vertices by degree (leaves first) to reduce branching.
        order = sorted(range(k), key=lambda v: len(adj[v]))
        # But root must be first.
        order = [root] + [v for v in order if v != root]

        def dfs(pos):
            if pos == len(order):
                return dict(assigned)
            tv = order[pos]
            # find assigned neighbors
            for nh in list(assigned.keys()):
                if tv in adj[nh]:
                    host_n = assigned[nh]
                    # try host neighbors of host_n that are free and adjacent
                    for h in range(N):
                        if h == host_n: continue
                        if edge_used(host_n, h): continue
                        if h in assigned.values(): continue
                        # degree check: host h must have enough remaining degree
                        # tv needs degree len(adj[tv]) in this tree, but some edges to already-assigned
                        # neighbors are "used" by this tree. Count how many of tv's tree-neighbors
                        # are already assigned (must map to neighbors of h).
                        ok = True
                        cnt_assigned_nb = 0
                        for tnb in adj[tv]:
                            if tnb in assigned:
                                cnt_assigned_nb += 1
                                if assigned[tnb] != h:
                                    # tnb is assigned but not to h; but tv must be adjacent to tnb in tree,
                                    # so host h must be adjacent to assigned[tnb]
                                    if edge_used(h, assigned[tnb]):
                                        ok = False; break
                                    # also h must equal? no, tv maps to h, tnb maps to assigned[tnb],
                                    # edge (tv,tnb) in tree => edge (h, assigned[tnb]) in host must be free
                                    # already checked edge_used.
                        if not ok: continue
                        # remaining degree needed for tv: len(adj[tv]) - cnt_assigned_nb
                        need = len(adj[tv]) - cnt_assigned_nb
                        if host_rem_deg[h] < need: continue
                        # also each assigned neighbor's host edge must be free (checked)
                        # commit
                        assigned[tv] = h
                        # mark edges to assigned neighbors
                        marked = []
                        for tnb in adj[tv]:
                            if tnb in assigned:
                                mark_edge(h, assigned[tnb]); marked.append((h, assigned[tnb]))
                        host_rem_deg[h] -= need
                        # also decrement host_rem_deg for the assigned neighbors? They already
                        # accounted for their own degree when they were placed. Actually host_rem_deg
                        # should reflect remaining free edges incident to host. When we place tv->h and
                        # connect to assigned neighbor nh->host_n, we consume one edge at h and one at host_n.
                        # But host_n's rem_deg was decremented when it was placed? Not exactly. Let's
                        # redefine: host_rem_deg[x] = number of currently-free edges incident to x.
                        # Initially 19. Each time we mark an edge (a,b), decrement both.
                        # So we should decrement host_rem_deg for both endpoints when marking.
                        # Let's fix: when we mark edge (h, host_n), decrement both.
                        # We'll restructure to decrement on mark.
                        res = dfs(pos + 1)
                        if res is not None:
                            return res
                        # undo
                        for (a, b) in marked:
                            if a > b: a, b = b, a
                            used[a * N + b] = False
                            host_rem_deg[a] += 1
                            host_rem_deg[b] += 1
                        host_rem_deg[h] += need  # this is double counting; see note
                        del assigned[tv]
                    # end for h
            return None
        # NOTE: the host_rem_deg accounting above is inconsistent. Let me simplify by NOT using
        # host_rem_deg as a separate constraint and instead just check edge_used and that the host
        # vertex has enough free neighbors. Given small N, we can just check free neighbor count.
        # I'll rewrite embed_tree more cleanly below.
        return None

    # Given complexity, use a simpler but effective heuristic:
    # Greedy randomized embedding with backtracking limited.
    # For each tree, try to embed via BFS from a random root, choosing host neighbors randomly
    # among free ones, with backtracking.
    def embed_tree_simple(t_idx, attempts=2000):
        adj = _ADJ[t_idx]
        k = len(adj)
        for _ in range(attempts):
            if time.time() > deadline: return None
            # random root
            root = rng.randrange(k)
            # BFS order
            assigned = {root: rng.randrange(N)}
            # order: BFS
            from collections import deque
            q = deque([root])
            order = [root]
            seen = {root}
            while q:
                v = q.popleft()
                for nb in sorted(adj[v], key=lambda x: rng.random()):
                    if nb not in seen:
                        seen.add(nb); order.append(nb); q.append(nb)
            # now assign in order
            def rec(pos):
                if pos == len(order):
                    return dict(assigned)
                tv = order[pos]
                # find a parent (assigned neighbor)
                parents = [p for p in adj[tv] if p in assigned]
                if not parents:
                    return None
                # candidate hosts: neighbors of assigned parents, free, not used
                cands = set()
                for p in parents:
                    hp = assigned[p]
                    for h in range(N):
                        if h == hp: continue
                        if edge_used(hp, h): continue
                        if h in assigned.values(): continue
                        cands.add(h)
                # also tv must connect to ALL its assigned neighbors via free edges
                # filter: for h in cands, for each assigned neighbor p of tv, edge (h, assigned[p]) must be free
                good = []
                for h in cands:
                    ok = True
                    for p in parents:
                        if edge_used(h, assigned[p]):
                            ok = False; break
                    if ok:
                        # also ensure h not equal to any assigned value
                        good.append(h)
                rng.shuffle(good)
                for h in good:
                    assigned[tv] = h
                    marked = []
                    for p in parents:
                        mark_edge(h, assigned[p]); marked.append((h, assigned[p]))
                    res = rec(pos + 1)
                    if res is not None:
                        return res
                    for (a, b) in marked:
                        if a > b: a, b = b, a
                        used[a * N + b] = False
                    del assigned[tv]
                return None
            r = rec(1)
            if r is not None:
                return r
        return None

    # Main loop
    result = [None] * _N
    for t_idx in _ORDER:
        if time.time() > deadline:
            return None
        r = embed_tree_simple(t_idx)
        if r is None:
            return None
        # mark all edges of this tree as used (they were marked during recursion)
        result[t_idx] = r
    # Build final mapping: for tree k (1-indexed), entry is list of host vertices for tree vertices 0..k-1
    out = [None] * _N
    for t_idx in range(_N):
        k = t_idx + 1
        if k == 1:
            out[0] = None
        else:
            r = result[t_idx]
            out[t_idx] = [r[i] for i in range(k)]
    return out

def construct():
    deadline = time.time() + 145  # safety margin
    best = [None] * 19 + [list(range(20))]  # fallback: only T20 identity
    # Try randomized restarts
    for seed in range(100000):
        if time.time() > deadline - 5:
            break
        rng = Random(seed)
        r = _try_embed(rng, deadline)
        if r is not None:
            # validate
            ok = True
            used = set()
            for k in range(1, _N + 1):
                entry = r[k - 1]
                if k == 1:
                    if entry is not None: ok = False; break
                    continue
                if not isinstance(entry, (list, tuple)): ok = False; break
                if len(entry) != k: ok = False; break
                if len(set(entry)) != k: ok = False; break
                for v in entry:
                    if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v >= _N:
                        ok = False; break
                if not ok: break
                # edges
                for a, b in _TREES[k - 1]:
                    u, w = entry[a], entry[b]
                    if u == w: ok = False; break
                    key = (min(u, w), max(u, w))
                    if key in used: ok = False; break
                    used.add(key)
                if not ok: break
            if ok:
                return [list(e) if e is not None else None for e in r]
    return best
# EVOLVE-BLOCK-END
