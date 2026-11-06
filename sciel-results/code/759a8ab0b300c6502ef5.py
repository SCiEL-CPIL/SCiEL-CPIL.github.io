import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    def time_left():
        return deadline - (time.time() - start_time)

    # Bitset adjacency: adj[i] has bit j set iff (i,j) is an edge
    adj = [0] * n

    def can_add(u, v):
        """True if adding edge (u,v) creates no C6.

        A C6 through the new edge has form u-v-x-w-z-y-u, so for some
        w in N(u)\{v}, x in N(v)\{u}, w != x, there is an edge between
        N(w)\{u,v,x} and N(x)\{u,v,w}.  Bitset intersection per (w,x).
        """
        Nu = adj[u]
        Nv = adj[v]
        # iterate over the smaller neighbor set first
        if Nv.bit_count() > Nu.bit_count():
            Nu, Nv = Nv, Nu
        for w in _iter_bits(Nu):
            if w == v:
                continue
            Aw = adj[w]
            for x in _iter_bits(Nv):
                if x == u or x == w:
                    continue
                Ax = adj[x]
                # edge between N(w)\{u,v,x} and N(x)\{u,v,w}?
                if (Aw & Ax & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)) != 0:
                    return False
        return True

    def can_remove(u, v):
        """True if removing edge (u,v) destroys no C6.

        A C6 through edge (u,v) has form u-v-x-w-z-y-u, so for some
        w in N(u)\{v}, x in N(v)\{u}, w != x, there is an edge between
        N(w)\{u,v,x} and N(x)\{u,v,w}.
        """
        Nu = adj[u]
        Nv = adj[v]
        if Nv.bit_count() > Nu.bit_count():
            Nu, Nv = Nv, Nu
        for w in _iter_bits(Nu):
            if w == v:
                continue
            Aw = adj[w]
            for x in _iter_bits(Nv):
                if x == u or x == w:
                    continue
                Ax = adj[x]
                if (Aw & Ax & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)) != 0:
                    return False
        return True

    def swap_ok(u1, v1, u2, v2):
        """True if removing (u1,v1) and adding (u2,v2) leaves the graph C6-free.

        u1,v1 is an edge; u2,v2 is a non-edge.  The only new C6s use the
        added edge; the only C6s destroyed use the removed edge.  We check
        that (a) the added edge creates no C6 and (b) the removed edge
        was not the unique member of any C6.
        """
        # temporarily remove (u1,v1)
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        ok = can_add(u2, v2) and can_remove(u1, v1)
        # restore
        adj[u1] |= (1 << v1)
        adj[v1] |= (1 << u1)
        return ok

    def edge_count():
        return sum(a.bit_count() for a in adj) // 2

    def get_edges():
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edges.append([i, j])
        return edges

    random.seed(42)

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    best_edges = 0
    best_adj = [0] * n

    # ---- Phase 1: greedy construction with a few strategies ----
    for trial in range(40):
        if time_left() < 8:
            break
        adj_try = [0] * n
        ec = 0

        if trial % 2 == 0:
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
        else:
            # prefer pairs whose endpoints currently have low degree
            shuffled = all_pairs[:]
            shuffled.sort(key=lambda p: (
                adj_try[p[0]].bit_count() + adj_try[p[1]].bit_count(),
                random.random()))
            # note: static sort; still a useful heuristic

        for u, v in shuffled:
            if time_left() < 3:
                break
            if can_add(u, v):
                adj_try[u] |= (1 << v)
                adj_try[v] |= (1 << u)
                ec += 1

        if ec > best_edges:
            best_edges = ec
            best_adj = adj_try[:]

    # Fallback: 8 disjoint K5s (80 edges, C6-free)
    if best_edges == 0:
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        best_edges = 80

    # ---- Phase 2: local search (add / remove / swap) ----
    adj = best_adj[:]
    current_edges = best_edges

    # build current edge list and non-edge list for fast sampling
    def edge_list():
        el = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    el.append((i, j))
        return el

    def nonedge_list():
        ne = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if not (mask >> j) & 1:
                    ne.append((i, j))
        return ne

    no_improve = 0
    while time_left() > 2:
        progress = False

        # (a) try to add a random non-edge
        if time_left() > 2:
            ne = nonedge_list()
            random.shuffle(ne)
            for u, v in ne[:200]:
                if time_left() < 2:
                    break
                if can_add(u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
                    progress = True
                    break

        # (b) try a random swap: remove an edge, add a non-edge
        if time_left() > 2:
            el = edge_list()
            ne = nonedge_list()
            random.shuffle(el)
            random.shuffle(ne)
            for u1, v1 in el[:200]:
                if time_left() < 2:
                    break
                for u2, v2 in ne[:200]:
                    # must not be the same edge, and added edge must be a non-edge
                    if (u1, v1) == (u2, v2):
                        continue
                    if adj[u2] & (1 << v2):
                        continue
                    if swap_ok(u1, v1, u2, v2):
                        adj[u1] &= ~(1 << v1)
                        adj[v1] &= ~(1 << u1)
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        progress = True
                        break
                if progress:
                    break

        # (c) if stuck, remove a random edge to escape
        if not progress:
            no_improve += 1
            if no_improve >= 15:
                el = edge_list()
                if el:
                    u, v = random.choice(el)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_edges -= 1
                no_improve = 0
        else:
            no_improve = 0

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

    return get_edges()


def _iter_bits(mask):
    while mask:
        yield (mask & -mask).bit_length() - 1
        mask &= mask - 1
