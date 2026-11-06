"""Decompose three fixed random graphs into edge-disjoint cycles and edges, minimizing pieces."""
import time
from random import Random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin under 150s wall-clock

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def edge_set(cycle):
        s = set()
        for i in range(len(cycle)):
            a, b = cycle[i], cycle[(i + 1) % len(cycle)]
            s.add((a, b) if a < b else (b, a))
        return s

    def greedy_decompose(edges, deadline):
        """Repeatedly extract a simple cycle via DFS; leftovers become single edges."""
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        pieces = []

        while remaining and time.time() < deadline:
            # vertex degrees within remaining
            deg = defaultdict(int)
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1

            candidates = sorted(deg.keys(), key=lambda x: -deg[x])
            found = None
            found_len = 0

            for start in candidates:
                if deg[start] < 2:
                    continue
                cyc = _find_cycle(start, remaining, adj, 40)
                if cyc and len(cyc) > found_len:
                    found = cyc
                    found_len = len(cyc)
                if found_len >= 12:
                    break

            if found:
                for e in edge_set(found):
                    remaining.discard(e)
                    a, b = e
                    adj[a].discard(b)
                    adj[b].discard(a)
                pieces.append(found)
            else:
                e = next(iter(remaining))
                remaining.discard(e)
                a, b = e
                adj[a].discard(b)
                adj[b].discard(a)
                pieces.append([a, b])

        for e in remaining:
            pieces.append(list(e))
        return pieces

    def _find_cycle(start, remaining, adj, max_depth):
        # iterative DFS to avoid recursion limits; find simple cycle back to start
        # stack entries: (node, iter_state_index)
        neighbors_cache = {}
        for u, v in remaining:
            neighbors_cache.setdefault(u, []).append(v)
            neighbors_cache.setdefault(v, []).append(u)

        path = [start]
        visited = {start}
        # manual DFS with stack of (node, next_index)
        stack = [(start, 0)]
        while stack:
            node, idx = stack.pop()
            nbrs = neighbors_cache.get(node, [])
            placed = False
            for i in range(idx, len(nbrs)):
                nb = nbrs[i]
                if nb == start:
                    if len(path) >= 3:
                        return list(path)
                    else:
                        continue
                if nb not in visited:
                    # push remaining siblings then descend
                    stack.append((node, i + 1))
                    visited.add(nb)
                    path.append(nb)
                    stack.append((nb, 0))
                    placed = True
                    break
            if not placed:
                # backtrack
                if len(path) > 1:
                    visited.discard(node)
                    path.pop()
        return None

    def local_search(pieces, deadline):
        """Splice cycles/edges at shared vertices to reduce piece count."""
        def is_cycle(p):
            return len(p) >= 3

        # adjacency of the union graph
        adj = defaultdict(set)
        for p in pieces:
            for i in range(len(p)):
                a, b = p[i], p[(i + 1) % len(p)]
                adj[a].add(b)
                adj[b].add(a)

        improved = True
        while improved and time.time() < deadline:
            improved = False

            # map vertex -> list of (piece_index, position)
            vpos = defaultdict(list)
            for i, p in enumerate(pieces):
                for j, v in enumerate(p):
                    vpos[v].append((i, j))

            # try splicing pairs of distinct pieces that share a vertex
            for v, lst in vpos.items():
                if len(lst) < 2:
                    continue
                # group by piece
                by_piece = defaultdict(list)
                for i, j in lst:
                    by_piece[i].append(j)
                piece_ids = list(by_piece.keys())
                if len(piece_ids) < 2:
                    continue
                for a in range(len(piece_ids)):
                    for b in range(a + 1, len(piece_ids)):
                        i, j = piece_ids[a], piece_ids[b]
                        P1, P2 = pieces[i], pieces[j]
                        res = try_splice(P1, P2, v, adj)
                        if res is not None:
                            new_piece = res
                            # remove the two old pieces, add new one
                            # keep order: replace i with new_piece, drop j
                            new_pieces = []
                            for k, p in enumerate(pieces):
                                if k == i:
                                    new_pieces.append(new_piece)
                                elif k == j:
                                    continue
                                else:
                                    new_pieces.append(p)
                            pieces = new_pieces
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return pieces

    def try_splice(P1, P2, v, adj):
        """
        Attempt to merge two pieces sharing vertex v into one piece.
        Returns new piece (list of vertices) or None.
        """
        # positions of v in each
        pos1 = P1.index(v)
        pos2 = P2.index(v)

        if len(P1) >= 3 and len(P2) >= 3:
            # cycle/cycle splice:
            # P1: ... a, v, b ... ; P2: ... c, v, d ...
            a = P1[(pos1 - 1) % len(P1)]
            b = P1[(pos1 + 1) % len(P1)]
            c = P2[(pos2 - 1) % len(P2)]
            d = P2[(pos2 + 1) % len(P2)]
            # try two reconnections: (a,c)+(b,d) or (a,d)+(b,c)
            for (x, y, z, w) in ((a, c, b, d), (a, d, b, c)):
                if (x, y) in adj or (y, x) in adj:
                    if (z, w) in adj or (w, z) in adj:
                        # build merged cycle
                        # P1 without v: sequence from a to b going around through ... 
                        # We need the two arcs of P1 between a and b (excluding v)
                        # arc1: a, ..., b (the path in P1 not containing v)
                        # Since v is between a and b in P1, the other arc goes the long way.
                        arc1 = _arc_excluding(P1, a, b, v)
                        arc2 = _arc_excluding(P2, c, d, v)
                        if arc1 is None or arc2 is None:
                            continue
                        # merged: x .. y then z .. w? We reconnect x-y and z-w.
                        # The merged cycle: start at x, follow arc1 to y? 
                        # Actually x is in P1, y in P2. We connect x-y (new edge) and z-w (new edge).
                        # Cycle: x -> (arc1 from x to z) -> z -> w (new) -> (arc2 from w to y) -> y -> x (new)
                        # arc1 from x to z: x is a or b, z is b or a. In P1, the arc not through v.
                        # Let's build: take P1 as a cycle, remove v, it's a path from a to b.
                        # We need path from x to z along that path.
                        p1_path = _path_in_cycle_excluding(P1, v)  # path a..b
                        p2_path = _path_in_cycle_excluding(P2, v)  # path c..d
                        seg1 = _segment(p1_path, x, z)
                        seg2 = _segment(p2_path, w, y)
                        if seg1 is None or seg2 is None:
                            continue
                        merged = seg1 + seg2
                        # verify simple (no repeated vertices) and all consecutive edges exist
                        if _valid_piece(merged, adj):
                            return merged
            return None
        else:
            # at least one is a single edge (len 2)
            # edge piece: [v, w] where w is the other endpoint
            if len(P1) == 2:
                e1, Pbig = P1, P2
            else:
                e1, Pbig = P2, P1
            w = e1[0] if e1[1] == v else e1[1]
            # Pbig is a cycle (len>=3) containing v
            pos = Pbig.index(v)
            a = Pbig[(pos - 1) % len(Pbig)]
            b = Pbig[(pos + 1) % len(Pbig)]
            # splice: replace v in Pbig with w, connecting a-w and w-b if both exist
            if (a, w) in adj or (w, a) in adj:
                if (w, b) in adj or (b, w) in adj:
                    # new cycle: Pbig with v replaced by w
                    new_cycle = []
                    for x in Pbig:
                        if x == v:
                            new_cycle.append(w)
                        else:
                            new_cycle.append(x)
                    if _valid_piece(new_cycle, adj):
                        return new_cycle
            return None

    def _arc_excluding(cycle, x, y, v):
        # return list of vertices on the arc from x to y not passing through v (inclusive)
        # cycle is a list; find positions
        try:
            ix = cycle.index(x)
            iy = cycle.index(y)
        except ValueError:
            return None
        n = len(cycle)
        # two arcs; pick the one not containing v
        arc_a = []
        i = ix
        while True:
            arc_a.append(cycle[i])
            if cycle[i] == y:
                break
            i = (i + 1) % n
            if len(arc_a) > n:
                return None
        if v in arc_a:
            # try other direction
            arc_b = []
            i = ix
            while True:
                arc_b.append(cycle[i])
                if cycle[i] == y:
                    break
                i = (i - 1) % n
                if len(arc_b) > n:
                    return None
            if v in arc_b:
                return None
            return arc_b
        return arc_a

    def _path_in_cycle_excluding(cycle, v):
        # remove v from cycle, return the resulting path (list) in order
        n = len(cycle)
        idx = cycle.index(v)
        path = []
        i = (idx + 1) % n
        for _ in range(n - 1):
            path.append(cycle[i])
            i = (i + 1) % n
            if i == idx:
                break
        return path

    def _segment(path, x, y):
        # return subpath from x to y inclusive, or None
        if x not in path or y not in path:
            return None
        ix = path.index(x)
        iy = path.index(y)
        if ix <= iy:
            return path[ix:iy + 1]
        else:
            return path[ix:] + path[:iy + 1]

    def _valid_piece(piece, adj):
        if len(piece) < 3:
            return False
        if len(set(piece)) != len(piece):
            return False
        for i in range(len(piece)):
            a, b = piece[i], piece[(i + 1) % len(piece)]
            if (a, b) not in adj and (b, a) not in adj:
                return False
        return True

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 10:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        pieces = greedy_decompose(edges, deadline - 5)
        # local search with remaining budget
        if time.time() < deadline - 2:
            pieces = local_search(pieces, deadline - 2)
        results.append(pieces)

    return results
