"""Strong edge colouring: ordered greedy + post-hoc class merging, with deadline guard."""
import time
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin under 150 s kill

    def gen_instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def strong_colour(edges, adj, order):
        m = len(edges)
        # index of each edge in sorted list
        pos = {e: i for i, e in enumerate(edges)}
        v2e = {}
        for i, (u, v) in enumerate(edges):
            v2e.setdefault(u, []).append(i)
            v2e.setdefault(v, []).append(i)
        # conflict mask: edge i cannot share a colour with edge j if they share a vertex
        # or are joined by an edge (i.e. some endpoint of i is adjacent to some endpoint of j)
        # Represent per-edge "forbidden vertices" = endpoints of e plus all neighbours of endpoints
        forbid = []
        for (u, v) in edges:
            s = {u, v}
            s.update(adj.get(u, ()))
            s.update(adj.get(v, ()))
            forbid.append(s)
        # colour[i] = colour index, or -1
        colour = [-1] * m
        # classes: list of lists of edge indices
        classes = []
        # class_forbid[c] = set of vertices used (endpoints) in class c
        class_forbid = []
        # class_edge_mask not needed; we track per-class the forbid-union quickly via class_forbid
        # An edge i is compatible with class c iff forbid[i] is disjoint from class_forbid[c]
        #   AND no edge in class c is "joined" to i beyond vertex sharing.
        # Actually induced matching condition: no two edges share a vertex OR are joined by an edge.
        # "Joined by an edge" means: exists endpoint x of edge A and endpoint y of edge B with (x,y) in E.
        # This is exactly: forbid[i] (endpoints of i + neighbours) intersects endpoints of any edge in class.
        # Since class_forbid[c] = union of endpoints of all edges in class c, and forbid[i] includes
        # endpoints of i and their neighbours, the condition "forbid[i] disjoint from class_forbid[c]"
        # ensures: no shared vertex (endpoint of i not in class_forbid) and no joining edge
        # (neighbour of endpoint of i not in class_forbid). Correct.
        for i in order:
            fi = forbid[i]
            best_c = -1
            best_size = -1
            for c in range(len(classes)):
                if fi.isdisjoint(class_forbid[c]):
                # compatible
                    if len(classes[c]) > best_size:
                        best_size = len(classes[c])
                        best_c = c
            if best_c >= 0:
                colour[i] = best_c
                classes[best_c].append(i)
                class_forbid[best_c] |= {edges[i][0], edges[i][1]}
            else:
                c = len(classes)
                colour[i] = c
                classes.append([i])
                class_forbid.append({edges[i][0], edges[i][1]})
        # renumber colours to 0..k-1 compactly (they already are, since we append)
        return colour, len(classes)

    def merge_pass(edges, colour, k):
        # Try to merge two classes if their union is still an induced matching.
        # Build class_forbid from current colouring.
        m = len(edges)
        class_forbid = []
        for c in range(k):
            s = set()
            for i in range(m):
                if colour[i] == c:
                    s.add(edges[i][0]); s.add(edges[i][1])
            class_forbid.append(s)
        # Also need per-class forbid (endpoints + neighbours) to test compatibility of a whole class.
        # For merging class a into b: need every edge in a compatible with every edge in b,
        # equivalently: forbid_union(a) disjoint from endpoints_union(b) AND forbid_union(b) disjoint from endpoints_union(a).
        # forbid_union(c) = union over edges in c of (endpoints + neighbours).
        adj = {}
        for (u, v) in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        def class_forbid_union(c):
            s = set()
            for i in range(m):
                if colour[i] == c:
                    u, v = edges[i]
                    s.add(u); s.add(v)
                    s.update(adj.get(u, ()))
                    s.update(adj.get(v, ()))
            return s
        changed = True
        while changed:
            changed = False
            # recompute unions
            forb_union = [class_forbid_union(c) for c in range(k)]
            for a in range(k):
                if a >= k: continue
                for b in range(a + 1, k):
                    # can we merge a into b?
                    if forb_union[a].isdisjoint(class_forbid[b]) and forb_union[b].isdisjoint(class_forbid[a]):
                        # merge b into a
                        for i in range(m):
                            if colour[i] == b:
                                colour[i] = a
                        # remove b
                        colour[b] = -2  # mark removed
                        k -= 1
                        changed = True
                        break
                if changed:
                    break
        # renumber
        mapping = {}
        newk = 0
        for c in range(len(colour)):
            if colour[c] == -2:
                continue
            if colour[c] not in mapping:
                mapping[colour[c]] = newk
                newk += 1
        for c in range(len(colour)):
            if colour[c] != -2:
                colour[c] = mapping[colour[c]]
        return colour, newk

    def make_order(edges, adj, mode, rng):
        m = len(edges)
        idx = list(range(m))
        if mode == 0:
            # degree of endpoints descending
            deg = {}
            for (u, v) in edges:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            idx.sort(key=lambda i: (-(deg[edges[i][0]] + deg[edges[i][1]]), i))
        elif mode == 1:
            # random
            rng.shuffle(idx)
        elif mode == 2:
            # ascending degree (process low-degree first)
            deg = {}
            for (u, v) in edges:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            idx.sort(key=lambda i: (deg[edges[i][0]] + deg[edges[i][1]], i))
        elif mode == 3:
            # descending product of endpoint degrees
            deg = {}
            for (u, v) in edges:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            idx.sort(key=lambda i: (-(deg[edges[i][0]] * max(1, deg[edges[i][1]])), i))
        else:
            idx.sort(key=lambda i: i)
        return idx

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = gen_instance(n, d, seed)
        adj = {}
        for (u, v) in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        m = len(edges)
        best_k = m + 1
        best_colour = None
        rng = Random(12345 + seed)
        # Try a set of (mode, restart) combinations
        attempts = []
        for mode in (0, 3, 2, 1):
            for restart in range(4):
                attempts.append((mode, restart))
        # Add a few pure random
        for r in range(6):
            attempts.append((1, 100 + r))
        for (mode, restart) in attempts:
            if time.time() > DEADLINE - 8.0:
                break
            r2 = Random(seed * 1000 + mode * 17 + restart)
            order = make_order(edges, adj, mode, r2)
            colour, k = strong_colour(edges, adj, order)
            colour, k = merge_pass(edges, colour, k)
            if k < best_k:
                best_k = k
                best_colour = list(colour)
        # Fallback: ensure we always have a valid colouring
        if best_colour is None:
            order = list(range(m))
            best_colour, best_k = strong_colour(edges, adj, order)
            best_colour, best_k = merge_pass(edges, best_colour, best_k)
        results.append(best_colour)
        if time.time() > DEADLINE - 5.0:
            # If over, just finish with what we have for remaining (use quick greedy)
            pass
    return results
