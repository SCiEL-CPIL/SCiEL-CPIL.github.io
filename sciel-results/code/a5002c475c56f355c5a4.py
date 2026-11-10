"""Edge path cover via greedy path extension with simplicity guard."""
import time

def construct():
    deadline = time.time() + 140
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        m = len(edges)
        if m == 0:
            return []
        from collections import defaultdict
        nbr = defaultdict(list)
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, d = edges[j]
                if a == c or a == d or b == c or b == d:
                    shared = a if (a == c or a == d) else (b if (b == c or b == d) else None)
                    nbr[i].append((j, shared))
                    nbr[j].append((i, shared))
        used = [False] * m
        paths = []
        while True:
            start = -1
            best_deg = 10**9
            for i in range(m):
                if not used[i]:
                    if len(nbr[i]) < best_deg:
                        best_deg = len(nbr[i])
                        start = i
            if start == -1:
                break
            a0, b0 = edges[start]
            left_v, right_v = a0, b0
            path_edges = [start]
            used[start] = True
            visited = {a0, b0}
            # grow right
            while True:
                cur = path_edges[-1]
                nxt = -1
                for (j, shared) in nbr[cur]:
                    if shared == right_v and not used[j]:
                        na, nb = edges[j]
                        newv = na if na != right_v else nb
                        if newv not in visited:
                            nxt = j
                            break
                if nxt == -1:
                    break
                used[nxt] = True
                path_edges.append(nxt)
                na, nb = edges[nxt]
                newv = na if na != right_v else nb
                visited.add(newv)
                right_v = newv
            # grow left
            while True:
                cur = path_edges[0]
                nxt = -1
                for (j, shared) in nbr[cur]:
                    if shared == left_v and not used[j]:
                        na, nb = edges[j]
                        newv = na if na != left_v else nb
                        if newv not in visited:
                            nxt = j
                            break
                if nxt == -1:
                    break
                used[nxt] = True
                path_edges.insert(0, nxt)
                na, nb = edges[nxt]
                newv = na if na != left_v else nb
                visited.add(newv)
                left_v = newv
            verts = []
            for k, ei in enumerate(path_edges):
                a, b = edges[ei]
                if k == 0:
                    verts = [a, b]
                else:
                    last = verts[-1]
                    verts.append(a if a != last else b)
            paths.append(verts)
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            e = instance(*spec)
            results.append([list(ed) for ed in e])
            continue
        e = instance(*spec)
        results.append(decompose(e))
    return results
