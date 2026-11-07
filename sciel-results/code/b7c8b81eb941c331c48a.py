"""Clique partitions of split graphs: greedy maximal-neighborhood cover + local refinement."""
import time
import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve(n, k, p, seed, deadline):
        rng = random.Random(seed)
        C = list(range(k))
        I = list(range(k, n))
        N = [set() for _ in range(n)]
        for v in I:
            for u in C:
                if rng.random() < p:
                    N[v].add(u)

        remaining = set()
        for v in I:
            for u in N[v]:
                remaining.add((u, v))
        for u in C:
            for w in C:
                if u < w:
                    remaining.add((u, w))

        def find_clique(rem):
            if not rem:
                return None
            a, b = min(rem)
            clique = {a, b}
            for x in sorted((N[a] & N[b]) | {a, b}):
                if x in clique:
                    continue
                ok = True
                for y in clique:
                    if x == y:
                        continue
                    if (min(x, y), max(x, y)) not in rem:
                        ok = False
                        break
                if ok:
                    clique.add(x)
            return clique

        cliques = []
        while remaining:
            if time.time() > deadline:
                break
            cl = find_clique(remaining)
            if cl is None or len(cl) < 2:
                a, b = min(remaining)
                cl = {a, b}
            cliques.append(sorted(cl))
            for e in combinations(sorted(cl), 2):
                remaining.discard(e)

        def is_clique(vertices, cliq):
            vs = set(vertices)
            for a, b in combinations(sorted(vs), 2):
                if a == b:
                    continue
                found = False
                for cl in cliq:
                    if a in cl and b in cl:
                        found = True
                        break
                if not found:
                    return False
            return True

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(cliques)):
                for j in range(i + 1, len(cliques)):
                    union = set(cliques[i]) | set(cliques[j])
                    if len(union) < 2:
                        continue
                    if is_clique(union, cliques):
                        cliques[i] = sorted(union)
                        cliques.pop(j)
                        improved = True
                        break
                if improved:
                    break

        cliques = [c for c in cliques if len(c) >= 2]
        return cliques

    result = []
    start_time = time.time()
    deadline = start_time + 140.0

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        cliques = solve(*spec, deadline)
        result.append(cliques)

    while len(result) < 3:
        n, k, p, seed = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))[len(result)]
        rng = random.Random(seed)
        cliques = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cliques.append([u, v])
        for u in range(k):
            for w in range(u + 1, k):
                cliques.append([u, w])
        result.append(cliques)

    return result[:3]
