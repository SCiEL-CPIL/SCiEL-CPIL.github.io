import random
import time

N = 20
C = 20
DEADLINE = 95.0  # safety margin well before the 150 s wall-clock kill


def construct():
    start = time.time()
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    best_edges = None
    best_count = -1

    def try_build_c4free(rng):
        """Guaranteed-valid floor: C4-free graph, all edges colour 0."""
        ec = [[-1] * N for _ in range(N)]
        adj = [set() for _ in range(N)]
        pairs = all_pairs[:]
        rng.shuffle(pairs)
        for (u, v) in pairs:
            ok = True
            for x in adj[v]:
                if x == u or x == v:
                    continue
                for y in adj[u]:
                    if y == v or y == u or y == x:
                        continue
                    if y in adj[x]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                ec[u][v] = ec[v][u] = 0
                adj[u].add(v)
                adj[v].add(u)
        edges = []
        for i in range(N):
            row = ec[i]
            for j in range(i + 1, N):
                if row[j] >= 0:
                    edges.append([i, j, row[j]])
        return edges

    def try_build(rng):
        """Randomised greedy maintaining the rainbow-4-cycle invariant."""
        ec = [[-1] * N for _ in range(N)]
        adj = [set() for _ in range(N)]
        pairs = all_pairs[:]
        rng.shuffle(pairs)
        for (u, v) in pairs:
            # collect colour triples of every existing 4-cycle through (u,v): u-v-x-y-u
            triples = []
            for x in adj[v]:
                if x == u or x == v:
                    continue
                for y in adj[u]:
                    if y == v or y == u or y == x:
                        continue
                    if y in adj[x]:
                        triples.append((ec[v][x], ec[x][y], ec[y][u]))
            valid = []
            for c in range(C):
                ok = True
                for (c1, c2, c3) in triples:
                    if c1 == c2 or c1 == c3 or c2 == c3:
                        ok = False
                        break
                    if c == c1 or c == c2 or c == c3:
                        ok = False
                        break
                if ok:
                    valid.append(c)
            if valid:
                c = rng.choice(valid)
                ec[u][v] = ec[v][u] = c
                adj[u].add(v)
                adj[v].add(u)
        edges = []
        for i in range(N):
            row = ec[i]
            for j in range(i + 1, N):
                if row[j] >= 0:
                    edges.append([i, j, row[j]])
        return edges

    def verify(edges):
        """Full check: simple graph + every 4-cycle rainbow (all 3 cyclic orders per 4-set)."""
        ec = [[-1] * N for _ in range(N)]
        for (u, v, c) in edges:
            if ec[u][v] != -1:
                return False
            ec[u][v] = ec[v][u] = c
        for a in range(N):
            for b in range(a + 1, N):
                for c in range(b + 1, N):
                    for d in range(c + 1, N):
                        if (ec[a][b] >= 0 and ec[b][c] >= 0 and ec[c][d] >= 0 and ec[d][a] >= 0
                                and len({ec[a][b], ec[b][c], ec[c][d], ec[d][a]}) != 4):
                            return False
                        if (ec[a][b] >= 0 and ec[b][d] >= 0 and ec[d][c] >= 0 and ec[c][a] >= 0
                                and len({ec[a][b], ec[b][d], ec[d][c], ec[c][a]}) != 4):
                            return False
                        if (ec[a][c] >= 0 and ec[c][b] >= 0 and ec[b][d] >= 0 and ec[d][a] >= 0
                                and len({ec[a][c], ec[c][b], ec[b][d], ec[d][a]}) != 4):
                            return False
        return True

    # guaranteed-valid floor
    base = try_build_c4free(random.Random(12345))
    if verify(base):
        best_edges = base
        best_count = len(base)

    master = random.Random(20240517)
    it = 0
    while time.time() - start < DEADLINE:
        rng = random.Random(master.randint(0, 2 ** 31 - 1))
        edges = try_build(rng)
        it += 1
        if len(edges) > best_count and verify(edges):
            best_edges = edges
            best_count = len(edges)
            if best_count >= 170:
                break
        if it % 25 == 0 and time.time() - start > DEADLINE:
            break

    if best_edges is None:
        # last-resort valid graph (a star has no 4-cycles)
        return [[0, v, 0] for v in range(1, N)]
    return best_edges
