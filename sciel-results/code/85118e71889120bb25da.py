import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130
    n = 18

    def is_k4_free(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if adj[a][b]:
                    for c in range(b + 1, n):
                        if adj[a][c] and adj[b][c]:
                            for d in range(c + 1, n):
                                if adj[a][d] and adj[b][d] and adj[c][d]:
                                    return False
        return True

    def max_triangle_free(adj):
        adj_mask = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and adj[i][j]:
                    adj_mask[i] |= (1 << j)
        best = [0]

        def bb(S_mask, C_mask):
            if S_mask.bit_count() + C_mask.bit_count() <= best[0]:
                return
            if C_mask == 0:
                if S_mask.bit_count() > best[0]:
                    best[0] = S_mask.bit_count()
                return
            v = -1
            min_nb = n + 1
            cm = C_mask
            while cm:
                lb = cm & (-cm)
                u = lb.bit_length() - 1
                nb = (adj_mask[u] & C_mask).bit_count()
                if nb < min_nb:
                    min_nb = nb
                    v = u
                cm ^= lb
            if v == -1:
                return
            bb(S_mask, C_mask & ~(1 << v))
            S_adj_v = S_mask & adj_mask[v]
            forbidden = 0
            sm = S_adj_v
            while sm:
                lb = sm & (-sm)
                w = lb.bit_length() - 1
                forbidden |= (adj_mask[w] & adj_mask[v])
                sm ^= lb
            new_C = C_mask & ~(1 << v) & ~forbidden
            bb(S_mask | (1 << v), new_C)

        bb(0, (1 << n) - 1)
        return best[0]

    def generate_random_k4_free():
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.4:
                    adj[i][j] = adj[j][i] = True
        changed = True
        while changed:
            changed = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not adj[a][b]:
                        continue
                    for c in range(b + 1, n):
                        if not (adj[a][c] and adj[b][c]):
                            continue
                        for d in range(c + 1, n):
                            if adj[a][d] and adj[b][d] and adj[c][d]:
                                edges_in_k4 = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
                                e = random.choice(edges_in_k4)
                                adj[e[0]][e[1]] = False
                                adj[e[1]][e[0]] = False
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        return adj

    def edges_from_adj(adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i][j]]

    # Seed: K6,6,6
    seed_adj = [[False] * n for _ in range(n)]
    for i in range(6):
        for j in range(6, 12):
            seed_adj[i][j] = seed_adj[j][i] = True
        for j in range(12, 18):
            seed_adj[i][j] = seed_adj[j][i] = True
    for i in range(6, 12):
        for j in range(12, 18):
            seed_adj[i][j] = seed_adj[j][i] = True

    best_score = max_triangle_free(seed_adj)
    best_adj = seed_adj

    while time.time() < deadline:
        adj = generate_random_k4_free()
        s = max_triangle_free(adj)
        if s < best_score:
            best_score = s
            best_adj = adj

    return edges_from_adj(best_adj)
