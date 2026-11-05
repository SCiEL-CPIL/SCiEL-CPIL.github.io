import random
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def time_ok():
        return time.time() < deadline

    def count_k4(adj, n):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                common = adj[i] & adj[j]
                cm = common
                while cm:
                    k = (cm & (-cm)).bit_length() - 1
                    cm &= cm - 1
                    if k <= j:
                        continue
                    lmask = common & (~((1 << (k + 1)) - 1))
                    count += (adj[k] & lmask).bit_count()
        return count // 6

    def count_ind6(adj, n):
        count = 0
        for combo in itertools.combinations(range(n), 6):
            ok = True
            for a in range(6):
                u = combo[a]
                for b in range(a + 1, 6):
                    v = combo[b]
                    if (adj[u] >> v) & 1:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                count += 1
        return count

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    best_n = 0
    best_edges = None

    for n in range(18, 27):
        if not time_ok():
            break
        if n <= best_n:
            continue

        found = False
        n_budget = 25.0
        n_start = time.time()

        for restart in range(30):
            if not time_ok():
                break
            if time.time() - n_start > n_budget:
                break

            adj = [0] * n
            p = 0.4 + random.random() * 0.15
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            best_v = float('inf')
            best_adj = adj[:]

            max_iter = 100
            for it in range(max_iter):
                if not time_ok():
                    break
                if time.time() - n_start > n_budget:
                    break

                k4 = count_k4(adj, n)
                i6 = count_ind6(adj, n)
                v = k4 + i6

                if v < best_v:
                    best_v = v
                    best_adj = adj[:]
                    if v == 0:
                        found = True
                        break

                if v == 0:
                    found = True
                    break

                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i

                adj[i] ^= (1 << j)
                adj[j] ^= (1 << i)

                nk4 = count_k4(adj, n)
                ni6 = count_ind6(adj, n)
                nv = nk4 + ni6

                if nv >= v:
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)

            if found:
                edges = adj_to_edges(best_adj, n)
                best_n = n
                best_edges = edges
                break

    if best_n < 17:
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
