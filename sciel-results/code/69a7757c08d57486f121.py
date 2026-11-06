import time
import random

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 145.0

    def popcount(x):
        return bin(x).count('1')

    def compute_all_books(adj):
        books = {}
        for i in range(n):
            m = adj[i]
            while m:
                lsb = m & (-m)
                j = lsb.bit_length() - 1
                m ^= lsb
                if j > i:
                    books[(i, j)] = popcount(adj[i] & adj[j])
        return books

    best_max = 999
    best_adj = None

    random.seed(42)

    def make_init(trial):
        if trial == 0:
            adj = [0] * n
            for a in range(20):
                for b in range(20, 40):
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
            adj[0] |= (1 << 1)
            adj[1] |= (1 << 0)
            return adj
        else:
            random.seed(trial * 137 + 7)
            adj = [0] * n
            for a in range(20):
                for b in range(20, 40):
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
            num_swaps = 30 + trial * 15
            for _ in range(num_swaps):
                edges = []
                for i in range(n):
                    m = adj[i]
                    while m:
                        lsb = m & (-m)
                        j = lsb.bit_length() - 1
                        m ^= lsb
                        if j > i:
                            edges.append((i, j))
                if not edges:
                    break
                ru, rv = random.choice(edges)
                non_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not (adj[i] >> j) & 1:
                            non_edges.append((i, j))
                if not non_edges:
                    break
                nu, nv = random.choice(non_edges)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                adj[nu] |= (1 << nv)
                adj[nv] |= (1 << nu)
            return adj

    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline:
            break

        adj = make_init(trial)
        books = compute_all_books(adj)
        current_max = max(books.values()) if books else 0

        if current_max < best_max:
            best_max = current_max
            best_adj = adj[:]

        if current_max <= 7:
            continue

        no_improve = 0
        max_no_improve = 300

        while time.time() < deadline and no_improve < max_no_improve:
            max_edge = max(books, key=books.get)
            max_val = books[max_edge]

            if max_val <= 7:
                break

            rem_u, rem_v = max_edge
            improved = False

            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] >> j) & 1:
                        non_edges.append((i, j))

            if not non_edges:
                break

            random.shuffle(non_edges)

            for nu, nv in non_edges[:80]:
                if nu == rem_u and nv == rem_v:
                    continue

                adj[rem_u] &= ~(1 << rem_v)
                adj[rem_v] &= ~(1 << rem_u)
                adj[nu] |= (1 << nv)
                adj[nv] |= (1 << nu)

                new_books = compute_all_books(adj)
                new_max = max(new_books.values()) if new_books else 0

                if new_max < current_max:
                    current_max = new_max
                    books = new_books
                    improved = True
                    if current_max < best_max:
                        best_max = current_max
                        best_adj = adj[:]
                    no_improve = 0
                    break
                else:
                    adj[nu] &= ~(1 << nv)
                    adj[nv] &= ~(1 << nu)
                    adj[rem_u] |= (1 << rem_v)
                    adj[rem_v] |= (1 << rem_u)

            if not improved:
                no_improve += 1

    if best_adj is None:
        adj = [0] * n
        for a in range(20):
            for b in range(20, 40):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        best_adj = adj

    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])

    assert len(result) == target_edges, f"Expected {target_edges} edges, got {len(result)}"

    return result
