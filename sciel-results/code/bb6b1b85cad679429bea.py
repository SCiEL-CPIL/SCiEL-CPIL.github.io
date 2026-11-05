"""Combined program: structural approach (complete-graph repair + structured starts)
with spectral/bipartite initialization, fast validity via pair-triple common-neighbor
counts, and a simulated-annealing local search (add / swap / removal) to escape
local optima. Refines the partition using a spectral clustering idea."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 142  # safety margin for 150s limit

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    full_mask = (1 << n) - 1

    # ---- fast validity machinery -------------------------------------------
    # pair_cnt[i,j] = number of common neighbours of i and j
    # triple_cnt[i,j,k] = number of common neighbours of i, j, k  (k>j>i)
    pair_cnt = [[0] * n for _ in range(n)]
    triple_cnt = {}

    def build_counts(adj):
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                cij = ai & adj[j]
                pair_cnt[i][j] = cij.bit_count()
                for k in range(j + 1, n):
                    triple_cnt[(i, j, k)] = (cij & adj[k]).bit_count()

    def is_valid(adj):
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    if (cij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def find_violation(adj):
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    if (cij & adj[k]).bit_count() >= 3:
                        return (i, j, k)
        return None

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def make_valid(adj):
        adj = adj[:]
        for _ in range(5000):
            viol = find_violation(adj)
            if viol is None:
                break
            i, j, k = viol
            cijk = adj[i] & adj[j] & adj[k]
            if cijk == 0:
                break
            v = cijk
            low = v & (-v)
            x = low.bit_length() - 1
            adj[i] &= ~(1 << x)
            adj[x] &= ~(1 << i)
        return adj

    # ---- local search with fast validity + simulated annealing -------------
    def local_search(adj, max_iter=6000, sa=True):
        adj = adj[:]
        build_counts(adj)
        cur_count = count_edges(adj)
        iters = 0
        T = 2.0
        while iters < max_iter and time.time() < deadline - 20:
            if time.time() < deadline - 25:
                T = max(0.05, T * 0.995)
            # 1) try to add an edge
            added = False
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                # check whether adding (u,v) creates a K3,3
                ok = True
                for w in range(n):
                    if w == u or w == v:
                        continue
                    # u and v must share >=3 common neighbours with w after add
                    # common(u,v) after add = common(u,v) before + {w if w adj to both}
                    cu = adj[u] & adj[w]
                    cv = adj[v] & adj[w]
                    shared = cu & cv
                    # w becomes a common neighbour of u,v if adj[w] has u and v
                    if (adj[w] & (1 << u)) and (adj[w] & (1 << v)):
                        shared |= (1 << w)
                    if shared.bit_count() >= 3:
                        # need a third vertex? K3,3 needs 3 on each side.
                        # three vertices {u,v,w} with 3 common neighbours => violation
                        ok = False
                        break
                if ok:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    build_counts(adj)
                    cur_count += 1
                    added = True
                    iters += 1
                    break
            if added:
                continue

            # 2) try a swap: remove one edge, add another
            edges_in = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            edges_out = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]
            if not edges_out:
                break
            random.shuffle(edges_in)
            random.shuffle(edges_out)
            swapped = False
            for e_rem in edges_in[:100]:
                if swapped:
                    break
                for e_add in edges_out[:100]:
                    (u1, v1) = e_rem
                    (u2, v2) = e_add
                    if (u1, v1) == (u2, v2):
                        continue
                    old = (adj[u1], adj[v1], adj[u2], adj[v2])
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    if is_valid(adj):
                        build_counts(adj)
                        swapped = True
                        break
                    else:
                        adj[u1], adj[v1], adj[u2], adj[v2] = old
                if not swapped:
                    continue
            if swapped:
                iters += 1
                continue

            # 3) simulated annealing: accept a removal to escape local optimum
            if sa and edges_in and random.random() < T:
                (u1, v1) = random.choice(edges_in)
                old = (adj[u1], adj[v1])
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                build_counts(adj)
                cur_count -= 1
                iters += 1
                # try to re-add a different edge
                for u, v in all_pairs:
                    if adj[u] & (1 << v):
                        continue
                    old2 = (adj[u], adj[v])
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if is_valid(adj):
                        build_counts(adj)
                        cur_count += 1
                        break
                    else:
                        adj[u], adj[v] = old2
            else:
                break
        return adj

    # ---- spectral / bipartite initialization -------------------------------
    def spectral_seed():
        """Partition vertices by a leading eigenvector of the Laplacian of a
        random dense graph, then build a dense seed respecting the partition."""
        random.seed(777)
        # random dense base graph
        A = [[0] * n for _ in range(n)]
        for i, j in all_pairs:
            if random.random() < 0.6:
                A[i][j] = A[j][i] = 1
        # Laplacian L = D - A
        D = [sum(A[i]) for i in range(n)]
        L = [[D[i] - A[i][j] if i != j else -A[i][j] for j in range(n)] for i in range(n)]
        # power iteration on L for leading eigenvector (L is symmetric PSD)
        x = [1.0] * n
        for _ in range(200):
            y = [sum(L[i][j] * x[j] for j in range(n)) for i in range(n)]
            norm = sum(v * v for v in y) ** 0.5
            if norm < 1e-12:
                break
            x = [v / norm for v in y]
        # partition by sign of (x - mean)
        mean = sum(x) / n
        part = [1 if x[i] >= mean else 0 for i in range(n)]
        # build dense seed: connect within parts sparsely, across parts densely
        # but respect the partition to reduce K3,3
        adj = [0] * n
        for i, j in all_pairs:
            if part[i] != part[j]:
                if random.random() < 0.85:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            else:
                if random.random() < 0.4:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # ---- best solution tracking -------------------------------------------
    best_adj = [0] * n
    best_count = 0
    best_edges = []

    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)

    # Strategy 1: complete graph -> make valid -> improve
    adj = [(full_mask ^ (1 << i)) for i in range(n)]
    adj = make_valid(adj)
    adj = local_search(adj, max_iter=4000)
    update_best(adj)

    # Strategy 2: spectral / bipartite seed
    for trial in range(4):
        if time.time() > deadline - 50:
            break
        random.seed(777 + trial)
        adj = spectral_seed()
        if not is_valid(adj):
            adj = make_valid(adj)
        adj = local_search(adj, max_iter=4000)
        update_best(adj)

    # Strategy 3: circulant structured starts
    for s in range(5, 12):
        if time.time() > deadline - 40:
            break
        for trial in range(3):
            if time.time() > deadline - 40:
                break
            random.seed(42 + s * 10 + trial)
            offsets = set()
            while len(offsets) < s:
                d = random.randint(1, n - 1)
                offsets.add(d)
                offsets.add(n - d)
            offsets.discard(0)
            adj = [0] * n
            for u in range(n):
                for d in offsets:
                    v = (u + d) % n
                    if v > u:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            if not is_valid(adj):
                adj = make_valid(adj)
            adj = local_search(adj, max_iter=3000)
            update_best(adj)

    # Strategy 4: random dense graphs
    for trial in range(8):
        if time.time() > deadline - 25:
            break
        random.seed(1000 + trial * 7)
        adj = [0] * n
        p = 0.5 + trial * 0.05
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        adj = make_valid(adj)
        adj = local_search(adj, max_iter=3000)
        update_best(adj)

    # Strategy 5: final high-precision polish on the best
    if best_edges:
        adj = best_adj[:]
        adj = local_search(adj, max_iter=8000, sa=True)
        update_best(adj)
        for _ in range(4):
            if time.time() > deadline - 5:
                break
            adj = best_adj[:]
            adj = local_search(adj, max_iter=2500, sa=True)
            update_best(adj)

    if not best_edges:
        return [[i, i + 1] for i in range(19)]
    return best_edges
