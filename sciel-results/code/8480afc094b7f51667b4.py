import random
import time

def _find_k4(adj):
    n = len(adj)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    c ^= lsb
                    if adj[w] & common:
                        cw = adj[w] & common
                        lsb2 = cw & -cw
                        x = lsb2.bit_length() - 1
                        return (u, v, w, x)
    return None

def _find_indep6(adj):
    n = len(adj)
    mask = (1 << n) - 1

    def dfs(candidates, size):
        if size == 0:
            return []
        if candidates.bit_count() < size:
            return None
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            comp_neigh = (~adj[v]) & mask & ~(1 << v)
            new_cands = candidates & comp_neigh
            res = dfs(new_cands, size - 1)
            if res is not None:
                return [v] + res
        return None

    return dfs(mask, 6)

def _is_valid(adj):
    return _find_k4(adj) is None and _find_indep6(adj) is None

def _blowup(base_adj, k, n):
    """Create a k-blow-up of base graph, padded/truncated to n vertices."""
    m = len(base_adj)
    adj = [0] * n
    # Assign each vertex to a base vertex: vertex i -> base vertex i % m
    # Within each group, make a clique
    for i in range(n):
        for j in range(i + 1, n):
            bi = i % m
            bj = j % m
            if bi == bj:
                # Same group: always connected (clique)
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            elif base_adj[bi] & (1 << bj):
                # Connected in base: connect
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _random_graph(n, p=0.5):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _local_repair(adj, time_limit):
    """Local search: repair K4s and independent 6-sets."""
    n = len(adj)
    start = time.time()
    steps = 0
    perturb_every = 80

    while time.time() - start < time_limit:
        if steps > 30000:
            break

        k4 = _find_k4(adj)
        if k4:
            u, v, w, x = k4
            edges = [(u, v), (u, w), (u, x), (v, w), (v, x), (w, x)]
            ru, rv = random.choice(edges)
            adj[ru] &= ~(1 << rv)
            adj[rv] &= ~(1 << ru)
        else:
            indep = _find_indep6(adj)
            if indep:
                i, j = random.sample(indep, 2)
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            else:
                return True

        steps += 1

        if steps % perturb_every == 0:
            for _ in range(3):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    if adj[u] & (1 << v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

    return _is_valid(adj)

def _make_base_graphs():
    """Generate small valid base graphs (K4-free, alpha <= 3)."""
    bases = []

    # C5: K4-free, alpha=2
    c5 = [0] * 5
    for i in range(5):
        c5[i] |= (1 << ((i + 1) % 5))
        c5[i] |= (1 << ((i - 1) % 5))
    bases.append(c5)

    # C7: K4-free, alpha=3
    c7 = [0] * 7
    for i in range(7):
        c7[i] |= (1 << ((i + 1) % 7))
        c7[i] |= (1 << ((i - 1) % 7))
    bases.append(c7)

    # Paley 17: K4-free, alpha=4 (verified: no K4, no independent 4)
    squares = {x * x % 17 for x in range(1, 17)}
    p17 = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                p17[i] |= (1 << j)
                p17[j] |= (1 << i)
    bases.append(p17)

    # Complement of C5 blown up: use a 6-vertex graph
    # Take C6: K4-free, alpha=3
    c6 = [0] * 6
    for i in range(6):
        c6[i] |= (1 << ((i + 1) % 6))
        c6[i] |= (1 << ((i - 1) % 6))
    bases.append(c6)

    # Random small graphs that are K4-free and alpha<=3
    for _ in range(5):
        m = random.randint(5, 10)
        adj = [0] * m
        for i in range(m):
            for j in range(i + 1, m):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if _find_k4(adj) is None and _find_indep6(adj) is None:
            # Check alpha <= 3 (no independent 4)
            # Quick check: just use it if it's valid for our purposes
            bases.append(adj)

    return bases

def construct():
    deadline = time.time() + 110
    best_n = 0
    best_adj = None

    bases = _make_base_graphs()

    # Try n from 35 down to 20
    for n in range(35, 19, -1):
        if time.time() > deadline - 5:
            break

        attempts = 0
        max_attempts = 30
        while time.time() < deadline - 5 and attempts < max_attempts:
            attempts += 1

            # Choose initialization strategy
            strategy = random.random()
            if strategy < 0.6:
                # Blow-up of a base graph
                base = random.choice(bases)
                k = max(1, n // len(base))
                adj = _blowup(base, k, n)
            elif strategy < 0.85:
                # Random graph
                p = random.uniform(0.3, 0.7)
                adj = _random_graph(n, p)
            else:
                # Blow-up with random perturbation
                base = random.choice(bases)
                k = max(1, n // len(base))
                adj = _blowup(base, k, n)
                # Perturb: flip some random edges
                for _ in range(n // 2):
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        if adj[u] & (1 << v):
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

            # Time per attempt
            remaining = deadline - time.time() - 5
            time_per_attempt = min(2.0, max(0.1, remaining / max(1, max_attempts - attempts)))

            if _local_repair(adj, time_per_attempt):
                best_n = n
                best_adj = adj
                break

    # Fallback: try smaller n
    if best_n == 0:
        for n in range(19, 10, -1):
            if time.time() > deadline - 2:
                break
            for _ in range(15):
                base = random.choice(bases)
                k = max(1, n // len(base))
                adj = _blowup(base, k, n)
                if _local_repair(adj, 0.5):
                    best_n = n
                    best_adj = adj
                    break
            if best_n > 0:
                break

    # Absolute fallback: C4
    if best_n == 0:
        best_n = 4
        best_adj = [9, 3, 6, 12]

    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}
