import time
import random

def construct():
    n = 15
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin before 90s kill

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges_total = len(all_edges)
    rng = random.Random(20240517)

    # Use bitmask for adjacency
    # adj[i] is a bitmask of neighbors of vertex i

    def get_deg(adj):
        return [bin(adj[i]).count('1') for i in range(n)]

    def is_valid_fast(adj):
        deg = get_deg(adj)
        # For each edge (u,v), check all s in N(u)\{v} and t in N(v)\{u}
        # s != t and deg[s] != deg[t]
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if not (au >> v) & 1:
                    continue
                av = adj[v]
                # s in N(u), s != v
                s_mask = au & ~(1 << v)
                s = 0
                while s_mask:
                    s = (s_mask & -s_mask).bit_length() - 1
                    s_mask &= s_mask - 1
                    ds = deg[s]
                    # t in N(v), t != u, t != s
                    t_mask = av & ~(1 << u) & ~(1 << s)
                    t = 0
                    while t_mask:
                        t = (t_mask & -t_mask).bit_length() - 1
                        t_mask &= t_mask - 1
                        if deg[t] == ds:
                            return False
        return True

    def find_violation(adj):
        deg = get_deg(adj)
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if not (au >> v) & 1:
                    continue
                av = adj[v]
                s_mask = au & ~(1 << v)
                s = 0
                while s_mask:
                    s = (s_mask & -s_mask).bit_length() - 1
                    s_mask &= s_mask - 1
                    ds = deg[s]
                    t_mask = av & ~(1 << u) & ~(1 << s)
                    t = 0
                    while t_mask:
                        t = (t_mask & -t_mask).bit_length() - 1
                        t_mask &= t_mask - 1
                        if deg[t] == ds:
                            return (s, u, v, t)
        return None

    def ecount(adj):
        c = 0
        for i in range(n):
            c += bin(adj[i]).count('1')
        return c // 2

    best_adj = None
    bestc = -1

    def consider(adj):
        nonlocal best_adj, bestc
        c = ecount(adj)
        if c > bestc and is_valid_fast(adj):
            bestc = c
            best_adj = list(adj)

    # Baseline K7,8
    a0 = [0] * n
    for i in range(7):
        for j in range(7, 15):
            a0[i] |= (1 << j)
            a0[j] |= (1 << i)
    consider(a0)

    def local_improve(start):
        adj = list(start)
        run_best = list(adj)
        run_bc = ecount(adj)
        while time.time() < deadline:
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] >> j) & 1]
            if not missing:
                break
            rng.shuffle(missing)
            progress = False
            for (u, v) in missing:
                if time.time() > deadline:
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid_fast(adj):
                    progress = True
                    c = ecount(adj)
                    if c > run_bc:
                        run_bc = c
                        run_best = list(adj)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            if not progress:
                # kick: remove 1-3 random edges
                es = [(i, j) for i in range(n) for j in range(i + 1, n) if (adj[i] >> j) & 1]
                if not es:
                    break
                k = rng.randint(1, 3)
                rng.shuffle(es)
                for (x, y) in es[:k]:
                    adj[x] &= ~(1 << y)
                    adj[y] &= ~(1 << x)
        return run_best, run_bc

    def greedy_add(order):
        adj = [0] * n
        for (u, v) in order:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not is_valid_fast(adj):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj

    def repair(start):
        adj = list(start)
        for _ in range(200):
            viol = find_violation(adj)
            if viol is None:
                break
            s, u, v, t = viol
            cand = [(s, u), (u, v), (v, t)]
            done = False
            for (x, y) in cand:
                adj[x] &= ~(1 << y)
                adj[y] &= ~(1 << x)
                if is_valid_fast(adj):
                    done = True
                    break
                adj[x] |= (1 << y)
                adj[y] |= (1 << x)
            if not done:
                (x, y) = cand[rng.randrange(3)]
                adj[x] &= ~(1 << y)
                adj[y] &= ~(1 << x)
        return adj

    # Try multiple strategies
    iter_no = 0
    while time.time() < deadline:
        iter_no += 1
        if iter_no % 3 == 1:
            # Greedy from random order
            order = list(all_edges)
            rng.shuffle(order)
            g = greedy_add(order)
            consider(g)
            rb, rc = local_improve(g)
            consider(rb)
        elif iter_no % 3 == 2:
            # Random dense + repair
            p = rng.uniform(0.3, 0.9)
            adj = [0] * n
            for (u, v) in all_edges:
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            r = repair(adj)
            consider(r)
            rb, rc = local_improve(r)
            consider(rb)
        else:
            # Start from K7,8 and try to add intra-partition edges
            adj = list(a0)
            # Try adding edges within the two partitions
            intra_edges = [(i, j) for i in range(7) for j in range(i + 1, 7)] + \
                          [(i, j) for i in range(7, 15) for j in range(i + 1, 15)]
            rng.shuffle(intra_edges)
            for (u, v) in intra_edges:
                if time.time() > deadline:
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not is_valid_fast(adj):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            consider(adj)
            rb, rc = local_improve(adj)
            consider(rb)

    # Final
    if best_adj is None or not is_valid_fast(best_adj):
        best_adj = a0

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([int(u), int(v)])
    return edges
