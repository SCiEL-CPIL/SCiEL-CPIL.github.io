import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill

    n = 20
    total_possible = n * (n - 1) // 2  # 190

    def popcount(x):
        return bin(x).count('1')

    # ---- feasibility: can we add edge (u,v)? ----
    def can_add(adj, u, v):
        if (adj[u] >> v) & 1:
            return False
        # pair (u,v) must have <= 2 common neighbours
        if popcount(adj[u] & adj[v]) > 2:
            return False
        # for w in N(v)\{u}: common(u,w) must be <= 1
        nv = adj[v] & ~(1 << u)
        while nv:
            lsb = nv & -nv
            w = lsb.bit_length() - 1
            nv ^= lsb
            if popcount(adj[u] & adj[w]) > 1:
                return False
        # for w in N(u)\{v}: common(v,w) must be <= 1
        nu = adj[u] & ~(1 << v)
        while nu:
            lsb = nu & -nu
            w = lsb.bit_length() - 1
            nu ^= lsb
            if popcount(adj[v] & adj[w]) > 1:
                return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def edge_count(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2

    def get_neighbors(adj, i):
        res = []
        t = adj[i]
        while t:
            lsb = t & -t
            res.append(lsb.bit_length() - 1)
            t ^= lsb
        return res

    # ---- deterministic structured seed: two 5-cliques + two K4s + 5 cross edges ----
    def build_seed():
        adj = [0] * n
        # clique on {0,1,2,3,4}
        for i in range(5):
            for j in range(i + 1, 5):
                add_edge(adj, i, j)
        # clique on {5,6,7,8,9}
        for i in range(5, 10):
            for j in range(i + 1, 10):
                add_edge(adj, i, j)
        # K4 on {10,11,12,13}
        for i in range(10, 14):
            for j in range(i + 1, 14):
                add_edge(adj, i, j)
        # K4 on {14,15,16,17}
        for i in range(14, 18):
            for j in range(i + 1, 18):
                add_edge(adj, i, j)
        # 5 cross edges: 18-i (i=0..4) -> 19-i
        for i in range(5):
            add_edge(adj, 18 - i, 19 - i)
        return adj

    # ---- complete fill: repeatedly add any feasible missing edge until stuck ----
    def complete_fill(adj):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1) and can_add(adj, u, v):
                        add_edge(adj, u, v)
                        changed = True
        return adj

    # ---- 2-opt: remove an existing edge, add a missing edge ----
    def two_opt(adj, time_limit):
        t0 = time.time()
        improved = True
        while improved and time.time() < t0 + time_limit:
            improved = False
            # collect existing edges
            existing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        existing.append((u, v))
            random.shuffle(existing)
            # collect missing edges
            missing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        missing.append((u, v))
            random.shuffle(missing)
            for e1 in existing:
                if time.time() > t0 + time_limit:
                    break
                u1, v1 = e1
                remove_edge(adj, u1, v1)
                for e2 in missing:
                    if time.time() > t0 + time_limit:
                        break
                    u2, v2 = e2
                    if (adj[u2] >> v2) & 1:
                        continue  # already present (shouldn't happen for missing)
                    if can_add(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        improved = True
                        break
                    # else try next e2
                # if no e2 worked, put e1 back
                if not improved:
                    add_edge(adj, u1, v1)
        return adj

    # ---- 1-opt: try to add a single missing edge ----
    def one_opt_add(adj):
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1) and can_add(adj, u, v):
                        add_edge(adj, u, v)
                        improved = True
        return adj

    # ---- main multi-start search ----
    best_adj = [0] * n
    best_count = 0

    def update_best(adj):
        nonlocal best_adj, best_count
        c = edge_count(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    # Start 0: structured seed
    if time.time() < deadline:
        adj = build_seed()
        complete_fill(adj)
        update_best(adj)

    # Start 1: random seed
    if time.time() < deadline:
        adj = [0] * n
        for _ in range(25):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not ((adj[u] >> v) & 1):
                if can_add(adj, u, v):
                    add_edge(adj, u, v)
        complete_fill(adj)
        update_best(adj)

    # More restarts: perturb best and search
    restart_budget = deadline - start_time - 3.0
    if restart_budget > 0:
        max_restarts = int(restart_budget / 4.0) + 1
        for r in range(max_restarts):
            if time.time() > deadline - 3.0:
                break
            # perturb best_adj
            adj = best_adj[:]
            # remove k random edges
            k = random.randint(2, 6)
            for _ in range(k):
                if edge_count(adj) == 0:
                    break
                u = random.randint(0, n - 1)
                neigh = get_neighbors(adj, u)
                if not neigh:
                    continue
                v = random.choice(neigh)
                remove_edge(adj, u, v)
            # complete fill
            complete_fill(adj)
            # 2-opt for a short time
            t_left = deadline - start_time - 3.0
            if t_left > 1.0:
                two_opt(adj, min(2.0, t_left * 0.4))
            # final 1-opt
            one_opt_add(adj)
            update_best(adj)

    # Final polish: complete fill + 1-opt on best
    if time.time() < deadline - 1.0:
        adj = best_adj[:]
        complete_fill(adj)
        one_opt_add(adj)
        update_best(adj)

    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
