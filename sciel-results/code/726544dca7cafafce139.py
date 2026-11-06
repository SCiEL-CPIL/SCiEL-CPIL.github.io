import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 112  # safety margin under 120s

    n = 40

    def now_dead():
        return time.time() >= deadline

    def makes_c6(adj, u, v):
        """Return True if adding edge (u,v) creates a C6. Assumes (u,v) not yet in adj."""
        mask_uv = (1 << u) | (1 << v)
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        a = nu
        while a:
            lb = a & (-a)
            ai = lb.bit_length() - 1
            a ^= lb
            d = nv & ~lb
            while d:
                ld = d & (-d)
                di = ld.bit_length() - 1
                d ^= ld
                bmask = adj[ai] & ~lb & ~ld & ~mask_uv
                b = bmask
                while b:
                    lb2 = b & (-b)
                    bi = lb2.bit_length() - 1
                    b ^= lb2
                    cmask = adj[bi] & adj[di] & ~lb & ~lb2 & ~ld & ~mask_uv
                    if cmask:
                        return True
        return False

    def edge_count(adj):
        return sum(adj[u].bit_count() for u in range(n)) // 2

    def build_edges(adj):
        edges = []
        for u in range(n):
            m = adj[u]
            for v in range(u + 1, n):
                if (m >> v) & 1:
                    edges.append([u, v])
        return edges

    def local_search(adj, deadline_check):
        """Try to improve by removing one edge and adding up to two others."""
        improved = True
        while improved and not deadline_check():
            improved = False
            # Try removing each edge and see if we can add more back
            for u in range(n):
                m = adj[u]
                v = u + 1
                while v < n:
                    if (m >> v) & 1:
                        # Try removing (u,v)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        # Try to add back edges that were blocked
                        added = 0
                        for x in range(n):
                            for y in range(x + 1, n):
                                if not (adj[x] >> y) & 1:
                                    if not makes_c6(adj, x, y):
                                        adj[x] |= (1 << y)
                                        adj[y] |= (1 << x)
                                        added += 1
                        if added > 1:
                            improved = True
                        else:
                            # Restore (u,v)
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                    v += 1
                if deadline_check():
                    break
        return adj

    best_adj = [0] * n
    best_cnt = 0

    # Initial structured seed: 8 disjoint K5s
    def make_k5_seed():
        adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    u = base + a
                    v = base + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    # Greedy fill from a seed
    def greedy_fill(seed_adj, seed):
        adj = seed_adj[:]
        remaining = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    remaining.append((u, v))
        random.seed(seed)
        random.shuffle(remaining)
        for (u, v) in remaining:
            if time.time() >= deadline - 2:
                break
            if not makes_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    attempts = 0
    while not now_dead():
        attempts += 1
        seed = 100000 + attempts * 7919

        # Start from K5 seed
        seed_adj = make_k5_seed()
        adj = greedy_fill(seed_adj, seed)
        cnt = edge_count(adj)

        if cnt > best_cnt:
            best_cnt = cnt
            best_adj = adj[:]

        # Local search on best so far
        if cnt >= best_cnt - 2 and not now_dead():
            adj_ls = best_adj[:]
            adj_ls = local_search(adj_ls, now_dead)
            cnt_ls = edge_count(adj_ls)
            if cnt_ls > best_cnt:
                best_cnt = cnt_ls
                best_adj = adj_ls[:]

        # Also try empty-start greedy with different seeds for diversity
        if attempts % 3 == 0:
            empty_adj = [0] * n
            adj2 = greedy_fill(empty_adj, seed + 1)
            cnt2 = edge_count(adj2)
            if cnt2 > best_cnt:
                best_cnt = cnt2
                best_adj = adj2[:]

    return build_edges(best_adj)
