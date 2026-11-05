import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    N = 18
    full_mask = (1 << N) - 1

    # Independent-set check on a bitmask (independent in graph adj)
    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    # Adding edge (u,v) keeps graph K4-free iff common neighbors of u,v are independent
    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)

    # Exact largest triangle-free induced subgraph (branch and bound) with time guard.
    # Returns (value, exact_flag). exact_flag False means value is a lower bound only.
    def largest_tf(adj, time_limit):
        best = [0]
        # Greedy lower bound
        for _ in range(4):
            order = list(range(N))
            random.shuffle(order)
            S = 0
            for v in order:
                if is_independent(adj[v] & S, adj):
                    S |= (1 << v)
            sz = bin(S).count('1')
            if sz > best[0]:
                best[0] = sz
        if best[0] == N:
            return N, True

        # Precompute adjacency list as list of vertex ints for candidate filtering
        def neighbors_in(mask, adj):
            res = []
            m = mask
            while m:
                lsb = m & -m
                res.append(lsb.bit_length() - 1)
                m ^= lsb
            return res

        def bnb(S_mask, C_list):
            if time.time() > time_limit:
                return False
            sz = bin(S_mask).count('1')
            if sz + len(C_list) <= best[0]:
                return True
            if not C_list:
                if sz > best[0]:
                    best[0] = sz
                return True
            # Filter candidates that can be added to S
            new_C = []
            for v in C_list:
                if is_independent(adj[v] & S_mask, adj):
                    new_C.append(v)
            if not new_C:
                if sz > best[0]:
                    best[0] = sz
                return True
            if sz + len(new_C) <= best[0]:
                return True
            # Branch on first candidate
            v = new_C[0]
            rest = new_C[1:]
            bit = 1 << v
            # Include
            bnb(S_mask | bit, rest)
            if time.time() > time_limit:
                return False
            # Exclude
            bnb(S_mask, new_C)
            return True

        exact = bnb(0, list(range(N)))
        return best[0], exact

    def get_edge_list(adj):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # C9 ∪ C9 baseline: two disjoint 9-cycles. K4-free; largest triangle-free induced subgraph = 8.
    def make_c9_union_adj():
        adj = [0] * N
        for i in range(9):
            a = i
            b = (i + 1) % 9
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
        for i in range(9):
            a = 9 + i
            b = 9 + (i + 1) % 9
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
        return adj

    def is_k4_free(adj):
        for i in range(N):
            for j in range(i + 1, N):
                if not ((adj[i] >> j) & 1):
                    continue
                for k in range(j + 1, N):
                    if not ((adj[i] >> k) & 1) or not ((adj[j] >> k) & 1):
                        continue
                    for l in range(k + 1, N):
                        if ((adj[i] >> l) & 1) and ((adj[j] >> l) & 1) and ((adj[k] >> l) & 1):
                            return False
        return True

    random.seed(42)

    # Start from the strong C9 ∪ C9 baseline
    current_adj = make_c9_union_adj()
    base_val, _ = largest_tf(current_adj, deadline - 5.0)
    best_adj = current_adj[:]
    best_score = base_val

    # Exhaustive single-edge-flip local search from current_adj
    def local_search(adj, best_score, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                for j in range(i + 1, N):
                    if time.time() > deadline:
                        break
                    old_val = (adj[i] >> j) & 1
                    if old_val == 0:
                        if not can_add_edge(adj, i, j):
                            continue
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    else:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    val, exact = largest_tf(adj, deadline - 5.0)
                    if exact and val < best_score:
                        best_score = val
                        improved = True
                    else:
                        # Revert
                        if old_val == 0:
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                        else:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
        return best_score

    best_score = local_search(current_adj, best_score, deadline - 15.0)
    if best_score < base_val:
        best_adj = current_adj[:]

    # Random restarts with greedy K4-free construction + local search
    while time.time() < deadline - 20.0:
        adj = [0] * N
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        if not is_k4_free(adj):
            continue
        val, exact = largest_tf(adj, deadline - 10.0)
        if exact and val < best_score:
            best_score = val
            best_adj = adj[:]
            # Polish from this new best
            best_score = local_search(adj, best_score, deadline - 15.0)
            if best_score < val:
                best_adj = adj[:]

    # Final safety: if we somehow lost the baseline, ensure we return a valid K4-free graph
    if not is_k4_free(best_adj):
        best_adj = make_c9_union_adj()
        best_score = base_val

    return get_edge_list(best_adj)
