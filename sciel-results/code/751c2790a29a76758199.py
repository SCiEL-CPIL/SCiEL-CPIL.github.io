import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 118

    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    adj = [0] * n

    def is_valid_add(u, v):
        if (adj[u] >> v) & 1:
            return False
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            if ((adj[w] & adj[v]) | (1 << u)).bit_count() > 2:
                return False
            temp ^= lsb
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            if ((adj[u] & adj[w]) | (1 << v)).bit_count() > 2:
                return False
            temp ^= lsb
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    best_count = 0
    best_adj = [0] * n

    def greedy_init():
        nonlocal adj
        adj = [0] * n
        edges = all_edges[:]
        random.shuffle(edges)
        for u, v in edges:
            if is_valid_add(u, v):
                add_edge(u, v)

    def polish():
        nonlocal adj
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(u, v):
                        add_edge(u, v)
                        improved = True

    def local_search():
        nonlocal adj
        # Swap-based local search
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            present = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
            missing = [(u, v) for u, v in all_edges if not ((adj[u] >> v) & 1)]
            if not present or not missing:
                break

            idx_p = random.randint(0, len(present) - 1)
            idx_m = random.randint(0, len(missing) - 1)
            u1, v1 = present[idx_p]
            u2, v2 = missing[idx_m]

            remove_edge(u1, v1)
            if is_valid_add(u2, v2):
                add_edge(u2, v2)
                polish()
            else:
                add_edge(u1, v1)

    def run_sa():
        nonlocal adj, best_count, best_adj
        # Greedy init
        greedy_init()
        polish()

        cur_count = sum(adj[i].bit_count() for i in range(n)) // 2
        if cur_count > best_count:
            best_count = cur_count
            best_adj = adj[:]

        T = 1.0
        alpha = 0.99995
        min_T = 0.01

        while time.time() < deadline - 5:
            present = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
            missing = [(u, v) for u, v in all_edges if not ((adj[u] >> v) & 1)]

            move_type = random.random()
            if move_type < 0.6 and missing:
                u, v = random.choice(missing)
                if is_valid_add(u, v):
                    add_edge(u, v)
                    cur_count += 1
                    if cur_count > best_count:
                        best_count = cur_count
                        best_adj = adj[:]
            elif move_type < 0.8 and present:
                u, v = random.choice(present)
                if random.random() < math.exp(-1.0 / max(T, min_T)):
                    remove_edge(u, v)
                    cur_count -= 1
            else:
                if present and missing:
                    u1, v1 = random.choice(present)
                    u2, v2 = random.choice(missing)
                    remove_edge(u1, v1)
                    if is_valid_add(u2, v2):
                        add_edge(u2, v2)
                    else:
                        add_edge(u1, v1)

            T = max(T * alpha, min_T)

        # Final polish
        polish()
        cur_count = sum(adj[i].bit_count() for i in range(n)) // 2
        if cur_count > best_count:
            best_count = cur_count
            best_adj = adj[:]

    # Run multiple SA restarts
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        run_sa()
        local_search()
        cur_count = sum(adj[i].bit_count() for i in range(n)) // 2
        if cur_count > best_count:
            best_count = cur_count
            best_adj = adj[:]

    res = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                res.append([i, j])
    return res

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    adj = [0] * 20
    for u, v in e:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    valid = True
    for i in range(20):
        for j in range(i + 1, 20):
            if (adj[i] & adj[j]).bit_count() > 2:
                valid = False
                print(f"Violation: {i}, {j} have {(adj[i] & adj[j]).bit_count()} common neighbors")
    print("Valid" if valid else "Invalid")
