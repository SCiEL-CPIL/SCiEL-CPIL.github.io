"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    n = 16
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

    # ---- Precompute Q3 masks (12-edge 3-regular submasks of K4,4) ----
    def is_q3_mask(mask):
        da = [0]*4
        db = [0]*4
        for i in range(4):
            for j in range(4):
                if mask & (1 << (i*4 + j)):
                    da[i] += 1
                    db[j] += 1
        return all(d == 3 for d in da) and all(d == 3 for d in db)

    q3_masks = []
    for combo in itertools.combinations(range(16), 12):
        m = 0
        for idx in combo:
            m |= (1 << idx)
        if is_q3_mask(m):
            q3_masks.append(m)
    q3_masks = tuple(q3_masks)  # 6 masks

    # ---- Precompute all K4,4 splits (A of size 4, B of size 4) ----
    all_verts = list(range(n))
    k44_list = []  # each: (A tuple, B tuple, pairs [(a,b,bitpos),...])
    for A in itertools.combinations(all_verts, 4):
        Aset = set(A)
        for B in itertools.combinations(all_verts, 4):
            if Aset & set(B):
                continue
            pairs = []
            for i in range(4):
                for j in range(4):
                    pairs.append((A[i], B[j], i*4 + j))
            k44_list.append((A, B, pairs))

    # ---- Index: for each edge, which K4,4 splits contain it ----
    edge_to_k44 = {}
    for idx, (A, B, pairs) in enumerate(k44_list):
        for a, b, _ in pairs:
            e = (a, b) if a < b else (b, a)
            edge_to_k44.setdefault(e, []).append(idx)

    # ---- Build graph as bitsets ----
    adj = [0] * n
    def has_edge(u, v):
        return bool(adj[u] & (1 << v))
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    def edge_count():
        return sum(adj[i].bit_count() for i in range(n)) // 2

    # ---- Seed: K7 + K7 + K2 (57 edges, Q3-free) ----
    for i in range(7):
        for j in range(i+1, 7):
            add_edge(i, j)
    for i in range(7, 14):
        for j in range(i+1, 14):
            add_edge(i, j)
    add_edge(14, 15)

    best_edges = edge_count()
    best_adj = adj[:]

    def time_left():
        return deadline - time.time()

    # ---- Check if adding (u,v) creates a Q3 ----
    def creates_q3(u, v):
        e = (u, v) if u < v else (v, u)
        for kidx in edge_to_k44.get(e, ()):
            A, B, pairs = k44_list[kidx]
            mask = 0
            for a, b, bit in pairs:
                if (a == u and b == v) or (a == v and b == u):
                    mask |= (1 << bit)  # the new edge
                elif has_edge(a, b):
                    mask |= (1 << bit)
            for qm in q3_masks:
                if (mask & qm) == qm:
                    return True
        return False

    # ---- Count number of Q3 subgraphs in current graph ----
    def count_q3():
        cnt = 0
        for kidx in range(len(k44_list)):
            A, B, pairs = k44_list[kidx]
            mask = 0
            for a, b, bit in pairs:
                if has_edge(a, b):
                    mask |= (1 << bit)
            for qm in q3_masks:
                if (mask & qm) == qm:
                    cnt += 1
        return cnt

    # ---- Greedy: try to add every missing edge ----
    missing = [e for e in all_pairs if not has_edge(*e)]
    random.seed(42)
    random.shuffle(missing)

    for u, v in missing:
        if time_left() < 15:
            break
        if not creates_q3(u, v):
            add_edge(u, v)
            if edge_count() > best_edges:
                best_edges = edge_count()
                best_adj = adj[:]

    # ---- Simulated Annealing on edge set ----
    if time_left() > 20:
        adj[:] = best_adj[:]
        sa_edges = edge_count()
        sa_q3 = count_q3()

        T = 5.0
        cooling = 0.9995
        min_T = 0.1
        all_pairs_list = all_pairs[:]

        while time_left() > 10 and T > min_T:
            u, v = random.choice(all_pairs_list)
            if has_edge(u, v):
                remove_edge(u, v)
                new_edges = sa_edges - 1
                new_q3 = count_q3()
                new_score = new_edges - 100 * new_q3
                old_score = sa_edges - 100 * sa_q3
                delta = new_score - old_score
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    sa_edges = new_edges
                    sa_q3 = new_q3
                    if sa_q3 == 0 and sa_edges > best_edges:
                        best_edges = sa_edges
                        best_adj = adj[:]
                else:
                    add_edge(u, v)
            else:
                add_edge(u, v)
                new_edges = sa_edges + 1
                new_q3 = count_q3()
                new_score = new_edges - 100 * new_q3
                old_score = sa_edges - 100 * sa_q3
                delta = new_score - old_score
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    sa_edges = new_edges
                    sa_q3 = new_q3
                    if sa_q3 == 0 and sa_edges > best_edges:
                        best_edges = sa_edges
                        best_adj = adj[:]
                else:
                    remove_edge(u, v)

            T *= cooling

        if count_q3() > 0:
            for _ in range(100):
                if count_q3() == 0:
                    break
                if time_left() < 5:
                    break
                found = False
                for u, v in all_pairs:
                    if not has_edge(u, v):
                        continue
                    old_q3 = count_q3()
                    remove_edge(u, v)
                    new_q3 = count_q3()
                    if new_q3 < old_q3:
                        found = True
                        break
                    add_edge(u, v)
                if not found:
                    break

    # ---- Local search: 1-1 edge swaps to gain density ----
    adj[:] = best_adj[:]
    for _ in range(2000):
        if time_left() < 10:
            break
        rem_edges = [e for e in all_pairs if has_edge(*e)]
        if not rem_edges:
            break
        ru, rv = random.choice(rem_edges)
        remove_edge(ru, rv)

        miss = [e for e in all_pairs if not has_edge(*e)]
        random.shuffle(miss)
        added = 0
        for u, v in miss:
            if added >= 2:
                break
            if not creates_q3(u, v):
                add_edge(u, v)
                added += 1

        if edge_count() > best_edges:
            best_edges = edge_count()
            best_adj = adj[:]
        else:
            for u, v in miss[:added]:
                remove_edge(u, v)
            add_edge(ru, rv)

    # ---- Final polish: exhaustive 1-swap search ----
    adj[:] = best_adj[:]
    improved = True
    while improved and time_left() > 5:
        improved = False
        for ru, rv in all_pairs:
            if time_left() < 5:
                break
            if not has_edge(ru, rv):
                continue
            remove_edge(ru, rv)
            for u, v in all_pairs:
                if time_left() < 5:
                    break
                if has_edge(u, v):
                    continue
                if not creates_q3(u, v):
                    add_edge(u, v)
                    if edge_count() > best_edges:
                        best_edges = edge_count()
                        best_adj = adj[:]
                        improved = True
                    remove_edge(u, v)
                    break
            add_edge(ru, rv)

    # Restore best state
    adj[:] = best_adj[:]

    # Final validity check
    if time_left() > 5:
        found = False
        for kidx in range(len(k44_list)):
            A, B, pairs = k44_list[kidx]
            mask = 0
            for a, b, bit in pairs:
                if has_edge(a, b):
                    mask |= (1 << bit)
            for qm in q3_masks:
                if (mask & qm) == qm:
                    found = True
                    break
            if found:
                break
        if found:
            adj = [0]*n
            for i in range(7):
                for j in range(i+1, 7):
                    add_edge(i, j)
            for i in range(7, 14):
                for j in range(i+1, 14):
                    add_edge(i, j)
            add_edge(14, 15)

    return [[i, j] for i in range(n) for j in range(i+1, n) if has_edge(i, j)]
# EVOLVE-BLOCK-END
