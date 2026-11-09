"""
K2,3-free graph construction on 20 vertices maximizing edges.

Strategy:
- Bit-parallel adjacency with a maintained common-neighbor count matrix.
- Vectorized numpy batch checks to identify all currently-addable non-edges.
- Multi-restart local search:
    * Greedy vectorized fill
    * Simulated annealing with 1-for-1, 1-for-2, 2-for-2 swap moves
    * Exhaustive final polish with 1-for-1 and 1-for-2 swaps
"""

import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    n = 20
    start_time = time.time()
    time_limit = 112.0

    def popcount(x):
        return bin(x).count('1')

    def count_edges(adj):
        return sum(popcount(x) for x in adj) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_edges(adj):
        edges = []
        for u in range(n):
            temp = adj[u]
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
                temp -= lsb
        return edges

    def get_non_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    edges.append((u, v))
        return edges

    def compute_cn_matrix(adj):
        cn = [[0] * n for _ in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                c = popcount(adj[a] & adj[b])
                cn[a][b] = c
                cn[b][a] = c
        return cn

    def valid_add(adj, u, v, cn=None):
        if adj[u] & (1 << v):
            return False
        if cn is None:
            cn = compute_cn_matrix(adj)
        if cn[u][v] > 2:
            return False
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if cn[w][v] > 1:
                return False
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if cn[w][u] > 1:
                return False
        return True

    def update_cn_after_add(adj, cn, u, v):
        for w in range(n):
            if w != u and w != v:
                cn[u][w] = popcount(adj[u] & adj[w])
                cn[w][u] = cn[u][w]
                cn[v][w] = popcount(adj[v] & adj[w])
                cn[w][v] = cn[v][w]
        cn[u][v] = popcount(adj[u] & adj[v])
        cn[v][u] = cn[u][v]

    def update_cn_after_remove(adj, cn, u, v):
        for w in range(n):
            if w != u and w != v:
                cn[u][w] = popcount(adj[u] & adj[w])
                cn[w][u] = cn[u][w]
                cn[v][w] = popcount(adj[v] & adj[w])
                cn[w][v] = cn[v][w]
        cn[u][v] = popcount(adj[u] & adj[v])
        cn[v][u] = cn[u][v]

    def vectorized_addable_non_edges(adj, cn):
        """
        Use numpy to check all non-edges in parallel for validity.
        Returns list of (u,v) that can be added.
        """
        # Build numpy adjacency
        adj_np = np.zeros((n, n), dtype=np.uint8)
        for u in range(n):
            temp = adj[u]
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                adj_np[u, v] = 1
                temp -= lsb

        # cn_np from cn matrix
        cn_np = np.array(cn, dtype=np.uint8)

        valid_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_np[u, v]:
                    continue
                # Check condition 1: cn[u][v] <= 2
                if cn_np[u, v] > 2:
                    continue
                # Check condition 2: for all w in N(u), cn[w][v] <= 1
                # N(u) as vector
                neighbors_u = adj_np[u]  # length n, 1s where neighbor
                # We need: max over w where neighbors_u[w]==1 of cn[w][v] <= 1
                # i.e., cn_np[neighbors_u==1, v].max() <= 1
                # But cn_np is 2D; cn_np[w, v] for w in N(u)
                # Extract column v, mask by neighbors_u
                col_v = cn_np[:, v]  # shape (n,)
                mask_u = neighbors_u  # shape (n,)
                # Also exclude u and v themselves? cn[u][v] already checked.
                # w ranges over all; but only w in N(u) matter.
                # Note: w could be v? No, v is not in N(u) if (u,v) is non-edge.
                # w could be u? u is not in N(u).
                vals = col_v[mask_u]
                if len(vals) > 0 and vals.max() > 1:
                    continue
                # Check condition 3: for all w in N(v), cn[w][u] <= 1
                neighbors_v = adj_np[v]
                col_u = cn_np[:, u]
                mask_v = neighbors_v
                vals2 = col_u[mask_v]
                if len(vals2) > 0 and vals2.max() > 1:
                    continue
                valid_list.append((u, v))
        return valid_list

    def greedy_fill_vectorized(adj, cn):
        """Repeatedly add all currently-valid non-edges until no more can be added."""
        improved = True
        while improved:
            improved = False
            addable = vectorized_addable_non_edges(adj, cn)
            for u, v in addable:
                if not (adj[u] & (1 << v)):  # double-check in case of races
                    add_edge(adj, u, v)
                    update_cn_after_add(adj, cn, u, v)
                    improved = True

    def make_k4_union():
        adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    add_edge(adj, base + a, base + b)
        return adj

    best_edges = 0
    best_adj = None

    # Baseline
    adj0 = make_k4_union()
    cn0 = compute_cn_matrix(adj0)
    greedy_fill_vectorized(adj0, cn0)
    best_edges = count_edges(adj0)
    best_adj = adj0[:]

    num_restarts = 30

    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break

        if r == 0:
            adj = make_k4_union()
        else:
            adj = best_adj[:]
            edges = get_edges(adj)
            random.shuffle(edges)
            for u, v in edges[:random.randint(2, 5)]:
                remove_edge(adj, u, v)

        cn = compute_cn_matrix(adj)
        greedy_fill_vectorized(adj, cn)
        current_edges = count_edges(adj)

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

        # Simulated annealing
        T = 0.6
        T_min = 0.02
        cooling = 0.993

        while T > T_min and (time.time() - start_time < time_limit):
            move_type = random.random()

            if move_type < 0.30:
                # Add a random non-edge if valid
                non_edges = get_non_edges(adj)
                if not non_edges:
                    T *= cooling
                    continue
                u, v = random.choice(non_edges)
                if valid_add(adj, u, v, cn):
                    add_edge(adj, u, v)
                    update_cn_after_add(adj, cn, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]

            elif move_type < 0.45:
                # Remove a random edge (SA acceptance)
                edges = get_edges(adj)
                if not edges:
                    T *= cooling
                    continue
                u, v = random.choice(edges)
                if random.random() < math.exp(-1.0 / T):
                    remove_edge(adj, u, v)
                    update_cn_after_remove(adj, cn, u, v)
                    current_edges -= 1

            elif move_type < 0.70:
                # 1-for-1 swap: remove one edge, add another
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or not non_edges:
                    T *= cooling
                    continue
                u1, v1 = random.choice(edges)
                u2, v2 = random.choice(non_edges)
                if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                    T *= cooling
                    continue

                remove_edge(adj, u1, v1)
                update_cn_after_remove(adj, cn, u1, v1)

                if valid_add(adj, u2, v2, cn):
                    add_edge(adj, u2, v2)
                    update_cn_after_add(adj, cn, u2, v2)
                    # Edge count same; accept with prob to explore
                    if random.random() < 0.7:
                        pass
                    else:
                        remove_edge(adj, u2, v2)
                        update_cn_after_remove(adj, cn, u2, v2)
                        add_edge(adj, u1, v1)
                        update_cn_after_add(adj, cn, u1, v1)
                else:
                    add_edge(adj, u1, v1)
                    update_cn_after_add(adj, cn, u1, v1)

            elif move_type < 0.85:
                # 1-for-2 swap: remove one edge, try to add two
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or len(non_edges) < 2:
                    T *= cooling
                    continue
                u1, v1 = random.choice(edges)
                # Pick two random non-edges
                idx1 = random.randint(0, len(non_edges) - 1)
                idx2 = random.randint(0, len(non_edges) - 1)
                if idx1 == idx2:
                    T *= cooling
                    continue
                u2, v2 = non_edges[idx1]
                u3, v3 = non_edges[idx2]
                if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2) or \
                   (u1, v1) == (u3, v3) or (u1, v1) == (v3, u3):
                    T *= cooling
                    continue

                remove_edge(adj, u1, v1)
                update_cn_after_remove(adj, cn, u1, v1)

                added = 0
                if valid_add(adj, u2, v2, cn):
                    add_edge(adj, u2, v2)
                    update_cn_after_add(adj, cn, u2, v2)
                    added += 1
                if valid_add(adj, u3, v3, cn):
                    add_edge(adj, u3, v3)
                    update_cn_after_add(adj, cn, u3, v3)
                    added += 1

                if added >= 1:
                    current_edges += added - 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    # Revert
                    add_edge(adj, u1, v1)
                    update_cn_after_add(adj, cn, u1, v1)

            else:
                # 2-for-2 swap: remove two edges, add two
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if len(edges) < 2 or len(non_edges) < 2:
                    T *= cooling
                    continue
                idx1 = random.randint(0, len(edges) - 1)
                idx2 = random.randint(0, len(edges) - 1)
                if idx1 == idx2:
                    T *= cooling
                    continue
                u1, v1 = edges[idx1]
                u2, v2 = edges[idx2]
                idx3 = random.randint(0, len(non_edges) - 1)
                idx4 = random.randint(0, len(non_edges) - 1)
                if idx3 == idx4:
                    T *= cooling
                    continue
                u3, v3 = non_edges[idx3]
                u4, v4 = non_edges[idx4]

                remove_edge(adj, u1, v1)
                update_cn_after_remove(adj, cn, u1, v1)
                remove_edge(adj, u2, v2)
                update_cn_after_remove(adj, cn, u2, v2)

                added = 0
                if valid_add(adj, u3, v3, cn):
                    add_edge(adj, u3, v3)
                    update_cn_after_add(adj, cn, u3, v3)
                    added += 1
                if valid_add(adj, u4, v4, cn):
                    add_edge(adj, u4, v4)
                    update_cn_after_add(adj, cn, u4, v4)
                    added += 1

                if added == 2:
                    pass  # keep
                elif added == 1:
                    # Revert partially: remove the added edge, restore both removed
                    # Find which was added
                    if adj[u3] & (1 << v3):
                        remove_edge(adj, u3, v3)
                        update_cn_after_remove(adj, cn, u3, v3)
                    else:
                        remove_edge(adj, u4, v4)
                        update_cn_after_remove(adj, cn, u4, v4)
                    add_edge(adj, u1, v1)
                    update_cn_after_add(adj, cn, u1, v1)
                    add_edge(adj, u2, v2)
                    update_cn_after_add(adj, cn, u2, v2)
                else:
                    add_edge(adj, u1, v1)
                    update_cn_after_add(adj, cn, u1, v1)
                    add_edge(adj, u2, v2)
                    update_cn_after_add(adj, cn, u2, v2)

            T *= cooling

    # Final polish
    if best_adj and (time.time() - start_time < time_limit - 5):
        polish_adj = best_adj[:]
        polish_cn = compute_cn_matrix(polish_adj)
        polish_edges = count_edges(polish_adj)
        improved = True
        while improved and (time.time() - start_time < time_limit - 5):
            improved = False
            # Add all valid non-edges
            addable = vectorized_addable_non_edges(polish_adj, polish_cn)
            for u, v in addable:
                if not (polish_adj[u] & (1 << v)):
                    add_edge(polish_adj, u, v)
                    update_cn_after_add(polish_cn, u, v)
                    polish_edges += 1
                    improved = True
            # 1-for-1 swaps
            edges = get_edges(polish_adj)
            non_edges = get_non_edges(polish_adj)
            for a, b in edges:
                if time.time() - start_time > time_limit - 5:
                    break
                for c, d in non_edges:
                    if (a, b) == (c, d) or (a, b) == (d, c):
                        continue
                    remove_edge(polish_adj, a, b)
                    update_cn_after_remove(polish_cn, a, b)
                    if valid_add(polish_adj, c, d, polish_cn):
                        add_edge(polish_adj, c, d)
                        update_cn_after_add(polish_cn, c, d)
                        # Try to add more
                        further = False
                        addable2 = vectorized_addable_non_edges(polish_adj, polish_cn)
                        for x, y in addable2:
                            if not (polish_adj[x] & (1 << y)):
                                add_edge(polish_adj, x, y)
                                update_cn_after_add(polish_cn, x, y)
                                polish_edges += 1
                                further = True
                        if further:
                            improved = True
                            break
                        else:
                            remove_edge(polish_adj, c, d)
                            update_cn_after_remove(polish_cn, c, d)
                            add_edge(polish_adj, a, b)
                            update_cn_after_add(polish_cn, a, b)
                    else:
                        add_edge(polish_adj, a, b)
                        update_cn_after_add(polish_cn, a, b)
                if improved:
                    break
            # 1-for-2 swaps
            if not improved:
                edges = get_edges(polish_adj)
                non_edges = get_non_edges(polish_adj)
                for a, b in edges:
                    if time.time() - start_time > time_limit - 5:
                        break
                    for i in range(len(non_edges)):
                        for j in range(i + 1, len(non_edges)):
                            c, d = non_edges[i]
                            e, f = non_edges[j]
                            if (a, b) in [(c, d), (d, c), (e, f), (f, e)]:
                                continue
                            remove_edge(polish_adj, a, b)
                            update_cn_after_remove(polish_cn, a, b)
                            added = 0
                            if valid_add(polish_adj, c, d, polish_cn):
                                add_edge(polish_adj, c, d)
                                update_cn_after_add(polish_cn, c, d)
                                added += 1
                            if valid_add(polish_adj, e, f, polish_cn):
                                add_edge(polish_adj, e, f)
                                update_cn_after_add(polish_cn, e, f)
                                added += 1
                            if added >= 2:
                                polish_edges += added - 1
                                improved = True
                                break
                            elif added == 1:
                                # Keep the one added edge (net 0), but try to add more
                                further = False
                                addable2 = vectorized_addable_non_edges(polish_adj, polish_cn)
                                for x, y in addable2:
                                    if not (polish_adj[x] & (1 << y)):
                                        add_edge(polish_adj, x, y)
                                        update_cn_after_add(polish_cn, x, y)
                                        polish_edges += 1
                                        further = True
                                if further:
                                    improved = True
                                    break
                                else:
                                    # Revert
                                    if polish_adj[c] & (1 << d):
                                        remove_edge(polish_adj, c, d)
                                        update_cn_after_remove(polish_cn, c, d)
                                    else:
                                        remove_edge(polish_adj, e, f)
                                        update_cn_after_remove(polish_cn, e, f)
                                    add_edge(polish_adj, a, b)
                                    update_cn_after_add(polish_cn, a, b)
                            else:
                                add_edge(polish_adj, a, b)
                                update_cn_after_add(polish_cn, a, b)
                        if improved:
                            break
                    if improved:
                        break
            if polish_edges > best_edges:
                best_edges = polish_edges
                best_adj = polish_adj[:]

    result = []
    for u, v in get_edges(best_adj):
        result.append([u, v])
    return result

# EVOLVE-BLOCK-END
