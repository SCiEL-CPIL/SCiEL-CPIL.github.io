import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    total_pairs = len(all_edges)  # 91

    # Structured value sets (keep the family; add multiplicative/divisor-rich sets)
    value_sets = [
        list(range(1, 15)),
        list(range(2, 16)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 72, 96, 144],
        [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48],
        [1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 18],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 40],
    ]

    best_score = float('inf')
    best_values = None
    best_edges = None

    # Global edge pool for random draws (avoid rebuilding list each iter)
    edge_index = {e: k for k, e in enumerate(all_edges)}

    def greedy_initial(pair_sums, pair_prods):
        # Build a 35-edge set greedily preferring high-collision sums/products.
        # Score an edge by how many existing edges already share its sum or product.
        chosen = []
        s_counts = {}
        p_counts = {}
        remaining = set(all_edges)
        for _ in range(m):
            best_e = None
            best_key = None
            # sample a subset to keep it fast; but 91 is small, do full scan
            for e in remaining:
                s = pair_sums[e]
                p = pair_prods[e]
                key = s_counts.get(s, 0) * 2 + p_counts.get(p, 0) * 2
                if best_key is None or key > best_key:
                    best_key = key
                    best_e = e
            # tie-break: prefer smaller sum then smaller product for determinism-ish
            chosen.append(best_e)
            remaining.remove(best_e)
            s = pair_sums[best_e]
            p = pair_prods[best_e]
            s_counts[s] = s_counts.get(s, 0) + 1
            p_counts[p] = p_counts.get(p, 0) + 1
        return chosen

    def evaluate_state(edge_list, pair_sums, pair_prods):
        s_counts = {}
        p_counts = {}
        for e in edge_list:
            s = pair_sums[e]
            p = pair_prods[e]
            s_counts[s] = s_counts.get(s, 0) + 1
            p_counts[p] = p_counts.get(p, 0) + 1
        return max(len(s_counts), len(p_counts))

    def local_search(initial_edges, pair_sums, pair_prods, max_iters, use_2swap):
        current = set(initial_edges)
        s_counts = {}
        p_counts = {}
        for e in current:
            s = pair_sums[e]
            p = pair_prods[e]
            s_counts[s] = s_counts.get(s, 0) + 1
            p_counts[p] = p_counts.get(p, 0) + 1

        cur_score = max(len(s_counts), len(p_counts))
        best_local_score = cur_score
        best_local_edges = set(current)

        T = 6.0
        cooling = 0.99995
        min_T = 0.15

        it = 0
        while it < max_iters:
            if time.time() > deadline:
                break
            it += 1

            if use_2swap and random.random() < 0.3:
                # 2-edge swap: remove two, add two
                a1 = random.choice(tuple(current))
                cur_list = tuple(current)
                a2 = random.choice([x for x in cur_list if x != a1]) if len(cur_list) > 1 else a1
                b1 = random.choice(all_edges)
                b2 = random.choice(all_edges)
                # ensure distinct and not in current
                if b1 in current or b1 == b2:
                    continue
                if b2 in current:
                    continue
                if a1 == a2:
                    continue

                old_s1, old_p1 = pair_sums[a1], pair_prods[a1]
                old_s2, old_p2 = pair_sums[a2], pair_prods[a2]
                new_s1, new_p1 = pair_sums[b1], pair_prods[b1]
                new_s2, new_p2 = pair_sums[b2], pair_prods[b2]

                # remove a1, a2
                s_counts[old_s1] -= 1
                if s_counts[old_s1] == 0: del s_counts[old_s1]
                p_counts[old_p1] -= 1
                if p_counts[old_p1] == 0: del p_counts[old_p1]
                s_counts[old_s2] -= 1
                if s_counts[old_s2] == 0: del s_counts[old_s2]
                p_counts[old_p2] -= 1
                if p_counts[old_p2] == 0: del p_counts[old_p2]

                # add b1, b2
                s_counts[new_s1] = s_counts.get(new_s1, 0) + 1
                p_counts[new_p1] = p_counts.get(new_p1, 0) + 1
                s_counts[new_s2] = s_counts.get(new_s2, 0) + 1
                p_counts[new_p2] = p_counts.get(new_p2, 0) + 1

                new_score = max(len(s_counts), len(p_counts))

                accept = new_score < cur_score or (
                    random.random() < min(1.0, (cur_score - new_score) / T) if new_score > cur_score else 1.0
                )
                if accept:
                    current.discard(a1); current.discard(a2)
                    current.add(b1); current.add(b2)
                    cur_score = new_score
                    if cur_score < best_local_score:
                        best_local_score = cur_score
                        best_local_edges = set(current)
                else:
                    # undo
                    s_counts[new_s1] -= 1
                    if s_counts[new_s1] == 0: del s_counts[new_s1]
                    p_counts[new_p1] -= 1
                    if p_counts[new_p1] == 0: del p_counts[new_p1]
                    s_counts[new_s2] -= 1
                    if s_counts[new_s2] == 0: del s_counts[new_s2]
                    p_counts[new_p2] -= 1
                    if p_counts[new_p2] == 0: del p_counts[new_p2]
                    s_counts[old_s1] = s_counts.get(old_s1, 0) + 1
                    p_counts[old_p1] = p_counts.get(old_p1, 0) + 1
                    s_counts[old_s2] = s_counts.get(old_s2, 0) + 1
                    p_counts[old_p2] = p_counts.get(old_p2, 0) + 1
            else:
                # 1-edge swap
                old_edge = random.choice(tuple(current))
                new_edge = random.choice(all_edges)
                if new_edge in current:
                    continue

                old_s, old_p = pair_sums[old_edge], pair_prods[old_edge]
                new_s, new_p = pair_sums[new_edge], pair_prods[new_edge]

                s_counts[old_s] -= 1
                if s_counts[old_s] == 0: del s_counts[old_s]
                p_counts[old_p] -= 1
                if p_counts[old_p] == 0: del p_counts[old_p]

                s_counts[new_s] = s_counts.get(new_s, 0) + 1
                p_counts[new_p] = p_counts.get(new_p, 0) + 1

                new_score = max(len(s_counts), len(p_counts))

                accept = new_score < cur_score or (
                    random.random() < min(1.0, (cur_score - new_score) / T) if new_score > cur_score else 1.0
                )
                if accept:
                    current.discard(old_edge)
                    current.add(new_edge)
                    cur_score = new_score
                    if cur_score < best_local_score:
                        best_local_score = cur_score
                        best_local_edges = set(current)
                else:
                    s_counts[new_s] -= 1
                    if s_counts[new_s] == 0: del s_counts[new_s]
                    p_counts[new_p] -= 1
                    if p_counts[new_p] == 0: del p_counts[new_p]
                    s_counts[old_s] = s_counts.get(old_s, 0) + 1
                    p_counts[old_p] = p_counts.get(old_p, 0) + 1

            T *= cooling
            if T < min_T:
                break

        return best_local_score, list(best_local_edges)

    def polish(edge_list, pair_sums, pair_prods, max_iters):
        # High-precision polish: only accept strictly improving moves, more iters.
        current = set(edge_list)
        s_counts = {}
        p_counts = {}
        for e in current:
            s = pair_sums[e]
            p = pair_prods[e]
            s_counts[s] = s_counts.get(s, 0) + 1
            p_counts[p] = p_counts.get(p, 0) + 1
        cur_score = max(len(s_counts), len(p_counts))
        best_score = cur_score
        best_edges = set(current)

        it = 0
        while it < max_iters:
            if time.time() > deadline:
                break
            it += 1
            old_edge = random.choice(tuple(current))
            new_edge = random.choice(all_edges)
            if new_edge in current:
                continue
            old_s, old_p = pair_sums[old_edge], pair_prods[old_edge]
            new_s, new_p = pair_sums[new_edge], pair_prods[new_edge]

            s_counts[old_s] -= 1
            if s_counts[old_s] == 0: del s_counts[old_s]
            p_counts[old_p] -= 1
            if p_counts[old_p] == 0: del p_counts[old_p]
            s_counts[new_s] = s_counts.get(new_s, 0) + 1
            p_counts[new_p] = p_counts.get(new_p, 0) + 1
            new_score = max(len(s_counts), len(p_counts))

            if new_score < cur_score:
                current.discard(old_edge); current.add(new_edge)
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_edges = set(current)
            else:
                s_counts[new_s] -= 1
                if s_counts[new_s] == 0: del s_counts[new_s]
                p_counts[new_p] -= 1
                if p_counts[new_p] == 0: del p_counts[new_p]
                s_counts[old_s] = s_counts.get(old_s, 0) + 1
                p_counts[old_p] = p_counts.get(old_p, 0) + 1

        return best_score, list(best_edges)

    for vals in value_sets:
        if time.time() > deadline:
            break

        pair_sums = {}
        pair_prods = {}
        for i in range(n):
            for j in range(i + 1, n):
                pair_sums[(i, j)] = vals[i] + vals[j]
                pair_prods[(i, j)] = vals[i] * vals[j]

        # Greedy initial
        greedy_edges = greedy_initial(pair_sums, pair_prods)
        g_score = evaluate_state(greedy_edges, pair_sums, pair_prods)
        if g_score < best_score:
            best_score = g_score
            best_values = vals
            best_edges = list(greedy_edges)

        # Multiple restarts: mix greedy + random
        for restart in range(30):
            if time.time() > deadline:
                break
            if restart == 0:
                initial = greedy_edges
            elif restart == 1:
                initial = random.sample(all_edges, m)
            else:
                # perturb greedy
                initial = list(greedy_edges)
                for _ in range(5):
                    initial[random.randrange(m)] = random.choice(all_edges)
                # dedupe to a set then back to list of 35
                initial = list(set(initial))
                while len(initial) < m:
                    e = random.choice(all_edges)
                    if e not in initial:
                        initial.append(e)

            score, edges = local_search(initial, pair_sums, pair_prods, 40000, use_2swap=True)
            if score < best_score:
                best_score = score
                best_values = vals
                best_edges = edges
            if best_score <= 8:
                break

    # Final polish on the best found
    if best_edges is not None and best_values is not None:
        pair_sums = {}
        pair_prods = {}
        for i in range(n):
            for j in range(i + 1, n):
                pair_sums[(i, j)] = best_values[i] + best_values[j]
                pair_prods[(i, j)] = best_values[i] * best_values[j]
        p_score, p_edges = polish(best_edges, pair_sums, pair_prods, 60000)
        if p_score < best_score:
            best_score = p_score
            best_edges = p_edges

    if best_edges is None:
        best_values = list(range(1, 15))
        best_edges = [list(e) for e in random.sample(all_edges, m)]

    return {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges]
    }
