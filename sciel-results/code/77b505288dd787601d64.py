"""Optimize graph on 16 vertices with 65 edges to minimize max triangle degree sum.
Uses structured initializations, fast bitset-based scoring, and multi-phase local search
with simulated annealing and deterministic polish."""
import random
import time
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0
FULL_MASK = (1 << N) - 1


def _build_adj_from_edges(edge_list):
    adj_bits = [0] * N
    for (u, v) in edge_list:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    return adj_bits


def _compute_degrees(adj_bits):
    deg = [0] * N
    for i in range(N):
        deg[i] = adj_bits[i].bit_count()
    return deg


def _max_tri_deg_sum(adj_bits, deg):
    best = 0
    for i in range(N):
        for j in range(i + 1, N):
            if not (adj_bits[i] >> j) & 1:
                continue
            common = adj_bits[i] & adj_bits[j]
            mask = FULL_MASK ^ ((1 << (j + 1)) - 1)
            common &= mask
            c = common
            while c:
                k = (c & (-c)).bit_length() - 1
                s = deg[i] + deg[j] + deg[k]
                if s > best:
                    best = s
                c &= c - 1
    return best


def _apply_swap(adj_bits, deg, e_remove, e_add):
    u, v = e_remove
    x, y = e_add
    new_adj = adj_bits[:]
    new_deg = deg[:]
    new_adj[u] &= ~(1 << v)
    new_adj[v] &= ~(1 << u)
    new_deg[u] -= 1
    new_deg[v] -= 1
    new_adj[x] |= (1 << y)
    new_adj[y] |= (1 << x)
    new_deg[x] += 1
    new_deg[y] += 1
    new_score = _max_tri_deg_sum(new_adj, new_deg)
    return new_adj, new_deg, new_score


def _try_bipartition(a_size, rng, n_trials=50):
    b_size = N - a_size
    cross = a_size * b_size
    internal_needed = TARGET_EDGES - cross
    if internal_needed < 0:
        return None
    max_internal = a_size * (a_size - 1) // 2 + b_size * (b_size - 1) // 2
    if internal_needed > max_internal:
        return None

    adj_base = [0] * N
    for i in range(a_size):
        for j in range(a_size, N):
            adj_base[i] |= (1 << j)
            adj_base[j] |= (1 << i)

    internal_a = [(i, j) for i in range(a_size) for j in range(i + 1, a_size)]
    internal_b = [(i, j) for i in range(a_size, N) for j in range(i + 1, N)]
    all_internal = internal_a + internal_b

    best_result = None
    best_score = float('inf')

    for _ in range(n_trials):
        if internal_needed > len(all_internal):
            break
        chosen = rng.sample(all_internal, internal_needed)
        adj_try = adj_base[:]
        for (i, j) in chosen:
            adj_try[i] |= (1 << j)
            adj_try[j] |= (1 << i)
        deg_try = _compute_degrees(adj_try)
        score_try = _max_tri_deg_sum(adj_try, deg_try)
        if score_try < best_score:
            best_score = score_try
            best_result = (adj_try, deg_try, score_try)

    return best_result


def _try_balanced_internal(a_size, rng, n_trials=50):
    b_size = N - a_size
    cross = a_size * b_size
    internal_needed = TARGET_EDGES - cross
    if internal_needed < 0:
        return None
    max_internal = a_size * (a_size - 1) // 2 + b_size * (b_size - 1) // 2
    if internal_needed > max_internal:
        return None

    adj_base = [0] * N
    for i in range(a_size):
        for j in range(a_size, N):
            adj_base[i] |= (1 << j)
            adj_base[j] |= (1 << i)

    internal_a = [(i, j) for i in range(a_size) for j in range(i + 1, a_size)]
    internal_b = [(i, j) for i in range(a_size, N) for j in range(i + 1, N)]

    best_result = None
    best_score = float('inf')

    for _ in range(n_trials):
        # Distribute internal edges between A and B
        n_a = rng.randint(0, min(len(internal_a), internal_needed))
        n_b = internal_needed - n_a
        if n_b > len(internal_b):
            n_b = len(internal_b)
            n_a = internal_needed - n_b
            if n_a > len(internal_a):
                n_a = len(internal_a)
                n_b = internal_needed - n_a
                if n_b < 0:
                    continue

        chosen = rng.sample(internal_a, n_a) + rng.sample(internal_b, n_b)
        adj_try = adj_base[:]
        for (i, j) in chosen:
            adj_try[i] |= (1 << j)
            adj_try[j] |= (1 << i)
        deg_try = _compute_degrees(adj_try)
        score_try = _max_tri_deg_sum(adj_try, deg_try)
        if score_try < best_score:
            best_score = score_try
            best_result = (adj_try, deg_try, score_try)

    return best_result


def _random_graph(rng):
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random_edges = set(rng.sample(all_pairs, TARGET_EDGES))
    adj = _build_adj_from_edges(list(random_edges))
    deg = _compute_degrees(adj)
    score = _max_tri_deg_sum(adj, deg)
    return adj, deg, score


def _run_sa(adj_bits, deg, cur_score, t_start, rng, time_budget):
    start_temp = 2.5
    end_temp = 0.005
    max_iters = 200000
    cooling = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
    temp = start_temp

    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
    nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]

    best_local_score = cur_score
    best_local_adj = adj_bits[:]
    best_local_deg = deg[:]

    iters = 0
    while iters < max_iters:
        if time.time() - t_start > time_budget:
            break

        if not edges or not nonedges:
            break

        e_remove = edges[rng.randrange(len(edges))]
        e_add = nonedges[rng.randrange(len(nonedges))]

        new_adj, new_deg, new_score = _apply_swap(adj_bits, deg, e_remove, e_add)

        if new_score < cur_score:
            accept = True
        elif new_score == cur_score:
            accept = rng.random() < 0.2
        else:
            delta = new_score - cur_score
            if temp < 1e-10:
                accept = False
            else:
                accept = math.exp(-delta / temp) > rng.random()

        if accept:
            adj_bits = new_adj
            deg = new_deg
            cur_score = new_score

            if cur_score < best_local_score:
                best_local_score = cur_score
                best_local_adj = adj_bits[:]
                best_local_deg = deg[:]

            edges.remove(e_remove)
            nonedges.append(e_remove)
            nonedges.remove(e_add)
            edges.append(e_add)

        iters += 1
        temp = max(end_temp, temp * cooling)

    return best_local_adj, best_local_deg, best_local_score


def _polish(adj_bits, deg, cur_score, t_start, time_budget, rng):
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
    nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]

    for polish_pass in range(20):
        if time.time() - t_start > time_budget:
            break

        improved = False
        best_swap = None
        best_new_score = cur_score
        best_new_adj = None
        best_new_deg = None

        for e_remove in edges:
            if time.time() - t_start > time_budget:
                break
            for e_add in nonedges:
                new_adj, new_deg, new_score = _apply_swap(adj_bits, deg, e_remove, e_add)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_swap = (e_remove, e_add)
                    best_new_adj = new_adj
                    best_new_deg = new_deg

        if best_swap is not None and best_new_score < cur_score:
            e_remove, e_add = best_swap
            adj_bits = best_new_adj
            deg = best_new_deg
            cur_score = best_new_score
            edges.remove(e_remove)
            nonedges.append(e_remove)
            nonedges.remove(e_add)
            edges.append(e_add)
            improved = True

        if not improved:
            break

    return adj_bits, deg, cur_score


def construct():
    t0 = time.time()
    rng = random.Random(123)

    def elapsed():
        return time.time() - t0

    # Phase 1: Try structured initializations
    best_adj = None
    best_deg = None
    best_score = float('inf')

    # K_{8,8} + 1 internal edge
    adj_turan = [0] * N
    for i in range(8):
        for j in range(8, 16):
            adj_turan[i] |= (1 << j)
            adj_turan[j] |= (1 << i)
    adj_turan[0] |= (1 << 1)
    adj_turan[1] |= (1 << 0)
    deg_turan = _compute_degrees(adj_turan)
    score_turan = _max_tri_deg_sum(adj_turan, deg_turan)
    if score_turan < best_score:
        best_adj = adj_turan[:]
        best_deg = deg_turan[:]
        best_score = score_turan

    # Try various bipartition sizes
    for a_size in range(5, 12):
        if elapsed() > DEADLINE - 80:
            break
        result = _try_bipartition(a_size, rng, n_trials=40)
        if result and result[2] < best_score:
            best_adj = result[0]
            best_deg = result[1]
            best_score = result[2]

    # Try balanced internal edges
    for a_size in range(5, 12):
        if elapsed() > DEADLINE - 70:
            break
        result = _try_balanced_internal(a_size, rng, n_trials=40)
        if result and result[2] < best_score:
            best_adj = result[0]
            best_deg = result[1]
            best_score = result[2]

    # Try some random starts
    for _ in range(15):
        if elapsed() > DEADLINE - 60:
            break
        adj_try, deg_try, score_try = _random_graph(rng)
        if score_try < best_score:
            best_adj = adj_try
            best_deg = deg_try
            best_score = score_try

    # Phase 2: Simulated Annealing with restarts
    time_budget_sa = DEADLINE - elapsed() - 20.0
    if time_budget_sa > 10.0:
        n_restarts = max(1, int(time_budget_sa / 25.0))
        time_per_restart = time_budget_sa / n_restarts

        for restart in range(n_restarts):
            if elapsed() > DEADLINE - 20.0:
                break

            if restart == 0:
                adj_bits = best_adj[:]
                deg = best_deg[:]
                cur_score = best_score
            else:
                # Perturb from best
                adj_bits = best_adj[:]
                deg = _compute_degrees(adj_bits)
                cur_score = _max_tri_deg_sum(adj_bits, deg)

                all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
                edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
                nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]

                for _ in range(8):
                    if not edges or not nonedges:
                        break
                    e_remove = edges[rng.randrange(len(edges))]
                    e_add = nonedges[rng.randrange(len(nonedges))]
                    new_adj, new_deg, new_score = _apply_swap(adj_bits, deg, e_remove, e_add)
                    adj_bits, deg, cur_score = new_adj, new_deg, new_score
                    edges.remove(e_remove)
                    nonedges.append(e_remove)
                    nonedges.remove(e_add)
                    edges.append(e_add)

            new_adj, new_deg, new_score = _run_sa(adj_bits, deg, cur_score, t0, rng, time_per_restart)

            if new_score < best_score:
                best_adj = new_adj
                best_deg = new_deg
                best_score = new_score

    # Phase 3: Deterministic polish
    if elapsed() < DEADLINE - 5.0:
        adj_bits = best_adj[:]
        deg = best_deg[:]
        cur_score = best_score
        new_adj, new_deg, new_score = _polish(adj_bits, deg, cur_score, t0, DEADLINE - 5.0, rng)
        if new_score < best_score:
            best_adj = new_adj
            best_deg = new_deg
            best_score = new_score

    # Final verification
    final_deg = _compute_degrees(best_adj)
    final_score = _max_tri_deg_sum(best_adj, final_deg)

    result_edges = [[i, j] for i in range(N) for j in range(i + 1, N)
                    if (best_adj[i] >> j) & 1]

    assert len(result_edges) == TARGET_EDGES, f"Expected {TARGET_EDGES} edges, got {len(result_edges)}"

    return {'n': N, 'edges': result_edges}
