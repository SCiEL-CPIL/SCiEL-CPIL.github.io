"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    start_time = time.time()
    DEADLINE = start_time + 145.0

    def time_left():
        return DEADLINE - time.time()

    E = int(np.sum(adj)) // 2
    degrees = adj.sum(axis=1)

    def cut_size(labels):
        x = 2 * labels.astype(np.float64) - 1
        xTadjx = x @ (adj @ x)
        return E - 0.5 * xTadjx

    def fast_cut_delta(labels, v):
        same_side = adj[v][labels == labels[v]].sum()
        opp_side = adj[v][labels != labels[v]].sum()
        return same_side - opp_side

    def local_search(labels, max_rounds=50):
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        current_cut = best_cut

        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break
            improved = False
            for v in range(n):
                if time_left() < 1.0:
                    break
                delta = fast_cut_delta(labels, v)
                if delta > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += delta
                    improved = True
            if not improved:
                break

        final_cut = cut_size(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
        return best_labels, best_cut

    def goe_heuristic(labels):
        unassigned = list(range(n))
        result = labels.copy()
        assigned = np.zeros(n, dtype=bool)

        for _ in range(n):
            if time_left() < 2.0:
                break
            best_v = -1
            best_deg = -1
            for v in unassigned:
                if not assigned[v]:
                    deg = degrees[v]
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v

            if best_v == -1:
                break

            neighbors = np.where(adj[best_v] == 1)[0]
            count_side0 = 0
            count_side1 = 0
            for nb in neighbors:
                if not assigned[nb]:
                    if result[nb] == 0:
                        count_side0 += 1
                    else:
                        count_side1 += 1

            if count_side1 >= count_side0:
                result[best_v] = 0
            else:
                result[best_v] = 1

            assigned[best_v] = True

        return result

    def simulated_annealing(labels, max_iters=200000, T_start=8.0, T_end=0.01, alpha=0.99995):
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        current_cut = best_cut
        T = T_start

        for it in range(max_iters):
            if time_left() < 3.0:
                break

            v = np.random.randint(0, n)
            delta = fast_cut_delta(labels, v)

            if delta >= 0:
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                if T > 0.001:
                    if delta > -100 * T:
                        if np.random.random() < np.exp(delta / T):
                            labels[v] = 1 - labels[v]
                            current_cut += delta

            T = max(T * alpha, T_end)

        return best_labels, best_cut

    def multi_start_local(num_starts):
        best_labels = None
        best_cut = -1
        for s in range(num_starts):
            if time_left() < 3.0:
                break
            labels = np.random.randint(0, 2, size=n)
            labels, cut = local_search(labels, max_rounds=30)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        return best_labels, best_cut

    # Strategy:
    # 1. Start with alternating pattern, run local search
    # 2. Try GOE heuristic
    # 3. Try multiple random restarts with local search
    # 4. If time permits, run simulated annealing from best solution
    # 5. Final polish with local search

    # Step 1: Alternating start
    labels_alt = np.arange(n) % 2
    best_labels, best_cut = local_search(labels_alt, max_rounds=50)

    # Step 2: GOE heuristic
    labels_goe = goe_heuristic(np.zeros(n, dtype=int))
    labels_goe_ls, cut_goe = local_search(labels_goe, max_rounds=50)
    if cut_goe > best_cut:
        best_cut = cut_goe
        best_labels = labels_goe_ls

    # Step 3: Random restarts with local search
    if time_left() > 10.0:
        labels_rs, cut_rs = multi_start_local(15)
        if cut_rs > best_cut:
            best_cut = cut_rs
            best_labels = labels_rs

    # Step 4: Simulated annealing from best solution
    if time_left() > 20.0:
        labels_sa, cut_sa = simulated_annealing(best_labels, max_iters=100000, T_start=8.0, T_end=0.01)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa

        # Try SA from a random start
        if time_left() > 10.0:
            labels_rand = np.random.randint(0, 2, size=n)
            labels_sa2, cut_sa2 = simulated_annealing(labels_rand, max_iters=50000, T_start=10.0, T_end=0.01)
            if cut_sa2 > best_cut:
                best_cut = cut_sa2
                best_labels = labels_sa2

    # Step 5: Final polish
    if time_left() > 3.0:
        for _ in range(3):
            if time_left() < 1.5:
                break
            labels_polish, cut_polish = local_search(best_labels, max_rounds=30)
            if cut_polish > best_cut:
                best_cut = cut_polish
                best_labels = labels_polish
            else:
                break

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
