"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Precompute some useful arrays
    # For faster delta computation, we can precompute row/col sums
    # but for now, let's use vectorized operations

    def compute_cost_fast(p):
        """Compute cost using vectorized matrix multiplication."""
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    def compute_swap_delta(i, j, p):
        """Compute cost delta when swapping facilities at indices i and j in permutation p."""
        li = p[i]
        lj = p[j]

        if li == lj:
            return 0

        # Diagonal terms
        delta = a[i, i] * (b[lj, lj] - b[li, li])
        delta += a[j, j] * (b[li, li] - b[lj, lj])

        # Cross terms between i and j
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])

        # Terms with other facilities k
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False

        ks = np.where(mask)[0]
        pk = p[ks]

        if len(ks) > 0:
            delta += np.sum(a[i, ks] * (b[lj, pk] - b[li, pk]))
            delta += np.sum(a[ks, i] * (b[pk, lj] - b[pk, li]))
            delta += np.sum(a[j, ks] * (b[li, pk] - b[lj, pk]))
            delta += np.sum(a[ks, j] * (b[pk, li] - b[pk, lj]))

        return delta

    def fast_local_search(p, current_cost, end_time):
        """
        Perform First Improvement 2-swap Local Search.
        Uses a more efficient iteration strategy.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost

        # Precompute all pairs for faster access
        # We'll iterate in a randomized order each pass
        indices = np.arange(n)

        while time.time() < end_time:
            improved = False
            # Shuffle the order of i indices
            shuffled_i = indices.copy()
            np.random.shuffle(shuffled_i)

            for i in shuffled_i:
                if time.time() > end_time:
                    return best_p, best_cost

                # For each i, try all j > i and j < i
                # To be efficient, we can precompute a list of j's
                j_indices = np.arange(n)
                j_indices = j_indices[j_indices != i]
                np.random.shuffle(j_indices)

                for j in j_indices:
                    if time.time() > end_time:
                        return best_p, best_cost

                    delta = compute_swap_delta(i, j, curr_p)

                    if delta < 0:
                        curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                        curr_cost += delta
                        improved = True

                        if curr_cost < best_cost:
                            best_cost = curr_cost
                            best_p = curr_p.copy()

                        # After a swap, re-shuffle j_indices to avoid bias
                        # But for efficiency, we just continue
                        break  # Restart j loop for this i after a swap

            if not improved:
                break

        return best_p, best_cost

    def tabu_search(p, current_cost, end_time, tabu_tenure=20):
        """
        Perform Tabu Search with 2-swap moves.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost

        rng = np.random.RandomState(789)

        # Tabu list: set of (i, j) pairs that are tabu
        # Use a dictionary with timestamps for expiration
        tabu_list = {}
        tabu_counter = 0

        while time.time() < end_time:
            tabu_counter += 1

            # Find the best non-tabu move
            best_delta = 0
            best_move = None
            best_aspiration = False

            # Sample a subset of moves to keep it fast
            # Try all pairs but with a limit
            i_indices = rng.choice(n, size=min(n, 30), replace=False)

            for i in i_indices:
                j_indices = rng.choice(n, size=min(n, 30), replace=False)
                for j in j_indices:
                    if i == j:
                        continue

                    # Check if tabu
                    key = (min(i, j), max(i, j))
                    if key in tabu_list and tabu_list[key] + tabu_tenure > tabu_counter:
                        # Check aspiration: if it improves best, allow it
                        delta = compute_swap_delta(i, j, curr_p)
                        if curr_cost + delta < best_cost:
                            best_aspiration = True
                        else:
                            continue

                    delta = compute_swap_delta(i, j, curr_p)

                    if best_aspiration or best_move is None or delta < best_delta:
                        best_delta = delta
                        best_move = (i, j)
                        best_aspiration = False

            if best_move is None:
                # No valid move found, try all pairs
                for i in range(n):
                    for j in range(i + 1, n):
                        key = (i, j)
                        if key in tabu_list and tabu_list[key] + tabu_tenure > tabu_counter:
                            continue
                        delta = compute_swap_delta(i, j, curr_p)
                        if best_move is None or delta < best_delta:
                            best_delta = delta
                            best_move = (i, j)

            if best_move is None:
                break

            i, j = best_move
            curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
            curr_cost += best_delta

            # Add to tabu list
            key = (min(i, j), max(i, j))
            tabu_list[key] = tabu_counter

            # Clean up old tabu entries
            if len(tabu_list) > 1000:
                tabu_list = {k: v for k, v in tabu_list.items() if v + tabu_tenure > tabu_counter}

            if curr_cost < best_cost:
                best_cost = curr_cost
                best_p = curr_p.copy()

        return best_p, best_cost

    def simulated_annealing(p, best_cost, end_time):
        """
        Perform Simulated Annealing to escape local minima.
        """
        curr_p = p.copy()
        curr_cost = best_cost
        best_p = p.copy()
        best_cost_sa = best_cost

        rng = np.random.RandomState(42)

        # Estimate initial temperature
        deltas = []
        for _ in range(20):
            i, j = rng.choice(n, 2, replace=False)
            d = compute_swap_delta(i, j, curr_p)
            deltas.append(abs(d))
        avg_delta = np.mean(deltas) if deltas else 1.0
        if avg_delta == 0:
            avg_delta = 1.0

        T = avg_delta * 15.0
        cooling_rate = 0.9995
        min_T = 1e-5

        while time.time() < end_time:
            if T < min_T:
                break

            # Generate random move
            i, j = rng.choice(n, 2, replace=False)

            delta = compute_swap_delta(i, j, curr_p)

            # Acceptance criterion
            if delta < 0:
                accept = True
            else:
                if T > 0:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                accept = rng.random_sample() < prob

            if accept:
                curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                curr_cost += delta

                if curr_cost < best_cost_sa:
                    best_cost_sa = curr_cost
                    best_p = curr_p.copy()

            T *= cooling_rate

        return best_p, best_cost_sa

    # 1. Hungarian Initialization
    C = a @ b.T
    row_ind, col_ind = linear_sum_assignment(-C)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind

    best_p = p_hung.copy()
    best_cost = compute_cost_fast(p_hung)

    # 2. Polish Hungarian solution with Local Search
    end_time_lk = deadline - 25.0
    p_lk, cost_lk = fast_local_search(best_p, best_cost, end_time_lk)
    if cost_lk < best_cost:
        best_p = p_lk
        best_cost = cost_lk

    # 3. Tabu Search from current best
    if time.time() < deadline - 15.0:
        end_time_ts = deadline - 10.0
        p_ts, cost_ts = tabu_search(best_p, best_cost, end_time_ts)
        if cost_ts < best_cost:
            best_p = p_ts
            best_cost = cost_ts

    # 4. Try Random Initialization + Local Search
    if time.time() < deadline - 10.0:
        rng_init = np.random.RandomState(123)
        p_rand = rng_init.permutation(n)
        cost_rand = compute_cost_fast(p_rand)

        end_time_lk2 = deadline - 8.0
        p_lk2, cost_lk2 = fast_local_search(p_rand, cost_rand, end_time_lk2)
        if cost_lk2 < best_cost:
            best_p = p_lk2
            best_cost = cost_lk2

    # 5. Perturb Best and Local Search
    if time.time() < deadline - 5.0:
        rng_perturb = np.random.RandomState(456)
        p_pert = best_p.copy()
        # Swap 5 random pairs to perturb
        for _ in range(5):
            i, j = rng_perturb.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]

        cost_pert = compute_cost_fast(p_pert)
        end_time_lk3 = deadline - 3.0
        p_lk3, cost_lk3 = fast_local_search(p_pert, cost_pert, end_time_lk3)
        if cost_lk3 < best_cost:
            best_p = p_lk3
            best_cost = cost_lk3

    # 6. Final SA Polish if time remains
    if time.time() < deadline - 1.0:
        p_sa, cost_sa = simulated_annealing(best_p, best_cost, deadline - 0.5)
        if cost_sa < best_cost:
            best_p = p_sa
            best_cost = cost_sa

    return best_p.tolist()
# EVOLVE-BLOCK-END
