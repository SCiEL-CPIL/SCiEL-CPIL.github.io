import time
import random
import numpy as np

N = 1000
MAX_SUM = 2 * N          # 2000
K = 2                    # max representations allowed


def _sum_counts_vec(chosen):
    """Vectorized count of representations a<=b for each sum."""
    a = np.asarray(chosen, dtype=np.int64)
    if a.size == 0:
        return np.zeros(MAX_SUM + 1, dtype=np.int64)
    iu, ju = np.triu_indices(a.size)
    sums = a[iu] + a[ju]
    counts = np.zeros(MAX_SUM + 1, dtype=np.int64)
    np.add.at(counts, sums, 1)
    return counts


def _validity_vec(counts):
    """max count and the set of 'full' (count==K) sums."""
    mx = int(counts.max())
    full = np.flatnonzero(counts >= K)
    return mx, full


def _can_add_vec(x, chosen_arr, counts):
    """True if adding x keeps every sum <= K. Vectorized."""
    s = chosen_arr + x
    if np.any(counts[s] >= K):
        return False
    s2 = 2 * x
    if counts[s2] >= K:
        return False
    return True


def _add_vec(x, chosen_arr, counts):
    """Return new chosen array (appended) and updated counts after adding x."""
    s = chosen_arr + x
    counts[s] += 1
    counts[2 * x] += 1
    return np.concatenate([chosen_arr, np.array([x], dtype=np.int64)]), counts


def _remove_vec(x, chosen_arr, counts):
    """Return new chosen array and counts after removing x (assumed present)."""
    mask = chosen_arr != x
    s = chosen_arr[mask] + x
    counts[s] -= 1
    counts[2 * x] -= 1
    return chosen_arr[mask], counts


def _saturating_greedy(order, counts):
    """Greedy that adds ANY valid element (order decides which first)."""
    chosen = np.empty(0, dtype=np.int64)
    for x in order:
        if _can_add_vec(x, chosen, counts):
            chosen, counts = _add_vec(x, chosen, counts)
    return chosen, counts


def _local_search(chosen, counts, rng, deadline, max_iter=2000000):
    """Grow/swap local search with fast vectorized validity checks.

    Moves:
      (A) INSERT a new element (grows size) -- primary objective.
      (B) SWAP one out for one in (keeps size, may enable later growth).
    """
    n = len(chosen)
    chosen_set = set(int(v) for v in chosen)
    best_size = n
    best_chosen = chosen.copy()
    best_counts = counts.copy()

    all_nums = np.arange(1, N + 1, dtype=np.int64)
    avail = np.array([v for v in range(1, N + 1) if v not in chosen_set], dtype=np.int64)
    rng.shuffle(avail)

    start = time.time()
    for it in range(max_iter):
        if (it & 511) == 0:
            if time.time() > deadline:
                break
        if avail.size == 0:
            break

        # pick a random available candidate
        x_add = int(avail[rng.randrange(avail.size)])
        if x_add in chosen_set:
            continue

        if _can_add_vec(x_add, chosen, counts):
            # INSERT: pure growth
            chosen, counts = _add_vec(x_add, chosen, counts)
            chosen_set.add(x_add)
            avail = avail[avail != x_add]
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_chosen = chosen.copy()
                best_counts = counts.copy()
            continue

        # cannot insert directly -> try a SWAP to free capacity
        idx = rng.randrange(len(chosen))
        x_rm = int(chosen[idx])

        # counts after removing x_rm
        c2 = counts.copy()
        s_rm = chosen[chosen != x_rm] + x_rm
        c2[s_rm] -= 1
        c2[2 * x_rm] -= 1

        # can we add x_add under c2?
        s_add = chosen[chosen != x_rm] + x_add
        if np.any(c2[s_add] >= K) or c2[2 * x_add] >= K:
            continue

        # apply swap
        chosen, counts = _add_vec(x_add, chosen[chosen != x_rm], c2)
        chosen_set.discard(x_rm)
        chosen_set.add(x_add)
        if x_add in avail:
            avail = avail[avail != x_add]
        if x_rm not in avail:
            avail = np.concatenate([avail, np.array([x_rm], dtype=np.int64)])
        # size unchanged; do not update best_size

    return best_chosen, best_counts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0   # safety margin vs 150s kill

    rng = random.Random(42)

    best_chosen = np.empty(0, dtype=np.int64)
    best_counts = np.zeros(MAX_SUM + 1, dtype=np.int64)
    best_size = 0

    # --- Phase 1: multiple greedy seeds (cheap, fast) ---
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
    ]
    for seed in (123, 7, 999, 2024):
        o = list(range(1, N + 1))
        r = random.Random(seed)
        r.shuffle(o)
        orders.append(o)

    for o in orders:
        ch, ct = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64))
        if len(ch) > best_size:
            best_size = len(ch)
            best_chosen = ch
            best_counts = ct

    # --- Phase 2: local search from the best seed (growth + swap) ---
    ch, ct = _local_search(best_chosen, best_counts, rng, deadline, max_iter=2000000)
    if len(ch) > best_size:
        best_size = len(ch)
        best_chosen = ch
        best_counts = ct

    # --- Phase 3: if time remains, try a couple more seeds + search ---
    for trial in range(4):
        if time.time() > deadline:
            break
        o = list(range(1, N + 1))
        rt = random.Random(5000 + trial * 13)
        rt.shuffle(o)
        ch2, ct2 = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64))
        if len(ch2) <= best_size:
            continue
        ch2, ct2 = _local_search(ch2, ct2, rng, deadline, max_iter=200000)
        if len(ch2) > best_size:
            best_size = len(ch2)
            best_chosen = ch2
            best_counts = ct2

    return [int(v) for v in best_chosen]


# EVOLVE-BLOCK-END
