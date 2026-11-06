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
      (C) KICK: remove two, add one (shrinks size by 1, may enable future growth).
    """
    n = len(chosen)
    chosen_set = set(int(v) for v in chosen)
    best_size = n
    best_chosen = chosen.copy()
    best_counts = counts.copy()

    avail = np.array([v for v in range(1, N + 1) if v not in chosen_set], dtype=np.int64)
    rng.shuffle(avail)

    # Precompute: for each x in chosen, the set of sums it contributes
    # We'll use a fast approach: for swap, check if removing x_rm frees enough capacity

    start = time.time()
    for it in range(max_iter):
        if (it & 255) == 0:
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
        # Efficient swap: find x_rm such that removing it makes x_add insertable
        # x_add conflicts with sums where counts[s] >= K and s = x_add + y for y in chosen
        # or s = 2*x_add
        conflict_sums = np.flatnonzero(counts >= K)
        # For each conflict sum s, the conflicting pairs are (x_add, s-x_add) where s-x_add in chosen
        # We need to remove at least one element from each conflicting pair
        # Simple approach: try random removals but be smarter about which ones to try

        # Identify which chosen elements are in conflicting pairs with x_add
        # A sum s is "full" if counts[s] == K. Adding x_add creates new pairs (x_add, y) for y in chosen
        # with sum x_add+y. If counts[x_add+y] >= K, we have a conflict.
        # Also 2*x_add if counts[2*x_add] >= K.

        # Find conflicting y values: y in chosen such that counts[x_add + y] >= K
        y_conflicts = chosen[np.maximum(counts[x_add + chosen], 0) >= K]
        # Also check 2*x_add
        if counts[2 * x_add] >= K:
            y_conflicts = np.concatenate([y_conflicts, np.array([x_add], dtype=np.int64)]) if x_add in chosen_set else y_conflicts

        if y_conflicts.size == 0:
            # Shouldn't happen if _can_add_vec returned False, but safety
            continue

        # Try removing one of the conflicting elements
        # Prefer removing elements that are in multiple conflicts (high "degree")
        if y_conflicts.size == 1:
            x_rm = int(y_conflicts[0])
        else:
            # Count how many conflicts each candidate removal resolves
            # For each candidate x_rm in y_conflicts, check if removing it makes x_add insertable
            # Fast check: after removing x_rm, can we add x_add?
            # The new counts would be counts - (contributions of x_rm)
            # x_rm contributes to sums: x_rm + y for y in chosen\{x_rm}, and 2*x_rm
            # We need: for all y in chosen\{x_rm}, counts[x_add+y] - (1 if x_rm+y == x_add+y else 0) < K
            # i.e., counts[x_add+y] < K for all y not equal to x_rm, AND counts[2*x_add] < K (if 2*x_add != x_rm+x_rm... wait)
            # Actually simpler: just test a few random candidates

            # Try up to 3 random candidates from y_conflicts
            n_try = min(3, y_conflicts.size)
            found = False
            for _ in range(n_try):
                x_rm = int(y_conflicts[rng.randrange(y_conflicts.size)])
                # Quick check: does removing x_rm resolve all conflicts?
                # After removing x_rm, the sums that change are: x_rm + y for y in chosen\{x_rm}, and 2*x_rm
                # New count for sum s = counts[s] - (number of pairs involving x_rm that sum to s)
                # For s = x_add + y (y in chosen\{x_rm}): new count = counts[s] - (1 if x_rm + ? = s... no)
                # Actually: removing x_rm decreases counts[x_rm + y] by 1 for each y in chosen\{x_rm}, and counts[2*x_rm] by 1
                # We need: for all y in chosen\{x_rm}, new_counts[x_add + y] < K and new_counts[2*x_add] < K
                # new_counts[x_add + y] = counts[x_add + y] - (1 if x_add + y == x_rm + y' for some y'... no, that's not right)
                # Let me think again. The count of sum s is the number of pairs (a,b) in chosen with a<=b and a+b=s.
                # Removing x_rm removes all pairs involving x_rm. So count_new[s] = count[s] - #{y in chosen\{x_rm}: x_rm + y = s} - #{x_rm: 2*x_rm = s}
                # = count[s] - #{y in chosen\{x_rm}: y = s - x_rm} - (1 if 2*x_rm == s else 0)
                # = count[s] - (1 if s - x_rm in chosen and s - x_rm != x_rm else 0) - (1 if 2*x_rm == s else 0)

                # For our purposes, we just need to check if x_add can be added after removing x_rm
                # This is equivalent to checking _can_add_vec(x_add, chosen_without_xrm, counts_without_xrm)
                # But we don't want to actually modify arrays. Let's do a fast check.

                # The sums that x_add would create: x_add + y for y in chosen\{x_rm}, and 2*x_add
                # For each such sum s, we need count_new[s] < K
                # count_new[s] = counts[s] - (1 if (s - x_rm) in chosen_set and (s - x_rm) != x_rm else 0) - (1 if 2*x_rm == s else 0)

                # Check 2*x_add first
                s2 = 2 * x_add
                c2 = counts[s2]
                if (s2 - x_rm) in chosen_set and (s2 - x_rm) != x_rm:
                    c2 -= 1
                if 2 * x_rm == s2:
                    c2 -= 1
                if c2 >= K:
                    continue

                # Check x_add + y for y in chosen\{x_rm}
                # We need counts[x_add + y] - (1 if x_add + y - x_rm in chosen_set and x_add + y - x_rm != x_rm else 0) - (1 if 2*x_rm == x_add + y else 0) < K
                # for all y in chosen\{x_rm}
                # This is: counts[x_add + y] < K + (1 if x_add + y - x_rm in chosen_set and x_add + y - x_rm != x_rm else 0) + (1 if 2*x_rm == x_add + y else 0)
                # Simplify: for y != x_rm, let s = x_add + y
                # The correction is: 1 if (s - x_rm) in chosen_set and s - x_rm != x_rm, plus 1 if 2*x_rm == s
                # Note: s - x_rm = x_add + y - x_rm. This is in chosen_set if x_add + y - x_rm is a chosen element.
                # And 2*x_rm == s means x_add + y == 2*x_rm, i.e., y == 2*x_rm - x_add.

                # Vectorized check:
                y_vals = chosen[chosen != x_rm]
                s_vals = x_add + y_vals
                # Correction 1: (s - x_rm) in chosen_set and s - x_rm != x_rm
                # s - x_rm = x_add + y - x_rm
                corr1 = np.zeros(y_vals.size, dtype=np.int64)
                diff = s_vals - x_rm
                # Check if diff is in chosen_set and diff != x_rm
                # We can use a set lookup but that's slow. Instead, use the fact that diff must be in [1, N] and in chosen_set
                # For now, approximate: just check if counts[s_vals] >= K and handle corrections
                # Actually, let's just do the direct check for a few candidates

                # Simpler: just test if _can_add would work by simulating
                # For speed, only check if the "worst" conflict is resolved
                # The most critical sums are those where counts[s] >= K
                conflict_s = s_vals[counts[s_vals] >= K]
                if conflict_s.size == 0:
                    # No conflicts remain (or were already below K)
                    found = True
                    break
                # Check if removing x_rm resolves these conflicts
                # For each conflict sum s in conflict_s:
                # new_count = counts[s] - (1 if s-x_rm in chosen_set and s-x_rm != x_rm else 0) - (1 if 2*x_rm == s else 0)
                # We need new_count < K, i.e., the correction must be >= counts[s] - K + 1
                all_resolved = True
                for s_i in conflict_s:
                    corr = 0
                    if (s_i - x_rm) in chosen_set and (s_i - x_rm) != x_rm:
                        corr += 1
                    if 2 * x_rm == s_i:
                        corr += 1
                    if counts[s_i] - corr >= K:
                        all_resolved = False
                        break
                if all_resolved:
                    found = True
                    break

            if not found:
                continue

        # Apply swap: remove x_rm, add x_add
        chosen_without = chosen[chosen != x_rm]
        # Update counts
        s_rm = chosen_without + x_rm
        counts[s_rm] -= 1
        counts[2 * x_rm] -= 1
        # Add x_add
        s_add = chosen_without + x_add
        counts[s_add] += 1
        counts[2 * x_add] += 1
        chosen = np.concatenate([chosen_without, np.array([x_add], dtype=np.int64)])
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
    deadline = start_time + 135.0   # safety margin vs 150s kill

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

    # --- Phase 3: if time remains, try more seeds + search with varied parameters ---
    for trial in range(8):
        if time.time() > deadline:
            break
        o = list(range(1, N + 1))
        rt = random.Random(5000 + trial * 13)
        rt.shuffle(o)
        ch2, ct2 = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64))
        if len(ch2) > best_size:
            ch2, ct2 = _local_search(ch2, ct2, rng, deadline, max_iter=300000)
            if len(ch2) > best_size:
                best_size = len(ch2)
                best_chosen = ch2
                best_counts = ct2
        elif len(ch2) >= best_size - 2:
            # Even if slightly smaller, local search might find a better arrangement
            ch2, ct2 = _local_search(ch2, ct2, rng, deadline, max_iter=150000)
            if len(ch2) > best_size:
                best_size = len(ch2)
                best_chosen = ch2
                best_counts = ct2

    # --- Phase 4: final polish - try to insert more elements from the best solution ---
    if time.time() < deadline - 5:
        ch4, ct4 = _local_search(best_chosen, best_counts, rng, deadline, max_iter=100000)
        if len(ch4) > best_size:
            best_size = len(ch4)
            best_chosen = ch4
            best_counts = ct4

    return [int(v) for v in best_chosen]


# EVOLVE-BLOCK-END
