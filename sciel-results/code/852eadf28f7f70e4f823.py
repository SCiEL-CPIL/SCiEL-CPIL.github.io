import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin under 90s wall-clock kill

    # ---------- Precompute critical APs (short ones dominate max discrepancy) ----------
    # We focus on APs of length 1..Lmax with step <= Dmax to keep the constraint set manageable.
    # Longer APs are implicitly bounded by the sum of shorter ones if short ones are small,
    # but we still verify the true max at the end.
    Lmax = 8      # max AP length to explicitly constrain
    Dmax = 16     # max step to explicitly constrain

    critical_aps = []  # list of (start, step, length, indices)
    for d in range(1, Dmax + 1):
        for start in range(1, n + 1):
            # compute how many terms fit
            length = 1
            while start + length * d <= n and length < Lmax:
                length += 1
            if length >= 1:
                idxs = tuple(start - 1 + k * d for k in range(length))  # 0-based
                critical_aps.append((start, d, length, idxs))

    # Index APs by their indices for quick lookup
    # Map each position to list of APs containing it (for constraint-directed repair)
    pos_to_aps = [[] for _ in range(n)]
    for ap_idx, (s, d, L, idxs) in enumerate(critical_aps):
        for pos in idxs:
            pos_to_aps[pos].append(ap_idx)

    m = len(critical_aps)

    def compute_ap_sums(arr):
        """Vectorized-ish computation of signed sums for all critical APs."""
        sums = np.empty(m, dtype=np.int32)
        for i, (s, d, L, idxs) in enumerate(critical_aps):
            sums[i] = arr[idxs].sum()
        return sums

    def true_max_disc(arr):
        """True max absolute sum over ALL APs in [1,n] (for final verification)."""
        max_disc = 0
        for d in range(1, n + 1):
            for r in range(d):
                idx = np.arange(r, n, d)
                if len(idx) == 0:
                    continue
                cs = np.concatenate(([0], np.cumsum(arr[idx])))
                disc = int(cs.max() - cs.min())
                if disc > max_disc:
                    max_disc = disc
        return max_disc

    # ---------- Initialization: Thue-Morse ----------
    def thue_morse(n):
        seq = []
        for i in range(n):
            c = bin(i).count('1')
            seq.append(-1 if c % 2 == 0 else 1)
        return seq

    best_seq = thue_morse(n)
    best_arr = np.array(best_seq, dtype=np.int8)
    best_true = true_max_disc(best_arr)

    rng = random.Random(12345)

    # ---------- CSP-style backtracking with forward checking to satisfy bound D ----------
    def backtrack_satisfy(arr, D, max_nodes=20000):
        """
        Randomized backtracking: assign signs one by one; at each step, check that
        no critical AP that is now "fully assigned" violates |sum| > D.
        Returns True if a full assignment satisfying all critical AP constraints is found.
        """
        # We'll do a randomized variable ordering: pick unassigned positions in random order
        # But for speed, we do a simple recursive DFS with random restarts on variable order.
        # Given time constraints, we do a limited-depth repair instead of full backtracking.
        # This function is a placeholder for the repair strategy below.
        return False

    # ---------- Constraint-directed repair loop ----------
    def repair(arr, D, max_iters=5000):
        """
        Iteratively find the worst critical AP (|sum| > D) and flip the position in it
        that most reduces the sum (or is in the most-violating APs). Repeat until
        all critical APs satisfy |sum| <= D or max_iters reached.
        Returns (success, arr).
        """
        cur = arr.copy()
        sums = compute_ap_sums(cur)
        for it in range(max_iters):
            # Find the most violated AP
            abs_sums = np.abs(sums)
            worst_idx = int(np.argmax(abs_sums))
            if abs_sums[worst_idx] <= D:
                return True, cur  # all critical APs satisfied

            s, d, L, idxs = critical_aps[worst_idx]
            cur_sum = int(sums[worst_idx])

            # Try flipping each position in this AP; pick the one that most reduces |sum|
            best_flip = None
            best_new_abs = abs(cur_sum)
            for pos in idxs:
                # Flipping pos changes sum by -2*cur[pos]
                new_sum = cur_sum - 2 * int(cur[pos])
                new_abs = abs(new_sum)
                if new_abs < best_new_abs:
                    best_new_abs = new_abs
                    best_flip = pos

            if best_flip is None:
                # No flip in this AP helps; pick a random position in it
                best_flip = idxs[rng.randrange(L)]

            # Perform the flip
            cur[best_flip] *= -1

            # Update sums for all APs containing best_flip
            for ap_i in pos_to_aps[best_flip]:
                ss, dd, LL, ixs = critical_aps[ap_i]
                sums[ap_i] = cur[ixs].sum()

        return False, cur

    # ---------- Main optimization: iterative bound tightening ----------
    current_D = best_true
    # Start with a slightly relaxed bound and tighten
    for attempt in range(30):
        if time.time() > deadline:
            break

        # Try to satisfy bound current_D - 1 (i.e., lower the max by 1)
        target_D = current_D - 1
        if target_D < 1:
            break

        # Run multiple repair attempts with different randomizations
        found = False
        for rep in range(5):
            if time.time() > deadline:
                break
            # Start from current best, add some random perturbation to escape local minima
            trial = best_arr.copy()
            # Perturb a few random positions
            for _ in range(rng.randrange(1, 6)):
                pos = rng.randrange(n)
                trial[pos] *= -1

            success, trial = repair(trial, target_D, max_iters=3000)
            if success:
                # Verify true max discrepancy
                true_val = true_max_disc(trial)
                if true_val <= target_D:
                    best_arr = trial
                    best_seq = trial.tolist()
                    best_true = true_val
                    current_D = true_val
                    found = True
                    break
                elif true_val < best_true:
                    # Even if not fully satisfying target, it's an improvement
                    best_arr = trial
                    best_seq = trial.tolist()
                    best_true = true_val
                    current_D = true_val
                    found = True
                    break

        if not found:
            # Try a few more aggressive perturbations
            for rep in range(3):
                if time.time() > deadline:
                    break
                trial = best_arr.copy()
                for _ in range(rng.randrange(5, 15)):
                    pos = rng.randrange(n)
                    trial[pos] *= -1
                success, trial = repair(trial, target_D, max_iters=5000)
                if success:
                    true_val = true_max_disc(trial)
                    if true_val < best_true:
                        best_arr = trial
                        best_seq = trial.tolist()
                        best_true = true_val
                        current_D = true_val
                        found = True
                        break

        if not found and current_D <= 3:
            # If we're at a very low bound, try direct local search polish
            break

    # ---------- Final greedy polish ----------
    if time.time() < deadline:
        polish_arr = best_arr.copy()
        polish_cost = best_true
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                polish_arr[i] *= -1
                new_c = true_max_disc(polish_arr)
                if new_c < polish_cost:
                    polish_cost = new_c
                    best_arr = polish_arr.copy()
                    best_seq = polish_arr.tolist()
                    improved = True
                else:
                    polish_arr[i] *= -1

    return best_seq
