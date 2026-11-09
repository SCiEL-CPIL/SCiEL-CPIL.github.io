import time
import random
import numpy as np
import math

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin before 90s kill

    # Precompute APs
    ap_indices = []
    for d in range(1, n + 1):
        for a in range(n):
            idx = np.arange(a, n, d)
            if len(idx) > 0:
                ap_indices.append(idx)
    M = len(ap_indices)

    # For each position, which APs contain it
    affected_aps = [[] for _ in range(n)]
    for k, idx in enumerate(ap_indices):
        for pos in idx:
            affected_aps[pos].append(k)

    # For each AP, store its indices as a list
    ap_lists = [list(idx) for idx in ap_indices]

    # Precompute for each AP: the sorted list of positions
    # and for each position in the AP, its index within the AP
    ap_pos_in_ap = {}  # (ap_k, pos) -> index within ap
    for k, idx_list in enumerate(ap_lists):
        for j, pos in enumerate(idx_list):
            ap_pos_in_ap[(k, pos)] = j

    # For each position, precompute the APs it affects AND for each such AP,
    # what the "next unfilled position" would be after filling this one.
    # Actually, for forward checking: after assigning position i, we need to
    # check all APs containing i where i is the maximum position assigned so far
    # in that AP (i.e., all positions in that AP up to i are now assigned).
    #
    # Simpler approach: after assigning position i, for each AP k containing i,
    # check if all positions in ap_lists[k] that are <= i are now assigned.
    # If so, the prefix sum is determined; prune if |sum| > B.
    #
    # Even simpler: maintain ap_sums[k] as the sum of assigned signs in AP k.
    # Also maintain ap_count[k] = number of assigned positions in AP k.
    # After assigning position i, for each AP k containing i:
    #   ap_sums[k] += sign, ap_count[k] += 1
    #   The prefix of AP k up to position i is fully assigned iff
    #   all positions in ap_lists[k] with value <= i are assigned.
    #   We can check: is the next position in AP k after i unassigned?
    #   If i is the last assigned position in the prefix, the prefix sum = ap_sums[k]
    #   only if ALL positions in AP k up to i are assigned.
    #
    # Let me use a cleaner approach:
    # For each AP k, track which positions are assigned.
    # After assigning position i, for each AP k containing i:
    #   Check if all positions in ap_lists[k] that are < i are assigned.
    #   If yes, then the prefix ending at i is complete, and its sum is
    #   the sum of all assigned positions in AP k that are <= i.
    #   Since we assign in some order, this is tricky.
    #
    # ALTERNATIVE: Assign positions in natural order 0,1,2,...,n-1.
    # Then after assigning position i, for each AP k containing i:
    #   The prefix of AP k up to i is fully assigned iff all positions
    #   in ap_lists[k] with value <= i are in {0,...,i}.
    #   Since we're assigning in order 0..i, all positions <= i are assigned.
    #   So the prefix sum = sum of s[j] for j in ap_lists[k] if j <= i.
    #   We can maintain this incrementally.
    #
    # BEST APPROACH: Assign in natural order. Maintain ap_sums[k] = sum of
    # assigned signs in AP k. After assigning position i, for each AP k
    # containing i: update ap_sums[k]. Then check: is i the current "frontier"
    # of AP k? I.e., are all positions in AP k with value <= i now assigned?
    # Yes, because we assign in order. So the prefix sum of AP k up to i is
    # exactly ap_sums[k] (since all positions <= i in AP k are assigned).
    # Wait, no: ap_sums[k] includes ALL assigned positions in AP k, which
    # might include positions > i if we've assigned them. But we assign in
    # order 0,1,...,i, so no positions > i are assigned yet. So ap_sums[k]
    # = sum of s[j] for j in ap_lists[k] with j <= i. This is the prefix sum.
    # So we just check |ap_sums[k]| <= B for all APs k containing i.

    # Assign in natural order: 0, 1, 2, ..., n-1
    # This makes forward checking simple.

    def backtrack(B, depth, s, ap_sums, ap_count):
        """
        depth: next position to assign (0-indexed)
        s: array of signs, s[0..depth-1] assigned
        ap_sums: current sum for each AP
        ap_count: current count of assigned positions in each AP
        Returns True if a valid completion exists.
        """
        if depth == n:
            return True

        i = depth
        # Try +1 and -1
        for sign in (1, -1):
            s[i] = sign
            # Update APs containing i
            affected = affected_aps[i]
            for k in affected:
                ap_sums[k] += sign
                ap_count[k] += 1
                # Check prefix: since we assign in natural order, all positions
                # in AP k with value <= i are now assigned.
                # The prefix sum is ap_sums[k].
                if abs(ap_sums[k]) > B:
                    # Undo and try other sign
                    ap_sums[k] -= sign
                    ap_count[k] -= 1
                    break
            else:
                # All APs OK, recurse
                if backtrack(B, depth + 1, s, ap_sums, ap_count):
                    return True
                # Undo
                for k in affected:
                    ap_sums[k] -= sign
                    ap_count[k] -= 1
                continue
            # Undo (if we broke out of the for loop)
            for k in affected:
                ap_sums[k] -= sign
                ap_count[k] -= 1
            s[i] = 0
        return False

    def backtrack_with_timeout(B, depth, s, ap_sums, ap_count, deadline):
        """Backtracking with time check."""
        if time.time() > deadline:
            return False
        if depth == n:
            return True

        i = depth
        for sign in (1, -1):
            if time.time() > deadline:
                return False
            s[i] = sign
            affected = affected_aps[i]
            pruned = False
            for k in affected:
                ap_sums[k] += sign
                ap_count[k] += 1
                if abs(ap_sums[k]) > B:
                    pruned = True
                    break
            if pruned:
                for k in affected:
                    ap_sums[k] -= sign
                    ap_count[k] -= 1
                s[i] = 0
                continue
            # Recurse
            if backtrack_with_timeout(B, depth + 1, s, ap_sums, ap_count, deadline):
                return True
            # Undo
            for k in affected:
                ap_sums[k] -= sign
                ap_count[k] -= 1
            s[i] = 0
        return False

    # Try iterative deepening on B
    best_B = None
    best_s = None

    for B in range(1, 10):
        s = np.zeros(n, dtype=np.int8)
        ap_sums = np.zeros(M, dtype=np.int32)
        ap_count = np.zeros(M, dtype=np.int32)
        
        found = backtrack_with_timeout(B, 0, s, ap_sums, ap_count, deadline)
        if found:
            best_B = B
            best_s = s.copy()
            break

    if best_s is not None:
        return [int(x) for x in best_s]

    # If backtracking didn't find a solution in time, fall back to
    # a heuristic construction + local search
    # Thue-Morse initialization
    s = np.zeros(n, dtype=np.int8)
    for i in range(n):
        cnt = bin(i).count('1')
        s[i] = 1 if cnt % 2 == 0 else -1

    # Compute AP sums
    ap_sums = np.zeros(M, dtype=np.int32)
    for k in range(M):
        ap_sums[k] = np.sum(s[ap_indices[k]])
    current_max = int(np.max(np.abs(ap_sums)))
    best_max = current_max
    best_s = s.copy()

    rng = random.Random(42)

    # Greedy descent
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(n):
            old_sign = s[i]
            s[i] = -old_sign
            delta = -2 * old_sign
            for k in affected_aps[i]:
                ap_sums[k] += delta
            new_max = int(np.max(np.abs(ap_sums)))
            if new_max < current_max:
                current_max = new_max
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_s = s.copy()
            else:
                s[i] = old_sign
                for k in affected_aps[i]:
                    ap_sums[k] -= delta

    # SA polish
    T = 2.0
    T_min = 0.1
    cooling = 0.9999
    s = best_s.copy()
    ap_sums = np.zeros(M, dtype=np.int32)
    for k in range(M):
        ap_sums[k] = np.sum(s[ap_indices[k]])
    current_max = best_max

    while time.time() < deadline - 1.0 and T > T_min:
        i = rng.randint(0, n - 1)
        old_sign = s[i]
        s[i] = -old_sign
        delta = -2 * old_sign
        for k in affected_aps[i]:
            ap_sums[k] += delta
        new_max = int(np.max(np.abs(ap_sums)))
        if new_max <= current_max:
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_s = s.copy()
        else:
            delta_obj = new_max - current_max
            prob = math.exp(-delta_obj / T)
            if rng.random() < prob:
                current_max = new_max
            else:
                s[i] = old_sign
                for k in affected_aps[i]:
                    ap_sums[k] -= delta
        T *= cooling

    return [int(x) for x in best_s]
