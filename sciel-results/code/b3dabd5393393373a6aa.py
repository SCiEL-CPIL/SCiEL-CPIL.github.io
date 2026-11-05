import time
import random
import itertools


def _standard_superperm(n):
    """Classic recursive superpermutation construction."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                           if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word


def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    # 1. Generate classic recursive superpermutation
    word = _standard_superperm(n)
    S = list(word)

    # 2. Build initial counts: for each position i, count how many times
    #    the window S[i:i+n] appears as a valid permutation
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i + n])
        counts[p] = counts.get(p, 0) + 1

    # Verify all n! permutations are present
    all_perms = set(itertools.permutations(range(1, n + 1)))
    total_perms = len(all_perms)
    if len(counts) < total_perms:
        # Shouldn't happen, but return what we have
        return "".join(map(str, S))

    best_S = S[:]
    best_len = len(S)

    def is_valid_window(p):
        """Check if p is a valid permutation of 1..n."""
        return p in all_perms

    def try_remove(i):
        """
        Try to remove position i from S.
        Returns True if successful.
        
        Windows affected: those starting at positions [max(0, i-n+1), min(i, len(S)-n)]
        For each such window, if its count is 1 and it's a valid permutation, removal is invalid.
        """
        start_old = max(0, i - n + 1)
        end_old = min(i, len(S) - n)

        # Check validity: no unique valid permutation window can be destroyed
        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j + n])
            if p in all_perms and counts.get(p, 0) == 1:
                return False

        # Perform removal: decrement counts for old windows
        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j + n])
            if p in all_perms:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]

        # Remove the character
        S.pop(i)

        # Add back new straddling windows
        # After removal, windows that cross the gap at position i
        # New windows start at positions [start_old, min(i-1, len(S)-n)]
        start_new = max(0, i - n + 1)
        end_new = min(i - 1, len(S) - n)
        for k in range(start_new, end_new + 1):
            p = tuple(S[k:k + n])
            if p in all_perms:
                counts[p] = counts.get(p, 0) + 1

        return True

    def try_swap(i, j):
        """
        Try to swap positions i and j in S.
        Returns True if successful (swap is valid and performed).
        """
        if i == j:
            return False
        if i > j:
            i, j = j, i

        # Collect all affected window start positions
        affected = set()
        for pos in (i, j):
            s = max(0, pos - n + 1)
            e = min(pos, len(S) - n)
            for k in range(s, e + 1):
                affected.add(k)

        # Check validity: no unique valid permutation window changes
        old_i, old_j = S[i], S[j]
        for k in affected:
            old_p = tuple(S[k:k + n])
            if old_p in all_perms and counts.get(old_p, 0) == 1:
                # Reconstruct window after swap
                new_window = []
                for idx in range(k, k + n):
                    if idx == i:
                        new_window.append(old_j)
                    elif idx == j:
                        new_window.append(old_i)
                    else:
                        new_window.append(S[idx])
                new_p = tuple(new_window)
                if new_p != old_p:
                    return False

        # Perform swap: decrement old counts
        for k in affected:
            old_p = tuple(S[k:k + n])
            if old_p in all_perms:
                counts[old_p] -= 1
                if counts[old_p] == 0:
                    del counts[old_p]

        S[i], S[j] = S[j], S[i]

        # Add new counts
        for k in affected:
            new_p = tuple(S[k:k + n])
            if new_p in all_perms:
                counts[new_p] = counts.get(new_p, 0) + 1

        return True

    # 3. Local search: multi-pass removal with perturbation
    pass_count = 0
    stuck_passes = 0
    max_stuck = 20  # after this many stuck passes, give up on swaps

    while time.time() < deadline:
        pass_count += 1
        removals_in_pass = 0

        # Try removals in multiple orders
        for order in ['forward', 'backward', 'random']:
            if time.time() > deadline:
                break

            if order == 'forward':
                indices = list(range(len(S)))
            elif order == 'backward':
                indices = list(range(len(S) - 1, -1, -1))
            else:
                indices = list(range(len(S)))
                random.shuffle(indices)

            i = 0
            while i < len(indices):
                if time.time() > deadline:
                    break
                pos = indices[i]
                if pos >= len(S):
                    i += 1
                    continue
                if try_remove(pos):
                    removals_in_pass += 1
                    best_S = S[:]
                    best_len = len(S)
                    # Rebuild indices since length changed
                    if order == 'forward':
                        indices = list(range(pos, len(S)))
                        i = 0
                    elif order == 'backward':
                        indices = list(range(pos - 1, -1, -1))
                        i = 0
                    else:
                        indices = list(range(len(S)))
                        random.shuffle(indices)
                        i = 0
                else:
                    i += 1

        if removals_in_pass > 0:
            stuck_passes = 0
        else:
            stuck_passes += 1
            if stuck_passes >= max_stuck:
                break

            # Perturbation: try random swaps to unlock removals
            if time.time() > deadline:
                break

            swaps_tried = 0
            max_swaps = min(30, int((deadline - time.time()) * 30))

            while swaps_tried < max_swaps and time.time() < deadline:
                swaps_tried += 1
                i = random.randint(0, len(S) - 1)
                j = random.randint(0, len(S) - 1)
                if i == j:
                    continue

                if try_swap(i, j):
                    # After swap, try a few removals
                    removed_any = False
                    for k in range(len(S) - 1, -1, -1):
                        if try_remove(k):
                            removed_any = True
                            best_S = S[:]
                            best_len = len(S)
                            break

                    if not removed_any:
                        # Revert the swap
                        try_swap(i, j)

    # 4. Final high-precision polish pass
    # Try removals in many different orderings to find any remaining removable positions
    for _polish in range(10):
        if time.time() > deadline:
            break
        polish_order = list(range(len(S)))
        random.shuffle(polish_order)
        for pos in polish_order:
            if time.time() > deadline:
                break
            if pos >= len(S):
                continue
            if try_remove(pos):
                best_S = S[:]
                best_len = len(S)
                # Restart polish from beginning with new order
                break

    # Use the best result found
    if len(S) < len(best_S):
        best_S = S[:]

    return "".join(map(str, best_S))
