import math
import time
import sys
from itertools import permutations

def construct(n):
    # Classic construction: length = sum_{k=1}^{n} k!
    word = [1]
    for m in range(2, n + 1):
        # Collect all distinct (m-1)-length windows that are permutations of 1..m-1
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap between end of new and start of block
            overlap = 0
            max_o = min(len(new), len(block) - 1)
            for o in range(max_o, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    # Local search: iteratively delete characters if the superpermutation property is preserved
    if n <= 7:
        start_time = time.time()
        deadline = 145.0  # 5s safety margin before 150s kill
        target_count = int(math.factorial(n))

        N = len(word)
        num_windows = N - n + 1

        # Build perm_counts: dict mapping permutation tuple -> count of occurrences
        perm_counts = {}
        for i in range(num_windows):
            w = word[i:i + n]
            if len(set(w)) == n:
                p = tuple(w)
                perm_counts[p] = perm_counts.get(p, 0) + 1

        # Verify initial validity
        assert len(perm_counts) == target_count, f"Initial word invalid: {len(perm_counts)}/{target_count}"

        improved = True
        while improved and time.time() < deadline:
            improved = False
            N = len(word)
            num_windows = N - n + 1

            for i in range(N - 1, -1, -1):
                if time.time() > deadline:
                    break

                # Identify windows in old word that contain position i
                win_lo = max(0, i - n + 1)
                win_hi = min(i, num_windows - 1)

                # Quick reject: if any valid permutation in these windows has count 1,
                # deleting i would destroy it (unless it reappears in a new window)
                # We'll handle this properly below, but this is a fast pre-filter
                can_proceed = True
                for j in range(win_lo, win_hi + 1):
                    w = word[j:j + n]
                    if len(set(w)) == n:
                        p = tuple(w)
                        if perm_counts[p] == 1:
                            # This permutation appears only here. Check if it can reappear
                            # in a new window after deletion. The new windows are those
                            # starting at positions [max(0, i-n+1), min(i-1, new_num_windows-1)]
                            # in the new word. We'll do the full check below.
                            pass  # Don't skip; let the full check handle it
                            break

                # Try deletion: new_word = word[:i] + word[i+1:]
                # Affected windows in new word: those crossing the gap at position i
                # In new_word, gap is between index i-1 and i
                # Window starting at j covers j..j+n-1; crosses gap if j <= i-1 and j+n-1 >= i
                # i.e., j >= i - n + 1
                new_N = N - 1
                new_num_windows = new_N - n + 1
                if new_num_windows <= 0:
                    continue

                new_win_lo = max(0, i - n + 1)
                new_win_hi = min(i - 1, new_num_windows - 1)

                # Build new_word slice for affected windows only (avoid full copy)
                # For window j in [new_win_lo, new_win_hi]:
                #   new_word[j:j+n] = word[j:i] + word[i+1:j+n+1] if j+n+1 <= N
                #   Actually: new_word[j] = word[j] for j < i, new_word[j] = word[j+1] for j >= i
                #   So new_word[j:j+n] = word[j:i] + word[i+1:j+n]  (length: (i-j) + (j+n-i) = n)
                #   Wait: word[j:i] has length i-j, word[i+1:j+n+1] has length j+n+1-(i+1) = j+n-i
                #   Total: i-j + j+n-i = n. Correct.

                # Update perm_counts: subtract old affected windows, add new affected windows
                # First, subtract old windows that contain position i
                for j in range(win_lo, win_hi + 1):
                    w = word[j:j + n]
                    if len(set(w)) == n:
                        p = tuple(w)
                        perm_counts[p] -= 1
                        if perm_counts[p] == 0:
                            del perm_counts[p]

                # Add new windows in the affected range
                for j in range(new_win_lo, new_win_hi + 1):
                    # new_word[j:j+n] = word[j:i] + word[i+1:j+n+1]
                    left = word[j:i]
                    right = word[i + 1:j + n + 1]
                    w = left + right
                    if len(set(w)) == n:
                        p = tuple(w)
                        perm_counts[p] = perm_counts.get(p, 0) + 1

                # Check if all target permutations are still present
                if len(perm_counts) == target_count:
                    # Commit the deletion
                    word = word[:i] + word[i + 1:]
                    improved = True
                    break  # Restart scan from the end
                else:
                    # Revert perm_counts
                    for j in range(new_win_lo, new_win_hi + 1):
                        left = word[j:i]
                        right = word[i + 1:j + n + 1]
                        w = left + right
                        if len(set(w)) == n:
                            p = tuple(w)
                            perm_counts[p] -= 1
                            if perm_counts[p] == 0:
                                del perm_counts[p]

                    for j in range(win_lo, win_hi + 1):
                        w = word[j:j + n]
                        if len(set(w)) == n:
                            p = tuple(w)
                            perm_counts[p] = perm_counts.get(p, 0) + 1

    return "".join(map(str, word))
