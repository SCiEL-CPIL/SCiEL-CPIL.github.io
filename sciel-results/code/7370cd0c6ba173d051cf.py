import itertools
import time
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142134213142314213241324123421342143214321"

    # Step 1: Build standard superpermutation (length = sum(i! for i=1..n))
    # For n=7: 1+2+6+24+120+720+5040 = 5913
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
            # Find max overlap with end of new
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if time.time() > deadline - 5:
        return "".join(map(str, word))

    # Step 2: Build frequency map of all n-windows
    L = len(word)
    window_freq = defaultdict(int)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        window_freq[p] += 1

    # Verify all 5040 permutations present
    required = set(itertools.permutations(range(1, n+1)))
    present = set(window_freq.keys())
    if present != required:
        return "".join(map(str, word))

    # Fast deletion check:
    def is_deletion_safe_fast(word, pos, n, window_freq, L):
        """Fast check if deleting position pos is safe."""
        new_len = L - 1
        if new_len < n:
            return False

        # Affected original window positions: those containing pos
        q_lo = max(0, pos - n + 1)
        q_hi = min(pos, L - n)

        # For each affected window, check if it's unique and if so, verify it reappears
        for q in range(q_lo, q_hi + 1):
            perm_q = tuple(word[q:q+n])
            if window_freq[perm_q] != 1:
                continue  # Not unique, safe

            # perm_q is unique and was at position q. Check if it appears in new word.
            # New windows are at positions w in [max(0, pos-n+1), min(pos, new_len-n)]
            # New window at position w (in new word) = word[w:pos] + word[pos+1:w+n+1]
            # (using original word indices, where w ranges over new word positions)

            # Check spanning windows
            w_lo = max(0, pos - n + 1)
            w_hi = min(pos - 1, L - n - 1)
            found = False
            for w in range(w_lo, w_hi + 1):
                # New window at position w: word[w:pos] + word[pos+1:w+n+1]
                # Check if this equals perm_q
                # perm_q = word[q:q+n]
                # We can compare piece by piece for speed
                # word[w:pos] should equal perm_q[0:pos-w]
                # word[pos+1:w+n+1] should equal perm_q[pos-w:n]
                left_len = pos - w
                if left_len < 0 or left_len > n:
                    continue
                # Compare left part
                match = True
                for i in range(left_len):
                    if word[w + i] != perm_q[i]:
                        match = False
                        break
                if not match:
                    continue
                # Compare right part
                for i in range(n - left_len):
                    if word[pos + 1 + i] != perm_q[left_len + i]:
                        match = False
                        break
                if match:
                    found = True
                    break

            if not found:
                return False

        return True

    # Step 3: Iterative deletion with fast verification
    def rebuild_data(word, n):
        L = len(word)
        window_freq = defaultdict(int)
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            window_freq[p] += 1
        return L, window_freq

    L, window_freq = rebuild_data(word, n)
    deleted_count = 0

    # Repeatedly find and delete safe positions
    while time.time() < deadline - 25:
        L = len(word)
        if L <= n:
            break

        # Find all positions with minimal "risk" (fewest unique windows containing them)
        unique_count_at = [0] * L
        for i in range(L - n + 1):
            if window_freq[tuple(word[i:i+n])] == 1:
                for j in range(i, i + n):
                    unique_count_at[j] += 1

        # Try positions from lowest unique count, skipping ends
        # Use a simple approach: try all positions in order of unique_count
        min_count = min(unique_count_at[1:-1]) if L > 2 else 0
        candidates = [pos for pos in range(1, L - 1) if unique_count_at[pos] == min_count]

        # If all have high count, try the minimum anyway
        if not candidates:
            candidates = [pos for pos in range(1, L - 1)]

        made_progress = False
        for pos in candidates:
            if time.time() > deadline - 25:
                break
            if pos >= len(word):
                continue

            if is_deletion_safe_fast(word, pos, n, window_freq, L):
                word = word[:pos] + word[pos+1:]
                deleted_count += 1
                made_progress = True
                # Rebuild data structures
                L, window_freq = rebuild_data(word, n)
                break

        if not made_progress:
            # No more deletions possible with current strategy
            break

    # Step 4: End trimming
    if time.time() < deadline - 15:
        changed = True
        while changed:
            changed = False
            L = len(word)
            if L > n:
                first_perm = tuple(word[:n])
                count = 0
                for i in range(L - n + 1):
                    if tuple(word[i:i+n]) == first_perm:
                        count += 1
                if count > 1:
                    word = word[1:]
                    changed = True

            if L > n:
                L = len(word)
                last_perm = tuple(word[-n:])
                count = 0
                for i in range(L - n + 1):
                    if tuple(word[i:i+n]) == last_perm:
                        count += 1
                if count > 1:
                    word = word[:-1]
                    changed = True

    # Step 5: Second round of deletions after end trimming
    if time.time() < deadline - 20:
        L, window_freq = rebuild_data(word, n)
        while time.time() < deadline - 20:
            L = len(word)
            if L <= n:
                break

            unique_count_at = [0] * L
            for i in range(L - n + 1):
                if window_freq[tuple(word[i:i+n])] == 1:
                    for j in range(i, i + n):
                        unique_count_at[j] += 1

            min_count = min(unique_count_at[1:-1]) if L > 2 else 0
            candidates = [pos for pos in range(1, L - 1) if unique_count_at[pos] == min_count]
            if not candidates:
                candidates = list(range(1, L - 1))

            made_progress = False
            for pos in candidates:
                if time.time() > deadline - 20:
                    break
                if pos >= len(word):
                    continue

                if is_deletion_safe_fast(word, pos, n, window_freq, L):
                    word = word[:pos] + word[pos+1:]
                    made_progress = True
                    L, window_freq = rebuild_data(word, n)
                    break

            if not made_progress:
                break

    # Step 6: Final end trimming
    if time.time() < deadline - 10:
        changed = True
        while changed:
            changed = False
            L = len(word)
            if L > n:
                first_perm = tuple(word[:n])
                count = 0
                for i in range(L - n + 1):
                    if tuple(word[i:i+n]) == first_perm:
                        count += 1
                if count > 1:
                    word = word[1:]
                    changed = True

            if L > n:
                L = len(word)
                last_perm = tuple(word[-n:])
                count = 0
                for i in range(L - n + 1):
                    if tuple(word[i:i+n]) == last_perm:
                        count += 1
                if count > 1:
                    word = word[:-1]
                    changed = True

    # Step 7: Third round of deletions
    if time.time() < deadline - 15:
        L, window_freq = rebuild_data(word, n)
        while time.time() < deadline - 15:
            L = len(word)
            if L <= n:
                break

            unique_count_at = [0] * L
            for i in range(L - n + 1):
                if window_freq[tuple(word[i:i+n])] == 1:
                    for j in range(i, i + n):
                        unique_count_at[j] += 1

            min_count = min(unique_count_at[1:-1]) if L > 2 else 0
            candidates = [pos for pos in range(1, L - 1) if unique_count_at[pos] == min_count]
            if not candidates:
                candidates = list(range(1, L - 1))

            made_progress = False
            for pos in candidates:
                if time.time() > deadline - 15:
                    break
                if pos >= len(word):
                    continue

                if is_deletion_safe_fast(word, pos, n, window_freq, L):
                    word = word[:pos] + word[pos+1:]
                    made_progress = True
                    L, window_freq = rebuild_data(word, n)
                    break

            if not made_progress:
                break

    # Step 8: Final end trimming
    if time.time() < deadline - 5:
        changed = True
        while changed:
            changed = False
            L = len(word)
            if L > n:
                first_perm = tuple(word[:n])
                count = 0
                for i in range(L - n + 1):
                    if tuple(word[i:i+n]) == first_perm:
                        count += 1
                if count > 1:
                    word = word[1:]
                    changed = True

            if L > n:
                L = len(word)
                last_perm = tuple(word[-n:])
                count = 0
                for i in range(L - n + 1):
                    if tuple(word[i:i+n]) == last_perm:
                        count += 1
                if count > 1:
                    word = word[:-1]
                    changed = True

    # Final verification
    final_perms = set()
    for i in range(len(word) - n + 1):
        final_perms.add(tuple(word[i:i+n]))

    if len(final_perms) != len(required):
        # Fallback: return the standard construction
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
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new

    return "".join(map(str, word))
