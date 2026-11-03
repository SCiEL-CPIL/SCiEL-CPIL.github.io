"""Shortest superpermutation on seven symbols — fast recursive build + incremental
local deletion search using window-count bookkeeping."""
import time
from itertools import permutations

# EVOLVE-BLOCK-START

def _build_recursive_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
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
            # find max overlap of new suffix with block prefix
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word


def _initial_window_counts(word, n):
    """Build dict: tuple -> count for every length-n window in word."""
    counts = {}
    L = len(word)
    for i in range(L - n + 1):
        w = tuple(word[i:i + n])
        counts[w] = counts.get(w, 0) + 1
    return counts


def _try_delete(word, counts, n, required_perms, pos):
    """
    Try to delete word[pos].
    Returns (new_word, new_counts, True) on success, (None, None, False) on failure.
    """
    L = len(word)
    if pos < 0 or pos >= L:
        return None, None, False

    # Windows that cover position pos:
    #   start indices s where s <= pos <= s + n - 1
    #   => s in [pos - n + 1, pos], clipped to [0, L - n]
    s_lo = max(0, pos - n + 1)
    s_hi = min(pos, L - n)

    # Check: no covered window has count == 1 (would be destroyed)
    for s in range(s_lo, s_hi + 1):
        w = tuple(word[s:s + n])
        if counts.get(w, 0) <= 1:
            return None, None, False

    # Remove word[pos]
    new_word = word[:pos] + word[pos + 1:]
    new_L = L - 1
    if new_L < n:
        return None, None, False

    new_counts = dict(counts)

    # Decrement counts for old windows that covered pos
    for s in range(s_lo, s_hi + 1):
        w = tuple(word[s:s + n])
        new_counts[w] = new_counts.get(w, 0) - 1
        if new_counts[w] <= 0:
            del new_counts[w]

    # Add new windows in new_word that span the deletion gap:
    # new windows starting at s' in [pos - n + 1, pos - 1] (clipped to [0, new_L - n])
    # where s' <= pos - 1 < s' + n - 1  i.e. s' in [pos - n, pos - 1]
    # But we already decremented the old ones; the new windows are different.
    # New windows in new_word starting at index s' correspond to:
    #   if s' < pos:  word[s':pos] + word[pos+1:s'+n]   (length n)
    #   if s' == pos: word[pos:pos+n-1] + word[pos+n]    -- wait, s' can be at most pos-1
    # Actually in new_word, indices after pos are shifted by 1.
    # A new window starting at s' in new_word:
    #   if s' + n - 1 <= pos - 1: entirely before the gap -> same as old window s'
    #   if s' <= pos - 1 < s' + n - 1: spans the gap -> new window
    #   if s' >= pos: entirely after the gap -> same as old window s' - 1... 
    # Wait, no. In new_word, position s' maps to:
    #   word[s'] if s' < pos
    #   word[s' + 1] if s' >= pos
    # So a window in new_word starting at s':
    #   new_word[s':s'+n] = word[s':pos] + word[pos+1:pos+1+(n-(pos-s'))] if s' < pos
    #   new_word[s':s'+n] = word[s'+1:s'+1+n] if s' >= pos

    # The windows that changed are those with s' in [max(0, pos-n+1), min(pos-1, new_L-n)]
    # For s' < pos:
    #   new window = word[s':pos] + word[pos+1:s'+n]
    #   length = (pos - s') + (s' + n - (pos + 1)) = n - 1 ... wait
    #   pos - s' elements from word[s':pos], then (s'+n) - (pos+1) = s' + n - pos - 1 from word[pos+1:...]
    #   total = pos - s' + s' + n - pos - 1 = n - 1. That's wrong, should be n.
    # Let me recount: word[s':pos] has length pos - s'. word[pos+1:pos+1+remaining] where remaining = n - (pos - s') = n - pos + s'.
    # So word[pos+1:pos+1+n-pos+s'] has length n - pos + s'.
    # Total: (pos - s') + (n - pos + s') = n. Correct.

    for s_prime in range(max(0, pos - n + 1), min(pos, new_L - n) + 1):
        if s_prime < pos:
            left_len = pos - s_prime
            right_len = n - left_len
            new_w = tuple(word[s_prime:s_prime + left_len] + word[pos + 1:pos + 1 + right_len])
        else:
            # s_prime == pos: window is word[pos+1:pos+1+n] in original
            new_w = tuple(word[pos + 1:pos + 1 + n])
        new_counts[new_w] = new_counts.get(new_w, 0) + 1

    # Verify all required perms are still present
    for perm in required_perms:
        if new_counts.get(perm, 0) == 0:
            return None, None, False

    return new_word, new_counts, True


def _local_delete_search(word, n, deadline):
    """
    Repeatedly try to delete symbols from word using incremental window counts.
    """
    L = len(word)
    required_perms = set(permutations(range(1, n + 1)))
    num_perms = len(required_perms)
    counts = _initial_window_counts(word, n)
    start_time = time.time()

    # Pre-filter: only positions where deletion is "plausible"
    # A quick heuristic: positions where the symbol appears in multiple windows
    # But for now, just try all positions in a smart order.

    # Sort positions by a heuristic: positions in the "middle" of repeated patterns
    # For now, iterate in a pseudo-random order using a simple LCG for determinism
    # Actually, let's do multiple passes: first pass in forward order, then backward,
    # then a random-ish order.

    def _shuffle_positions(L, seed):
        """Deterministic pseudo-shuffle."""
        positions = list(range(L))
        # Simple Fisher-Yates with a deterministic PRNG
        state = seed
        for i in range(L - 1, 0, -1):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            j = state % (i + 1)
            positions[i], positions[j] = positions[j], positions[i]
        return positions

    # Try multiple passes with different orderings
    pass_orders = [
        list(range(L)),           # forward
        list(range(L - 1, -1, -1)),  # backward
    ]
    # Add a few pseudo-random orders
    for seed in [42, 12345, 99999]:
        pass_orders.append(_shuffle_positions(L, seed))

    for pass_idx, order in enumerate(pass_orders):
        if time.time() > deadline:
            break
        # Adjust order length as word shrinks
        current_L = len(word)
        if current_L <= n:
            break
        progress = True
        while progress:
            progress = False
            if time.time() > deadline:
                break
            for pos in order:
                if time.time() > deadline:
                    break
                if pos >= len(word):
                    continue
                # Quick pre-check: is pos a candidate?
                # Position pos is covered by windows [max(0,pos-n+1), min(pos, len(word)-n)]
                # If any of those windows has count 1, skip.
                L_now = len(word)
                s_lo = max(0, pos - n + 1)
                s_hi = min(pos, L_now - n)
                removable = True
                for s in range(s_lo, s_hi + 1):
                    w = tuple(word[s:s + n])
                    if counts.get(w, 0) <= 1:
                        removable = False
                        break
                if not removable:
                    continue

                new_word, new_counts, ok = _try_delete(word, counts, n, required_perms, pos)
                if ok:
                    word = new_word
                    counts = new_counts
                    progress = True
                    # Adjust order for next iteration (positions shifted)
                    # Rebuild order for current length
                    if pass_idx < 2:
                        if pass_idx == 0:
                            order = list(range(len(word)))
                        else:
                            order = list(range(len(word) - 1, -1, -1))
                    else:
                        order = _shuffle_positions(len(word), 42 + pass_idx * 7919)
                    break

    return word


def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Build the standard recursive superpermutation
    word = _build_recursive_superperm(n)

    if n <= 7 and len(word) > n:
        word = _local_delete_search(word, n, deadline)

    # Final validation (should always pass, but just in case)
    required_perms = set(permutations(range(1, n + 1)))
    windows = set()
    for i in range(len(word) - n + 1):
        windows.add(tuple(word[i:i + n]))
    if not required_perms.issubset(windows):
        # Fallback: return the original recursive word
        word = _build_recursive_superperm(n)

    return "".join(map(str, word))

# EVOLVE-BLOCK-END
