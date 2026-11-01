import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 135.0

    # 1. Standard recursive construction
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n < 5:
        return "".join(map(str, word))

    # 2. Fast validity check using array-based counting
    target_count = 1
    for i in range(2, n + 1):
        target_count *= i

    # Map permutations to indices using Lehmer code
    def perm_to_idx(p, n):
        """Convert a permutation tuple to a 0-based index using Lehmer code."""
        remaining = list(range(1, n + 1))
        code = 0
        for i, val in enumerate(p):
            idx = remaining.index(val)
            code = code * (n - i) + idx
            remaining.pop(idx)
        return code

    # Precompute mapping for all permutations
    from itertools import permutations
    perm_map = {}
    idx_to_perm = {}
    for idx, perm in enumerate(permutations(range(1, n + 1))):
        perm_map[perm] = idx
        idx_to_perm[idx] = perm

    def count_windows(w):
        """Count occurrences of each permutation in the word. Returns (counts, total_unique)."""
        counts = [0] * target_count
        total = 0
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                idx = perm_map.get(window)
                if idx is not None:
                    if counts[idx] == 0:
                        total += 1
                    counts[idx] += 1
        return counts, total

    def is_valid_fast(w):
        """Check if word contains all n! permutations."""
        counts, total = count_windows(w)
        return total == target_count

    current_word = word
    current_len = len(current_word)

    # Precompute initial counts
    current_counts, _ = count_windows(current_word)

    # Optimization: Try removing symbols
    # Strategy: In each pass, try to remove each position in random order
    # Use incremental count updates when possible

    improved = True
    pass_count = 0
    while improved and time.time() < deadline:
        improved = False
        pass_count += 1
        indices = list(range(len(current_word)))
        random.shuffle(indices)

        for i in indices:
            if time.time() > deadline:
                break

            # Quick check: if removing position i would break a unique window, skip
            # A window starting at i-1, i, i-2, ..., i-(n-1) might be affected
            # But we need to check if any of those windows has count 1 and is destroyed

            # Try removal
            new_word = current_word[:i] + current_word[i + 1:]

            # Check validity
            if is_valid_fast(new_word):
                current_word = new_word
                current_len -= 1
                improved = True
                # Update counts
                current_counts, _ = count_windows(current_word)
                break  # Restart with new shuffled indices

    # Second phase: Try different strategies
    # Strategy 2: Try removing pairs of adjacent symbols
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            indices = list(range(len(current_word) - 1))
            random.shuffle(indices)

            for i in indices:
                if time.time() > deadline - 5:
                    break

                # Try removing two adjacent symbols
                new_word = current_word[:i] + current_word[i + 2:]

                if is_valid_fast(new_word):
                    current_word = new_word
                    current_len -= 2
                    improved = True
                    current_counts, _ = count_windows(current_word)
                    break

    # Third phase: More aggressive - try removing any single symbol again with different ordering
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            indices = list(range(len(current_word)))
            random.shuffle(indices)

            for i in indices:
                if time.time() > deadline - 2:
                    break

                new_word = current_word[:i] + current_word[i + 1:]

                if is_valid_fast(new_word):
                    current_word = new_word
                    current_len -= 1
                    improved = True
                    current_counts, _ = count_windows(current_word)
                    break

    return "".join(map(str, current_word))
