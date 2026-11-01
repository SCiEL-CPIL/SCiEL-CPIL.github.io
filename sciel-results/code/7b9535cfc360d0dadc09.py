import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

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

    # 2. Fast optimization
    target_count = 1
    for i in range(2, n + 1):
        target_count *= i

    # Map permutations to indices using Lehmer code
    def perm_to_idx(p, n):
        remaining = list(range(1, n + 1))
        code = 0
        for i, val in enumerate(p):
            idx = remaining.index(val)
            code = code * (n - i) + idx
            remaining.pop(idx)
        return code

    # Precompute mapping for all permutations
    perm_map = {}
    for perm in permutations(range(1, n + 1)):
        perm_map[perm] = perm_to_idx(perm, n)

    def compute_window_ids(w):
        ids = []
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            # Only valid if it's a permutation (no duplicates)
            # In a superpermutation, all windows of length n are permutations?
            # No, the construction guarantees that, but during removal, we must ensure validity.
            # If a window has duplicates, it's not a permutation.
            # But we only care about windows that ARE permutations.
            # If a window is not a permutation, its ID is -1 (or similar) and it doesn't contribute to counts.
            if len(set(window)) == n:
                ids.append(perm_map[window])
            else:
                ids.append(-1)
        return ids

    def compute_counts(ids):
        counts = [0] * target_count
        for idx in ids:
            if idx != -1:
                counts[idx] += 1
        return counts

    current_word = word
    current_ids = compute_window_ids(current_word)
    current_counts = compute_counts(current_ids)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        L = len(current_word)
        
        # Try removing each index
        # We can iterate in random order or sequential. Sequential is deterministic.
        # Let's try sequential first.
        for i in range(L):
            if time.time() > deadline:
                return "".join(map(str, current_word))
            
            # Determine affected windows
            # Window j is affected if i is in [j, j+n-1] => j in [i-n+1, i]
            start_j = max(0, i - n + 1)
            end_j = min(i, L - n)
            
            valid = True
            # Store counts to restore if invalid
            # Actually, we can just check if any count would become 0
            # We need to know the IDs of the affected windows
            affected_ids = current_ids[start_j:end_j+1]
            
            # Check if removing these windows leaves count > 0 for all
            # We don't need to decrement all, just check min count in affected
            # But we need to be careful: if a window ID appears multiple times in affected,
            # we need to decrement that many times.
            
            # Since n is small, we can just simulate
            temp_counts = current_counts[:] # This is O(N!) which is 5040. Cheap.
            
            for aid in affected_ids:
                if aid != -1:
                    temp_counts[aid] -= 1
                    if temp_counts[aid] == 0:
                        valid = False
                        break
            
            if valid:
                # Perform removal
                current_word = current_word[:i] + current_word[i+1:]
                # Recompute IDs and Counts
                current_ids = compute_window_ids(current_word)
                current_counts = compute_counts(current_ids)
                improved = True
                break # Restart scan

    return "".join(map(str, current_word))
