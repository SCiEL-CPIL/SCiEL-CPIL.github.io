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

    current_word = list(word)
    current_ids = compute_window_ids(current_word)
    current_counts = compute_counts(current_ids)

    # Batch removal: remove all removable positions in one pass, then recompute
    improved = True
    while improved and time.time() < deadline:
        improved = False
        L = len(current_word)
        
        # Mark positions that are in a window with count==1
        not_removable = [False] * L
        for j in range(L - n + 1):
            wid = current_ids[j]
            if wid != -1 and current_counts[wid] == 1:
                for i in range(j, j + n):
                    not_removable[i] = True
        
        # Collect removable positions
        positions_to_remove = [i for i in range(L) if not not_removable[i]]
        
        if positions_to_remove:
            # Remove in reverse order
            for i in reversed(positions_to_remove):
                del current_word[i]
            improved = True
            current_ids = compute_window_ids(current_word)
            current_counts = compute_counts(current_ids)
    
    return "".join(map(str, current_word))
