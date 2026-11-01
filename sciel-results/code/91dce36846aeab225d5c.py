import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word

    if n < 5:
        return "".join(map(str, word))

    # 2. Optimization via Randomized Local Search with Deletions
    target_count = 1
    for i in range(2, n + 1):
        target_count *= i

    def perm_to_idx(p, n):
        remaining = list(range(1, n + 1))
        code = 0
        for i, val in enumerate(p):
            idx = remaining.index(val)
            code = code * (n - i) + idx
            remaining.pop(idx)
        return code

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
        missing = 0
        for c in counts:
            if c == 0:
                missing += 1
        return counts, missing

    current_word = word
    current_ids = compute_window_ids(current_word)
    current_counts, current_missing = compute_counts(current_ids)
    current_length = len(current_word)
    
    best_word = current_word[:]
    best_length = current_length

    # Local Search Loop
    # We try to delete characters. If stuck, we insert a random character (kick) to escape local minima.
    
    while time.time() < deadline:
        improved = False
        L = len(current_word)
        
        # Try random deletions
        # We limit the number of failed attempts to avoid wasting time in a deep local minimum
        # before kicking.
        attempts = 0
        max_attempts = 10000
        
        while attempts < max_attempts and time.time() < deadline:
            L = len(current_word)
            if L == 0:
                break
            i = random.randint(0, L - 1)
            
            # Check validity of deleting i
            # Affected windows: j in [i-n+1, i]
            start_j = max(0, i - n + 1)
            end_j = min(i, L - n)
            
            valid = True
            affected_ids = current_ids[start_j:end_j+1]
            
            # Count occurrences of each ID in affected windows
            temp_counts = {}
            for aid in affected_ids:
                if aid != -1:
                    temp_counts[aid] = temp_counts.get(aid, 0) + 1
            
            for aid, cnt in temp_counts.items():
                if current_counts[aid] == cnt:
                    valid = False
                    break
            
            if valid:
                # Perform deletion
                current_word.pop(i)
                L -= 1
                # Recompute IDs and Counts
                current_ids = compute_window_ids(current_word)
                current_counts, current_missing = compute_counts(current_ids)
                
                if current_missing == 0 and L < best_length:
                    best_length = L
                    best_word = current_word[:]
                
                improved = True
                attempts = 0 # Reset attempts on success
            else:
                attempts += 1
        
        if not improved:
            # Stuck, apply a kick
            if len(current_word) > 0:
                pos = random.randint(0, len(current_word))
                val = random.randint(1, n)
                current_word.insert(pos, val)
                current_ids = compute_window_ids(current_word)
                current_counts, current_missing = compute_counts(current_ids)
            else:
                break

    return "".join(map(str, best_word))
