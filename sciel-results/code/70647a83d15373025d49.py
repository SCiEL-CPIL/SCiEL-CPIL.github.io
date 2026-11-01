import time
import numpy as np
from collections import defaultdict
import math

def construct(n):
    start_time = time.time()
    deadline = 140.0  # Safety margin

    # 1. Generate initial superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of length m-1 from current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        new = []
        for p in perms:
            # Construct block: p + [m] + p
            block = list(p) + [m] + list(p)
            # Find max overlap with current new
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n <= 4:
        return "".join(map(str, word))

    # 2. Local Search Deletion with corrected counting
    current = list(word)
    L = len(current)
    target_perms = math.factorial(n)

    # Precompute base for code computation
    # Using integer codes for speed
    base = [n ** (n - 1 - i) for i in range(n)]
    
    def calc_code(window):
        code = 0
        for i, val in enumerate(window):
            code += (val - 1) * base[i]
        return code

    # Initialize perm_count
    perm_count = defaultdict(int)
    for i in range(L - n + 1):
        code = calc_code(current[i:i+n])
        perm_count[code] += 1

    if len(perm_count) < target_perms:
        return "".join(map(str, current))

    # Deletion loop
    # We will try to delete positions.
    # To be efficient, we can iterate through positions multiple times.
    # We use a "scan" approach.
    
    max_passes = 50
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        deleted = False
        # Try both directions in alternating passes? Or just one?
        # Forward and backward can find different deletions.
        for direction in ['forward', 'backward']:
            if direction == 'forward':
                indices = range(L)
            else:
                indices = range(L - 1, -1, -1)

            for i in indices:
                if time.time() > deadline:
                    break

                # Identify old windows containing position i
                start_w = max(0, i - n + 1)
                end_w = min(i, L - n)
                
                if start_w > end_w:
                    continue # Should not happen if L >= n

                old_codes = []
                for w in range(start_w, end_w + 1):
                    code = calc_code(current[w:w + n])
                    old_codes.append(code)
                
                # Identify new windows formed
                # New windows start at j in [max(0, i-n+1), i-1] in the NEW string
                # The new string is current[:i] + current[i+1:]
                # A window in the new string starting at j (where j < i) corresponds to:
                # current[j:i] + current[i+1 : i+1 + (n - (i-j))]
                new_codes = []
                for j in range(max(0, i - n + 1), i):
                    prefix_len = i - j
                    suffix_len = n - prefix_len
                    # Check bounds for suffix
                    if i + 1 + suffix_len > L:
                        continue
                    # Construct window
                    # Optimization: we can just calculate the code directly
                    # But constructing the list is easier for correctness
                    window = current[j:i] + current[i+1:i+1+suffix_len]
                    if len(window) == n:
                        code = calc_code(window)
                        new_codes.append(code)

                # Calculate delta counts
                # We only care about perms in old_codes or new_codes
                affected_perms = set(old_codes) | set(new_codes)
                
                # Check validity
                valid = True
                for p in affected_perms:
                    count_in_old = old_codes.count(p)
                    count_in_new = new_codes.count(p)
                    new_total_count = perm_count[p] - count_in_old + count_in_new
                    if new_total_count < 1:
                        valid = False
                        break
                
                if valid:
                    # Perform deletion
                    # Update perm_count
                    for p in affected_perms:
                        count_in_old = old_codes.count(p)
                        count_in_new = new_codes.count(p)
                        perm_count[p] += -count_in_old + count_in_new
                        if perm_count[p] == 0:
                            del perm_count[p]
                    
                    # Update current
                    current = current[:i] + current[i+1:]
                    L -= 1
                    deleted = True
                    break # Restart scan after deletion

            if deleted:
                break

        if not deleted:
            break # No more deletions possible in this pass

    return "".join(map(str, current))
