import time
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Classic recursive construction
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
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n <= 4:
        return "".join(map(str, word))

    # Local deletion optimization
    current = list(word)
    L = len(current)
    
    # Precompute permutation counts
    perm_count = defaultdict(int)
    for i in range(L - n + 1):
        p = tuple(current[i:i + n])
        perm_count[p] += 1
    
    target_count = 1
    for i in range(2, n + 1):
        target_count *= i
    
    if len(perm_count) < target_count:
        return "".join(map(str, current))
    
    # Deletion pass: try to remove each symbol
    changed = True
    while changed and time.time() < deadline:
        changed = False
        L = len(current)
        for i in range(L - 1, -1, -1):
            if time.time() > deadline:
                return "".join(map(str, current))
            
            # Find old windows containing position i
            start_w = max(0, i - n + 1)
            end_w = min(i, L - n)
            
            old_windows = []
            for w in range(start_w, end_w + 1):
                p = tuple(current[w:w + n])
                old_windows.append(p)
            
            # Find new windows after deletion
            new_windows = []
            for j in range(max(0, i - n + 1), i):
                if j + n > L - 1:
                    continue
                prefix = current[j:i]
                suffix_len = n - (i - j)
                suffix = current[i + 1:i + 1 + suffix_len]
                if len(suffix) < suffix_len:
                    continue
                p = tuple(prefix + suffix)
                new_windows.append(p)
            
            new_set = set(new_windows)
            
            # Check if any unique old window is lost
            can_delete = True
            for p in old_windows:
                if perm_count[p] == 1 and p not in new_set:
                    can_delete = False
                    break
            
            if can_delete:
                # Perform deletion
                for p in old_windows:
                    perm_count[p] -= 1
                for p in new_windows:
                    perm_count[p] += 1
                del current[i]
                changed = True
                break
    
    return "".join(map(str, current))
