import itertools
import math

def construct(n):
    # 1. Generate classic superpermutation (Greedy)
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
            # Find max overlap
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Local Search: Delete characters
    # Precompute all valid permutations and map to IDs for speed
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    def get_counts(w):
        counts = [0] * num_perms
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if p in perm_to_id:
                counts[perm_to_id[p]] += 1
        return counts

    counts = get_counts(word)
    
    # Check if initial is valid (it should be)
    if sum(1 for c in counts if c > 0) != num_perms:
        # Fallback to classic if something went wrong
        return "".join(map(str, word))

    improved = True
    max_passes = 50 # Increase passes with faster ops
    
    while improved and max_passes > 0:
        improved = False
        max_passes -= 1
        i = 0
        while i < len(word):
            # Try deleting word[i]
            # Identify windows covering i
            j_min = max(0, i - n + 1)
            j_max = min(i, len(word) - n)
            
            if j_min > j_max:
                i += 1
                continue
            
            # Get perms in these windows
            lost_ids = []
            valid_deletion = True
            for j in range(j_min, j_max + 1):
                p = tuple(word[j:j+n])
                if p in perm_to_id:
                    pid = perm_to_id[p]
                    lost_ids.append(pid)
                    if counts[pid] == 1:
                        valid_deletion = False
                        break
            
            if not valid_deletion:
                i += 1
                continue
            
            # Tentatively remove
            for pid in lost_ids:
                counts[pid] -= 1
            
            # Create new word
            word.pop(i)
            
            # Calculate new windows in the gap
            # New windows are at indices j' in [i-n+1, i-1] (in new_word coordinates)
            # Note: len(word) has decreased by 1
            
            j_min_new = max(0, i - n + 1)
            j_max_new = min(i - 1, len(word) - n)
            
            for j in range(j_min_new, j_max_new + 1):
                p = tuple(word[j:j+n])
                if p in perm_to_id:
                    pid = perm_to_id[p]
                    counts[pid] += 1
            
            # If we deleted, we should check the new element at i (which was i+1)
            # So we don't increment i.
            # But if i was 0, we stay at 0.
            # If i > 0, we stay at i.
            # However, if i is now >= len(word), we break.
            if i >= len(word):
                break
            improved = True
            
    return "".join(map(str, word))
