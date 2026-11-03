import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Generate standard superpermutation
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n < 3:
        return "".join(map(str, word))

    # Precompute permutation to index mapping
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    def build_counts(w):
        c = [0] * num_perms
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if p in perm_to_idx:
                c[perm_to_idx[p]] += 1
        return c

    counts = build_counts(word)
    
    def try_delete(k, w, c):
        """
        Efficiently test deletion of index k using delta tracking.
        Returns (is_valid, new_counts) or (False, None).
        """
        L = len(w)
        j_min = max(0, k - n + 1)
        j_max = min(L - n, k)
        
        # Track all indices that will be modified
        delta = {}
        
        # Decrement old windows containing k
        for j in range(j_min, j_max + 1):
            p = tuple(w[j:j+n])
            idx = perm_to_idx.get(p)
            if idx is not None:
                delta[idx] = delta.get(idx, 0) - 1
                if c[idx] + delta[idx] <= 0:
                    return False, None
        
        # Increment new windows formed at junction
        j_prime_min = max(0, k - n + 1)
        j_prime_max = min(L - 1 - n, k - 1)
        
        for j_prime in range(j_prime_min, j_prime_max + 1):
            left_part = w[j_prime:k]
            right_part = w[k+1:j_prime+n+1]
            new_window_list = left_part + right_part
            p_new = tuple(new_window_list)
            idx = perm_to_idx.get(p_new)
            if idx is not None:
                delta[idx] = delta.get(idx, 0) + 1
        
        # Check all modified indices
        for idx, d in delta.items():
            if c[idx] + d <= 0:
                return False, None
        
        # Build new counts
        new_c = c[:]
        for idx, d in delta.items():
            new_c[idx] += d
        
        return True, new_c

    def try_swap(i, w, c):
        """
        Efficiently test swap of positions i and i+1.
        Returns (is_valid, new_counts) or (False, None).
        """
        L = len(w)
        # Windows affected: those containing i or i+1
        # Window start j where j <= i <= j+n-1 OR j <= i+1 <= j+n-1
        j_min = max(0, i - n + 1)
        j_max = min(L - n, i + 1)
        
        delta = {}
        
        # Decrement old windows
        for j in range(j_min, j_max + 1):
            p = tuple(w[j:j+n])
            idx = perm_to_idx.get(p)
            if idx is not None:
                delta[idx] = delta.get(idx, 0) - 1
        
        # Increment new windows after swap
        # Create swapped view
        for j in range(j_min, j_max + 1):
            # Build window with swap applied
            window = list(w[j:j+n])
            # Adjust indices: if j <= i < j+n, position i in window is at i-j
            # if j <= i+1 < j+n, position i+1 in window is at i+1-j
            if j <= i < j + n:
                pos1 = i - j
            else:
                pos1 = -1
            if j <= i + 1 < j + n:
                pos2 = i + 1 - j
            else:
                pos2 = -1
            
            if pos1 >= 0 and pos2 >= 0:
                window[pos1], window[pos2] = window[pos2], window[pos1]
            
            p_new = tuple(window)
            idx = perm_to_idx.get(p_new)
            if idx is not None:
                delta[idx] = delta.get(idx, 0) + 1
        
        # Check all modified indices
        for idx, d in delta.items():
            if c[idx] + d <= 0:
                return False, None
        
        # Build new counts
        new_c = c[:]
        for idx, d in delta.items():
            new_c[idx] += d
        
        return True, new_c

    # Main optimization loop with multiple passes
    while time.time() < deadline - 10:
        L = len(word)
        if L < n:
            break
        
        improved = False
        
        # Pass 1: Try deletions
        for k in range(L):
            if time.time() > deadline - 10:
                break
            valid, new_counts = try_delete(k, word, counts)
            if valid:
                word = word[:k] + word[k+1:]
                counts = new_counts
                improved = True
                break
        
        # Pass 2: Try swaps if no deletion worked
        if not improved:
            for i in range(L - 1):
                if time.time() > deadline - 10:
                    break
                valid, new_counts = try_swap(i, word, counts)
                if valid:
                    word[i], word[i+1] = word[i+1], word[i]
                    counts = new_counts
                    improved = True
                    break
        
        if not improved:
            break

    return "".join(map(str, word))
