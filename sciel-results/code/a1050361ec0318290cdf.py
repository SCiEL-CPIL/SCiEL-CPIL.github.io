import time
import random
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        word = word + [m] + word
        
    initial_word = list(word)
    
    # 2. Precompute permutation mapping
    perms = list(itertools.permutations(range(1, n+1)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    total_perms = len(perms)
    
    # Initialize counts
    perm_counts = [0] * total_perms
    
    # Fill counts for initial word
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if p in perm_to_idx:
            perm_counts[perm_to_idx[p]] += 1
            
    # Check validity
    if any(c == 0 for c in perm_counts):
        return "".join(map(str, initial_word))

    def get_perm_idx(window_tuple):
        return perm_to_idx.get(window_tuple, -1)

    def try_delete(i):
        L = len(word)
        if L <= n:
            return False
            
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        
        if start_j > end_j:
            return False
        
        affected_indices = []
        for j in range(start_j, end_j + 1):
            idx = get_perm_idx(tuple(word[j:j+n]))
            if idx != -1:
                affected_indices.append(idx)
                perm_counts[idx] -= 1
                if perm_counts[idx] == 0:
                    for a_idx in affected_indices:
                        perm_counts[a_idx] += 1
                    return False
        
        word.pop(i)
        return True

    best_len = len(word)
    best_word = list(word)
    
    def update_best():
        nonlocal best_len, best_word
        if len(word) < best_len:
            best_len = len(word)
            best_word = list(word)

    # Main Loop
    while time.time() < deadline:
        L = len(word)
        if L <= n:
            break
            
        progress = False
        
        # Try random deletions
        for _ in range(100):
            if time.time() > deadline:
                break
            i = random.randint(0, L - 1)
            if try_delete(i):
                progress = True
                update_best()
                break
        
        if progress:
            continue
            
        # Try random swaps
        swap_done = False
        for _ in range(100):
            if time.time() > deadline:
                break
            i = random.randint(0, L - 1)
            j = random.randint(0, L - 1)
            if i == j:
                continue
                
            start_j = max(0, min(i, j) - n + 1)
            end_j = min(max(i, j), L - n)
            
            old_counts = {}
            for j_idx in range(start_j, end_j + 1):
                p = tuple(word[j_idx:j_idx+n])
                idx = get_perm_idx(p)
                if idx != -1:
                    old_counts[idx] = perm_counts[idx]
                    perm_counts[idx] -= 1
                    
            word[i], word[j] = word[j], word[i]
            
            valid = True
            for j_idx in range(start_j, end_j + 1):
                p = tuple(word[j_idx:j_idx+n])
                idx = get_perm_idx(p)
                if idx != -1:
                    perm_counts[idx] += 1
                    if perm_counts[idx] == 0:
                        valid = False
                        break
                        
            if valid:
                swap_done = True
            else:
                word[i], word[j] = word[j], word[i]
                for idx, count in old_counts.items():
                    perm_counts[idx] = count
                    
            if swap_done:
                break
                
    if len(word) < best_len:
        best_word = list(word)
        
    # Final validation
    seen = set()
    for i in range(len(best_word) - n + 1):
        seen.add(tuple(best_word[i:i+n]))
    if len(seen) < total_perms:
        return "".join(map(str, initial_word))
        
    return "".join(map(str, best_word))
