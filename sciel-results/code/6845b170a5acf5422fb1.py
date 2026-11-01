import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0

    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341231423124312"
    
    # Standard recursive construction
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
    
    if n <= 4:
        return "".join(map(str, word))
    
    target_count = 1
    for i in range(2, n + 1):
        target_count *= i
        
    current_word = list(word)
    
    def compute_counts(w):
        counts = {}
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                counts[window] = counts.get(window, 0) + 1
        return counts

    perm_count = compute_counts(current_word)
    
    def is_removal_valid(i, w, counts, n):
        # Check if removing index i is valid
        # Windows affected: starting at j in [i-n+1, i]
        old_start = max(0, i - n + 1)
        old_end = min(i, len(w) - n)
        
        for j in range(old_start, old_end + 1):
            window = tuple(w[j:j+n])
            if len(set(window)) == n:
                if counts[window] == 1:
                    return False
        return True

    def get_safety_score(i, w, counts, n):
        # Min count of windows containing i
        old_start = max(0, i - n + 1)
        old_end = min(i, len(w) - n)
        min_count = float('inf')
        for j in range(old_start, old_end + 1):
            window = tuple(w[j:j+n])
            if len(set(window)) == n:
                c = counts[window]
                if c < min_count:
                    min_count = c
        return min_count

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Find all valid indices and their safety scores
        valid_indices = []
        for i in range(len(current_word)):
            if time.time() > deadline:
                break
            if is_removal_valid(i, current_word, perm_count, n):
                score = get_safety_score(i, current_word, perm_count, n)
                valid_indices.append((score, i))
                
        if not valid_indices:
            break
            
        # Sort by safety score (descending) and pick the best one
        # Or pick randomly from the top K to add diversity?
        # Let's pick the safest one (highest min count).
        valid_indices.sort(key=lambda x: x[0], reverse=True)
        
        # To avoid getting stuck in a specific pattern, we might pick randomly from the top 10
        # But deterministic "safest first" is often good for superpermutations.
        # Let's try picking the safest.
        best_score, best_idx = valid_indices[0]
        
        # Perform removal
        i = best_idx
        # Update counts
        old_start = max(0, i - n + 1)
        old_end = min(i, len(current_word) - n)
        
        affected_old = []
        for j in range(old_start, old_end + 1):
            window = tuple(current_word[j:j+n])
            if len(set(window)) == n:
                affected_old.append((j, window))
        
        for j, window in affected_old:
            perm_count[window] -= 1
            if perm_count[window] == 0:
                del perm_count[window]
        
        new_word = current_word[:i] + current_word[i+1:]
        
        new_start = max(0, i - n + 1)
        new_end = min(i - 1, len(new_word) - n)
        
        for k in range(new_start, new_end + 1):
            new_window = tuple(new_word[k:k+n])
            if len(set(new_window)) == n:
                perm_count[new_window] = perm_count.get(new_window, 0) + 1
        
        current_word = new_word
        improved = True

    # Final verification
    if time.time() < deadline - 2:
        seen = set()
        for i in range(len(current_word) - n + 1):
            window = tuple(current_word[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        if len(seen) != target_count:
            current_word = list(word)
    
    return "".join(map(str, current_word))
