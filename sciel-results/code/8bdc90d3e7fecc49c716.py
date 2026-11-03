import time
from collections import defaultdict

def construct(n):
    """
    Constructs a superpermutation using the classic recursive method,
    then applies a deletion heuristic to reduce the length.
    """
    start_time = time.time()
    
    # 1. Generate initial superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        w_size = m - 1
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            if len(set(window)) == w_size:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
        
        word = new_word
        
        if time.time() - start_time > 110:
            break

    if n == 1:
        return "1"

    # 2. Deletion Heuristic
    # Initialize permutation counts
    perm_counts = defaultdict(int)
    for i in range(len(word) - n + 1):
        perm_counts[tuple(word[i:i+n])] += 1
        
    # Improved Deletion Strategy:
    # 1. Find all valid deletion candidates.
    # 2. Select the best candidate based on a heuristic (e.g., clustering).
    # 3. Delete and update.
    # 4. Repeat until no valid deletions or time limit.
    
    deadline = 118.0
    while time.time() < deadline:
        valid_indices = []
        
        # Find all valid deletion indices
        for i in range(len(word)):
            if time.time() > deadline:
                break
                
            # Try to delete word[i]
            start_j = max(0, i - n + 1)
            end_j = min(i, len(word) - n)
            
            old_windows = []
            valid = True
            
            for j in range(start_j, end_j + 1):
                w = tuple(word[j:j+n])
                old_windows.append(w)
                perm_counts[w] -= 1
                if perm_counts[w] == 0:
                    valid = False
                    break
            
            if valid:
                # Revert counts for the check
                for w in old_windows:
                    perm_counts[w] += 1
                valid_indices.append(i)
            else:
                # Revert counts
                for w in old_windows:
                    perm_counts[w] += 1
            
        if not valid_indices:
            break
            
        # Heuristic: Choose the index that is closest to other valid indices (clustering)
        # Or simply the first one?
        # Let's try clustering: Score = number of valid indices within distance n
        best_idx = valid_indices[0]
        best_score = -1
        
        for idx in valid_indices:
            score = 0
            for other_idx in valid_indices:
                if other_idx != idx and abs(idx - other_idx) < n:
                    score += 1
            if score > best_score:
                best_score = score
                best_idx = idx
                
        # Perform deletion at best_idx
        i = best_idx
        word.pop(i)
        
        # Update counts for new windows
        new_start_j = max(0, i - n + 1)
        new_end_j = min(i - 1, len(word) - n)
        
        for j in range(new_start_j, new_end_j + 1):
            w = tuple(word[j:j+n])
            perm_counts[w] += 1
            
    return "".join(map(str, word))
