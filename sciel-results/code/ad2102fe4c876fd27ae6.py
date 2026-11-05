import time
import itertools
import random

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # 1. Generate classic recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            target_overlap = m - 2
            overlap = 0
            if len(new_word) >= target_overlap and len(block) >= target_overlap:
                if new_word[-target_overlap:] == block[:target_overlap]:
                    overlap = target_overlap
                else:
                    max_overlap = min(len(new_word), len(block))
                    for o in range(max_overlap, -1, -1):
                        if o == 0:
                            break
                        if new_word[len(new_word) - o:] == block[:o]:
                            overlap = o
                            break
            else:
                max_overlap = min(len(new_word), len(block))
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        break
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
            
            new_word.extend(block[overlap:])
        
        word = new_word
        
    current_word = word[:]
    
    # Pre-compute all permutations
    all_perms = list(itertools.permutations(range(1, n+1)))
    perm_set = set(all_perms)
    counts = {p: 0 for p in perm_set}
    
    # Initial count
    for i in range(len(current_word) - n + 1):
        p = tuple(current_word[i:i+n])
        if p in counts:
            counts[p] += 1

    best_word = current_word[:]
    best_len = len(current_word)
    
    start_time = time.time()
    deadline = 110.0
    
    # Randomized Greedy Deletion
    while time.time() < deadline:
        # Shuffle indices to break symmetry/local minima
        indices = list(range(len(current_word)))
        random.shuffle(indices)
        
        # We need to be careful: deleting an index shifts subsequent indices.
        # So we should iterate and check if the index is still valid.
        # Or, we can iterate in the shuffled order, but if we delete, 
        # the indices of later elements in the *original* list change.
        # It's safer to iterate from end to start in the *current* list,
        # but with a random priority?
        
        # Alternative:
        # Create a list of (random_priority, index)
        # Sort by priority
        # Iterate through sorted list.
        # If index is still < len(current_word), try to delete.
        
        priorities = [(random.random(), i) for i in range(len(current_word))]
        priorities.sort(key=lambda x: x[0])
        
        for _, i in priorities:
            if i >= len(current_word):
                continue
            if time.time() > deadline:
                break
            
            # Check if we can delete i
            # Affected windows: j in [i-n+1, i]
            start_j = max(0, i - n + 1)
            end_j = min(i, len(current_word) - n)
            
            can_delete = True
            affected = []
            for j in range(start_j, end_j + 1):
                p = tuple(current_word[j:j+n])
                if counts.get(p, 0) <= 1:
                    can_delete = False
                    break
                affected.append(p)
            
            if can_delete:
                # Perform deletion
                for p in affected:
                    counts[p] -= 1
                current_word.pop(i)
                
                if len(current_word) < best_len:
                    best_len = len(current_word)
                    best_word = current_word[:]
                    
        # If we didn't improve, we might be stuck.
        # The shuffle helps.
        
    return "".join(map(str, best_word))
