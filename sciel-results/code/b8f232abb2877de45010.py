import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1213121"
    if n == 4:
        return "121312141213121"
    
    # Build initial superpermutation using standard recursive construction
    def build_initial(n):
        if n == 1:
            return [1]
        if n == 2:
            return [1, 2, 1]
        # S_n is built from S_{n-1}
        prev = build_initial(n - 1)
        # Find all (n-1)-permutations in prev and insert n
        result = []
        used_positions = [False] * len(prev)
        
        # For the standard construction, we interleave
        # Actually, let's use the known formula:
        # S_n = S_{n-1} with n inserted after each complete (n-1)-permutation
        # This is complex, so let's use a simpler approach:
        # Build by adding one symbol at a time with maximum overlap
        
        word = [1]
        for m in range(2, n + 1):
            # Get all permutations of 1..m-1 that appear in word
            perms_in_word = []
            seen = set()
            for i in range(len(word) - (m - 1) + 1):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_in_word.append(window)
            
            # Build new word by inserting m after each permutation
            new_word = []
            for p in perms_in_word:
                block = list(p) + [m]
                # Find maximum overlap with current new_word
                max_ov = min(len(new_word), len(block))
                overlap = 0
                for o in range(max_ov, -1, -1):
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
                new_word.extend(block[overlap:])
            word = new_word
        return word
    
    word = build_initial(n)
    L = len(word)
    
    # Precompute permutation to ID mapping
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    counts = [0] * num_perms
    pid = perm_to_id.get
    
    # Initial count pass
    for j in range(L - n + 1):
        t = tuple(word[j:j+n])
        id_val = pid(t, -1)
        if id_val != -1:
            counts[id_val] += 1
    
    # Check if all permutations are covered
    def is_valid(counts):
        return all(c > 0 for c in counts)
    
    # Greedy deletion with multiple strategies
    best_word = word[:]
    best_length = L
    
    # Strategy 1: Left-to-right greedy deletion
    def greedy_delete_left_to_right(word, counts, n, deadline):
        L = len(word)
        i = 0
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            L = len(word)
            i = 0
            while i < L and time.time() < deadline - 5:
                # Check if we can delete word[i]
                start_min = max(0, i - n + 1)
                start_max = min(i, L - n)
                
                safe = True
                affected = []
                for j in range(start_min, start_max + 1):
                    t = tuple(word[j:j+n])
                    id_val = pid(t, -1)
                    if id_val != -1:
                        affected.append(id_val)
                        if counts[id_val] == 1:
                            safe = False
                            break
                
                if safe:
                    for id_val in affected:
                        counts[id_val] -= 1
                    word.pop(i)
                    L -= 1
                    changed = True
                    
                    # Update counts for new windows
                    new_start_min = max(0, i - n + 1)
                    new_start_max = min(i, L - n)
                    for j in range(new_start_min, new_start_max + 1):
                        t = tuple(word[j:j+n])
                        id_val = pid(t, -1)
                        if id_val != -1:
                            counts[id_val] += 1
                else:
                    i += 1
        return word
    
    # Strategy 2: Right-to-left greedy deletion
    def greedy_delete_right_to_left(word, counts, n, deadline):
        L = len(word)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            L = len(word)
            i = L - 1
            while i >= 0 and time.time() < deadline - 5:
                # Check if we can delete word[i]
                start_min = max(0, i - n + 1)
                start_max = min(i, L - n)
                
                safe = True
                affected = []
                for j in range(start_min, start_max + 1):
                    t = tuple(word[j:j+n])
                    id_val = pid(t, -1)
                    if id_val != -1:
                        affected.append(id_val)
                        if counts[id_val] == 1:
                            safe = False
                            break
                
                if safe:
                    for id_val in affected:
                        counts[id_val] -= 1
                    word.pop(i)
                    L -= 1
                    changed = True
                    
                    # Update counts for new windows
                    new_start_min = max(0, i - n + 1)
                    new_start_max = min(i, L - n)
                    for j in range(new_start_min, new_start_max + 1):
                        t = tuple(word[j:j+n])
                        id_val = pid(t, -1)
                        if id_val != -1:
                            counts[id_val] += 1
                else:
                    i -= 1
        return word
    
    # Strategy 3: Random position deletion
    def greedy_delete_random(word, counts, n, deadline):
        L = len(word)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            L = len(word)
            # Try random positions
            for _ in range(min(1000, L)):
                if time.time() > deadline - 5:
                    break
                i = random.randint(0, L - 1)
                start_min = max(0, i - n + 1)
                start_max = min(i, L - n)
                
                safe = True
                affected = []
                for j in range(start_min, start_max + 1):
                    t = tuple(word[j:j+n])
                    id_val = pid(t, -1)
                    if id_val != -1:
                        affected.append(id_val)
                        if counts[id_val] == 1:
                            safe = False
                            break
                
                if safe:
                    for id_val in affected:
                        counts[id_val] -= 1
                    word.pop(i)
                    L -= 1
                    changed = True
                    
                    # Update counts for new windows
                    new_start_min = max(0, i - n + 1)
                    new_start_max = min(i, L - n)
                    for j in range(new_start_min, new_start_max + 1):
                        t = tuple(word[j:j+n])
                        id_val = pid(t, -1)
                        if id_val != -1:
                            counts[id_val] += 1
        return word
    
    # Apply strategies in sequence
    # First, left-to-right
    word = greedy_delete_left_to_right(word, counts, n, deadline)
    L = len(word)
    if L < best_length and is_valid(counts):
        best_word = word[:]
        best_length = L
    
    # Recount
    counts = [0] * num_perms
    for j in range(L - n + 1):
        t = tuple(word[j:j+n])
        id_val = pid(t, -1)
        if id_val != -1:
            counts[id_val] += 1
    
    # Then, right-to-left
    word = greedy_delete_right_to_left(word, counts, n, deadline)
    L = len(word)
    if L < best_length and is_valid(counts):
        best_word = word[:]
        best_length = L
    
    # Recount
    counts = [0] * num_perms
    for j in range(L - n + 1):
        t = tuple(word[j:j+n])
        id_val = pid(t, -1)
        if id_val != -1:
            counts[id_val] += 1
    
    # Then, random
    word = greedy_delete_random(word, counts, n, deadline)
    L = len(word)
    if L < best_length and is_valid(counts):
        best_word = word[:]
        best_length = L
    
    # Try more passes if time allows
    while time.time() < deadline - 10:
        # Alternate between left-to-right and right-to-left
        counts = [0] * num_perms
        L = len(best_word)
        for j in range(L - n + 1):
            t = tuple(best_word[j:j+n])
            id_val = pid(t, -1)
            if id_val != -1:
                counts[id_val] += 1
        
        # Try left-to-right
        word_copy = best_word[:]
        word_copy = greedy_delete_left_to_right(word_copy, counts[:], n, deadline)
        L = len(word_copy)
        if L < best_length:
            # Verify validity
            counts_check = [0] * num_perms
            for j in range(L - n + 1):
                t = tuple(word_copy[j:j+n])
                id_val = pid(t, -1)
                if id_val != -1:
                    counts_check[id_val] += 1
            if all(c > 0 for c in counts_check):
                best_word = word_copy
                best_length = L
        
        # Try right-to-left
        counts = [0] * num_perms
        L = len(best_word)
        for j in range(L - n + 1):
            t = tuple(best_word[j:j+n])
            id_val = pid(t, -1)
            if id_val != -1:
                counts[id_val] += 1
        
        word_copy = best_word[:]
        word_copy = greedy_delete_right_to_left(word_copy, counts[:], n, deadline)
        L = len(word_copy)
        if L < best_length:
            # Verify validity
            counts_check = [0] * num_perms
            for j in range(L - n + 1):
                t = tuple(word_copy[j:j+n])
                id_val = pid(t, -1)
                if id_val != -1:
                    counts_check[id_val] += 1
            if all(c > 0 for c in counts_check):
                best_word = word_copy
                best_length = L
        
        if time.time() > deadline - 15:
            break
    
    return "".join(map(str, best_word))
