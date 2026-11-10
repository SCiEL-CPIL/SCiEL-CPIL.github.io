import time
import math
import random
from collections import Counter

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Step 1: Generate classic superpermutation
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
    
    # If n is small, just return
    if n <= 5:
        return "".join(map(str, word))
        
    # Step 2: Simulated Annealing on Deletions/Insertions
    target_perms = math.factorial(n)
    
    # Initialize counts and seen_set
    counts = Counter()
    seen_set = set()
    L = len(word)
    for i in range(L - n + 1):
        perm = tuple(word[i:i+n])
        counts[perm] += 1
        seen_set.add(perm)
        
    current_missing = target_perms - len(seen_set)
    current_len = L
    M = 10000  # Penalty for missing permutations
    current_score = current_len + M * current_missing
    
    best_word = word[:]
    best_len = current_len
    best_score = current_score
    
    T = 50.0
    T_min = 0.1
    alpha = 0.999
    
    while time.time() < deadline:
        # Generate move
        if random.random() < 0.7:
            # Delete
            if len(word) <= n:
                # Can't delete below n
                pass
            else:
                i = random.randint(0, len(word) - 1)
                # Identify removed windows
                remove_perms = []
                for k in range(max(0, i - n + 1), min(i, len(word) - n) + 1):
                    p = tuple(word[k:k+n])
                    remove_perms.append(p)
                
                # Identify added windows (crossing the gap)
                add_perms = []
                # k in new word indices [0, len(word)-2]
                # Gap is at i (in old indices).
                # New word W' length L-1.
                # Window start k in W' crosses gap if k < i and k+n > i (in W' indices? No, careful with mapping)
                # Let's just construct the crossing windows in W' directly.
                # W' = word[:i] + word[i+1:]
                # Crossing windows are those starting at k in W' such that the window includes the junction.
                # Junction is between W'[i-1] and W'[i] (if i>0).
                # Actually, simpler: 
                # The windows in W' that are NEW are those that contain the junction.
                # In W', the junction is at index i (if we consider the right part starts at i).
                # Wait, W'[0..i-1] = W[0..i-1]. W'[i..] = W[i+1..].
                # A window in W' starting at k includes the junction if k <= i-1 and k+n-1 >= i.
                # So k <= i-1 and k >= i-n+1.
                # Also k must be valid start in W': 0 <= k <= (L-1)-n.
                
                for k in range(max(0, i - n + 1), min(i, len(word) - n)):
                    # k is start index in W'
                    # Window is W'[k:k+n]
                    # W'[k:k+n] = word[k:i] + word[i+1:i+1+(k+n-i)]
                    # Length of first part: i-k
                    # Length of second part: n - (i-k) = n - i + k
                    # End index in word: i+1 + (n - i + k) - 1 = n + k
                    w1 = word[k:i]
                    w2 = word[i+1:i+1 + (n - (i-k))]
                    if len(w1) + len(w2) == n:
                        p = tuple(w1 + w2)
                        add_perms.append(p)
                
                # Apply changes
                for p in remove_perms:
                    counts[p] -= 1
                    if counts[p] == 0:
                        seen_set.discard(p)
                
                for p in add_perms:
                    counts[p] += 1
                    seen_set.add(p)
                
                new_missing = target_perms - len(seen_set)
                new_len = len(word) - 1
                new_score = new_len + M * new_missing
                
                delta = new_score - current_score
                accept = False
                if new_score < current_score:
                    accept = True
                else:
                    if delta < 1000: # Avoid overflow
                        accept = random.random() < math.exp(-delta / T)
                    else:
                        accept = False
                
                if accept:
                    word = word[:i] + word[i+1:]
                    current_missing = new_missing
                    current_len = new_len
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_len = current_len
                        best_word = word[:]
                        if current_missing == 0:
                            best_len = current_len # Update best valid length
                else:
                    # Revert
                    for p in remove_perms:
                        counts[p] += 1
                        seen_set.add(p)
                    
                    for p in add_perms:
                        counts[p] -= 1
                        if counts[p] == 0:
                            seen_set.discard(p)
        else:
            # Insert
            s = random.randint(1, n)
            j = random.randint(0, len(word))
            
            # Identify added windows (containing the inserted element at j)
            add_perms = []
            # In new word W', inserted at j.
            # W'[0..j-1] = W[0..j-1]. W'[j] = s. W'[j+1..] = W[j..].
            # Windows containing j in W':
            # Start k such that k <= j <= k+n-1.
            # k in [j-n+1, j].
            # Also 0 <= k <= len(W')-n = len(W)-1.
            
            for k in range(max(0, j - n + 1), min(j, len(word)) + 1):
                # Window W'[k:k+n]
                # Part 1: W'[k:j] -> W[k:j]
                # Part 2: W'[j] -> s
                # Part 3: W'[j+1:k+n] -> W[j:k+n-1]
                
                w1 = word[k:j]
                w2 = [s]
                # Length of w3 needed: n - len(w1) - 1
                # w3 comes from W[j : j + (n - len(w1) - 1)]
                len_w3 = n - len(w1) - 1
                w3 = word[j:j+len_w3]
                
                if len(w1) + 1 + len(w3) == n:
                    p = tuple(w1 + w2 + w3)
                    add_perms.append(p)
            
            # Apply changes (only add)
            for p in add_perms:
                counts[p] += 1
                seen_set.add(p)
            
            new_missing = target_perms - len(seen_set)
            new_len = len(word) + 1
            new_score = new_len + M * new_missing
            
            delta = new_score - current_score
            accept = False
            if new_score < current_score:
                accept = True
            else:
                if delta < 1000:
                    accept = random.random() < math.exp(-delta / T)
                else:
                    accept = False
            
            if accept:
                word = word[:j] + [s] + word[j:]
                current_missing = new_missing
                current_len = new_len
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_len = current_len
                    best_word = word[:]
                    if current_missing == 0:
                        best_len = current_len
            else:
                # Revert
                for p in add_perms:
                    counts[p] -= 1
                    if counts[p] == 0:
                        seen_set.discard(p)
        
        T *= alpha
        if T < T_min:
            T = T_min

    # Return the best valid word found
    # If best_word is not valid (missing > 0), we should return the initial valid word?
    # The problem says "Valid when every permutation...".
    # So we must return a valid word.
    # best_word is only updated if current_score < best_score.
    # Initially best_word is the standard valid word.
    # So best_word is always valid?
    # Yes, because we start with a valid word, and we only update best_word if the new state is better.
    # But wait, if we accept an invalid move, current_missing > 0.
    # Then current_score = len + M*missing.
    # This score is likely higher than the initial valid score (len + 0).
    # So best_word will remain the initial valid word unless we find a *shorter* valid word.
    # So best_word is guaranteed to be valid.
    
    return "".join(map(str, best_word))
