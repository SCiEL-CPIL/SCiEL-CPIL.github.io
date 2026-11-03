import time
import sys
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish well before.
    deadline = start_time + 140.0
    
    # 1. Recursive Construction
    word = [1]
    
    for m in range(2, n + 1):
        # Extract unique permutations of length m-1 from word
        # in order of appearance
        perms = []
        seen = set()
        L = len(word)
        for i in range(L - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Construct new word
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, -1, -1):
                if o == 0:
                    break
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word.extend(block[max_overlap:])
        
        word = new_word
        
        if time.time() > deadline:
            break
    
    # If n is small, just return the constructed word
    if n <= 4:
        return "".join(map(str, word))

    # 2. Optimization Phase: Greedy Deletion with Perturbation
    
    # Map permutations to IDs for fast counting
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    def compute_counts(w):
        counts = [0] * num_perms
        L = len(w)
        for i in range(L - n + 1):
            p = tuple(w[i:i+n])
            if p in perm_to_id:
                counts[perm_to_id[p]] += 1
        return counts

    # Initial check
    initial_counts = compute_counts(word)
    if any(c == 0 for c in initial_counts):
        # Should not happen with valid construction, but just in case
        return "".join(map(str, word))

    best_word = word[:]
    best_len = len(word)
    
    # Helper to check if deletion at index i is valid
    # Returns (is_valid, new_counts) if valid, else (False, None)
    # We don't want to modify global counts in the check, so we simulate locally or revert.
    # For speed, we will modify a local copy of the relevant parts or just revert.
    
    def try_delete(i, w, counts):
        L = len(w)
        if L <= n:
            return False, None
        
        # Old windows containing i: start k in [i-n+1, i]
        # Valid old k: [0, L-n]
        old_start_min = max(0, i - n + 1)
        old_start_max = min(i, L - n)
        
        # New windows formed:
        # Windows in new word starting at k < i that cross the gap:
        # k in [i-n+1, i-1]
        # Valid new k: [0, L-1-n]
        new_start_min = max(0, i - n + 1)
        new_start_max = min(i - 1, L - n - 1)
        
        changes = []
        
        # 1. Decrement old windows
        for k in range(old_start_min, old_start_max + 1):
            p = tuple(w[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] -= 1
                changes.append((pid, -1))
                if counts[pid] < 0:
                    # Revert and fail
                    for cp, delta in changes:
                        counts[cp] -= delta
                    return False, None

        # 2. Increment new windows
        for k in range(new_start_min, new_start_max + 1):
            # Construct new window content
            # w[k:i] + w[i+1:k+n+1]
            w1 = w[k:i]
            w2 = w[i+1:k+n+1]
            p = tuple(w1 + w2)
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] += 1
                changes.append((pid, 1))

        # Check if any count is 0 among the decremented ones
        valid = True
        for pid, delta in changes:
            if delta == -1:
                if counts[pid] == 0:
                    valid = False
                    break
        
        if not valid:
            # Revert
            for pid, delta in changes:
                counts[pid] -= delta
            return False, None
        
        return True, counts

    # Main optimization loop
    current_word = best_word[:]
    current_counts = compute_counts(current_word)
    
    # We will try to improve the word by running greedy deletion multiple times,
    # potentially with random perturbations in between to escape local minima.
    
    # First, do a clean greedy pass
    i = 0
    while i < len(current_word):
        if time.time() > deadline - 5.0:
            break
        valid, new_counts = try_delete(i, current_word, current_counts)
        if valid:
            current_word.pop(i)
            current_counts = new_counts
            if len(current_word) < best_len:
                best_word = current_word[:]
                best_len = len(current_word)
            # Do not increment i
        else:
            i += 1
            
    # Now, try to perturb and re-optimize if we have time
    # This is a simple simulated annealing / local search approach
    # We pick a random position, move a character, and re-run greedy deletion.
    
    iterations = 0
    while time.time() < deadline - 10.0 and iterations < 100:
        iterations += 1
        
        # Start from the best word found so far
        cand_word = best_word[:]
        cand_counts = compute_counts(cand_word)
        
        # Perturb: pick a random index and move the character to a random other index
        if len(cand_word) > 2:
            idx1 = random.randint(0, len(cand_word) - 1)
            idx2 = random.randint(0, len(cand_word) - 1)
            if idx1 != idx2:
                # Move cand_word[idx1] to idx2
                val = cand_word.pop(idx1)
                # Adjust idx2 if necessary
                if idx2 > idx1:
                    idx2 -= 1
                cand_word.insert(idx2, val)
                
                # Recalculate counts after perturbation
                cand_counts = compute_counts(cand_word)
                
                # Check if still valid
                if any(c == 0 for c in cand_counts):
                    # If perturbation broke validity, skip this candidate
                    continue
                
                # Run greedy deletion on the perturbed word
                i = 0
                while i < len(cand_word):
                    if time.time() > deadline - 5.0:
                        break
                    valid, new_counts = try_delete(i, cand_word, cand_counts)
                    if valid:
                        cand_word.pop(i)
                        cand_counts = new_counts
                        if len(cand_word) < best_len:
                            best_word = cand_word[:]
                            best_len = len(cand_word)
                        # Do not increment i
                    else:
                        i += 1
    
    return "".join(map(str, best_word))

# Keep the function name and signature as required
# The task says: Interface: construct(n) -> a word ...
# The current program defines construct(n).
