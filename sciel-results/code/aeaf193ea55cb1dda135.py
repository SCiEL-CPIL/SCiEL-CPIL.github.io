import itertools
import time
import random
from collections import deque

def construct(n):
    """
    Construct the shortest superpermutation over symbols 1..n.
    For n < 7, uses the standard optimal construction.
    For n == 7, uses a sophisticated local search on the permutation graph.
    For n > 7, uses the standard construction (safe fallback).
    """
    start_time = time.time()
    timeout = 145.0  # Safety margin from 150s limit
    
    def standard_superperm(n):
        """Standard recursive superpermutation construction. Length = sum(i! for i=1..n)."""
        if n == 1:
            return [1]
        if n == 2:
            return [1, 2, 1]
        
        word = standard_superperm(n - 1)
        new_word = []
        # Extract all (n-1)-permutations from the previous superperm
        perms = []
        seen = set()
        for i in range(len(word) - n + 2):
            window = tuple(word[i:i + n - 1])
            if window not in seen and len(set(window)) == n - 1:
                seen.add(window)
                perms.append(window)
        
        for p in perms:
            block = list(p) + [n] + list(p)
            # Find maximum overlap with current new_word
            max_ov = min(len(new_word), len(block))
            ov = 0
            for o in range(max_ov, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    ov = o
                    break
            new_word += block[ov:]
        
        return new_word
    
    if n < 7:
        word = standard_superperm(n)
        return "".join(map(str, word))
    
    if n > 7:
        word = standard_superperm(n)
        return "".join(map(str, word))
    
    # n == 7 case: sophisticated optimization
    # The standard construction for n=7 gives length = 1!+2!+3!+4!+5!+6!+7! = 1+2+6+24+120+720+5040 = 5913
    # We need to find a shorter one.
    
    # Strategy: 
    # 1. Generate the standard superperm as a starting point
    # 2. Represent it as a sequence of permutations (windows of size n)
    # 3. Use local search to minimize the length by:
    #    a. Reordering permutations to maximize overlaps
    #    b. Deleting redundant characters
    
    # First, let's generate the standard superperm
    word = standard_superperm(n)
    L = len(word)
    
    # Precompute permutation IDs for fast lookup
    perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(perms)}
    num_perms = len(perms)
    
    # Build the sequence of permutation IDs in the standard superperm
    # Each consecutive window of size n is a permutation
    perm_seq = []
    for i in range(L - n + 1):
        w = tuple(word[i:i + n])
        pid = perm_to_id.get(w)
        if pid is not None:
            perm_seq.append(pid)
    
    # Now we have a sequence of permutation IDs.
    # The word can be reconstructed from the permutation sequence.
    # word[0:n] = perms[perm_seq[0]]
    # For i >= 1: word[n-1+i] = perms[perm_seq[i]][n-1]
    # (the last element of each subsequent permutation)
    
    def perm_seq_to_word(perm_seq):
        """Reconstruct word from permutation sequence."""
        if not perm_seq:
            return []
        w = list(perms[perm_seq[0]])
        for i in range(1, len(perm_seq)):
            w.append(perms[perm_seq[i]][-1])
        return w
    
    # Verify that perm_seq_to_word(perm_seq) == word
    # (It should, by construction)
    
    # Local search on the permutation sequence:
    # Goal: minimize the length of the resulting word, which is n + (len(perm_seq) - 1)
    # So we want to minimize len(perm_seq).
    # 
    # But we also need every permutation to appear at least once.
    # 
    # Key insight: The length of the word is n + len(perm_seq) - 1.
    # So minimizing len(perm_seq) is equivalent to minimizing the word length.
    # 
    # However, we can't just remove permutations arbitrarily because:
    # 1. We need every permutation to appear at least once.
    # 2. The reconstruction requires that consecutive permutations share an overlap of n-1.
    #    Specifically, perms[perm_seq[i]][1:] must equal perms[perm_seq[i+1]][:n-1]
    #    for the word to be well-formed.
    
    # Let's check this constraint:
    # In the standard superperm, this constraint is satisfied.
    # When we do local search, we need to maintain it.
    
    # Alternative approach: Work directly on the word and delete characters.
    # This is what the original code was doing, but it's slow.
    
    # Let's try a hybrid: 
    # 1. Start with the standard superperm
    # 2. Try to delete characters that are not essential
    # 3. Use a smarter ordering for checking deletions
    
    # Initialize counts
    counts = [0] * num_perms
    essential = set()
    
    def update_add(w):
        pid = perm_to_id.get(w)
        if pid is not None:
            old_count = counts[pid]
            counts[pid] = old_count + 1
            if old_count == 0:
                essential.add(pid)
            elif old_count == 1:
                essential.discard(pid)
    
    def update_remove(w):
        pid = perm_to_id.get(w)
        if pid is not None:
            old_count = counts[pid]
            counts[pid] = old_count - 1
            if old_count == 1:
                essential.discard(pid)
            elif old_count == 2:
                essential.add(pid)
    
    # Initialize counts for the current word
    L = len(word)
    for i in range(L - n + 1):
        w = tuple(word[i:i + n])
        update_add(w)
    
    # Verify all permutations are covered
    if len(essential) != num_perms:
        # Should not happen with standard superperm
        pass
    
    # Local search: try to delete characters
    # Use a queue-based approach for efficiency
    # Start from both ends and work inward
    
    best_word = list(word)
    best_len = L
    
    # First pass: try to delete from the beginning
    i = 0
    while i < len(word) and time.time() - start_time < timeout - 5:
        valid = True
        windows_to_remove = []
        for s in range(max(0, i - n + 1), min(i, len(word) - n) + 1):
            w = tuple(word[s:s + n])
            pid = perm_to_id.get(w)
            if pid is not None:
                if counts[pid] == 1:
                    valid = False
                    break
                windows_to_remove.append(w)
        
        if valid:
            for w in windows_to_remove:
                update_remove(w)
            word.pop(i)
            L_new = len(word)
            if L_new >= n:
                if i == 0:
                    w_new = tuple(word[0:n])
                    update_add(w_new)
                else:
                    start_s = max(0, i - n + 1)
                    end_s = min(i - 1, L_new - n)
                    for s in range(start_s, end_s + 1):
                        w_new = tuple(word[s:s + n])
                        update_add(w_new)
            if len(word) < best_len:
                best_word = list(word)
                best_len = len(word)
            # Don't increment i, check same position again
        else:
            i += 1
    
    # Second pass: try to delete from the end
    i = len(word) - 1
    while i >= 0 and time.time() - start_time < timeout - 5:
        valid = True
        windows_to_remove = []
        for s in range(max(0, i - n + 1), min(i, len(word) - n) + 1):
            w = tuple(word[s:s + n])
            pid = perm_to_id.get(w)
            if pid is not None:
                if counts[pid] == 1:
                    valid = False
                    break
                windows_to_remove.append(w)
        
        if valid:
            for w in windows_to_remove:
                update_remove(w)
            word.pop(i)
            L_new = len(word)
            if L_new >= n:
                if i == 0:
                    w_new = tuple(word[0:n])
                    update_add(w_new)
                else:
                    start_s = max(0, i - n + 1)
                    end_s = min(i - 1, L_new - n)
                    for s in range(start_s, end_s + 1):
                        w_new = tuple(word[s:s + n])
                        update_add(w_new)
            if len(word) < best_len:
                best_word = list(word)
                best_len = len(word)
            # Don't decrement i, check same position again
        else:
            i -= 1
    
    # Multiple passes of random deletion attempts
    random.seed(42)
    remaining_time = timeout - 5 - (time.time() - start_time)
    
    while time.time() - start_time < timeout - 5:
        # Try random deletions
        success = False
        for _ in range(100):
            if time.time() - start_time > timeout - 5:
                break
            i = random.randint(0, len(word) - 1)
            valid = True
            windows_to_remove = []
            for s in range(max(0, i - n + 1), min(i, len(word) - n) + 1):
                w = tuple(word[s:s + n])
                pid = perm_to_id.get(w)
                if pid is not None:
                    if counts[pid] == 1:
                        valid = False
                        break
                    windows_to_remove.append(w)
            
            if valid:
                for w in windows_to_remove:
                    update_remove(w)
                word.pop(i)
                L_new = len(word)
                if L_new >= n:
                    if i == 0:
                        w_new = tuple(word[0:n])
                        update_add(w_new)
                    else:
                        start_s = max(0, i - n + 1)
                        end_s = min(i - 1, L_new - n)
                        for s in range(start_s, end_s + 1):
                            w_new = tuple(word[s:s + n])
                            update_add(w_new)
                if len(word) < best_len:
                    best_word = list(word)
                    best_len = len(word)
                success = True
                break
        
        if not success:
            # Try swap-based local search
            found_swap = False
            for _ in range(50):
                if time.time() - start_time > timeout - 5:
                    break
                i = random.randint(0, len(word) - 2)
                # Try swapping word[i] and word[i+1]
                word[i], word[i+1] = word[i+1], word[i]
                
                # Check if all permutations are still covered
                # Quick check: verify the windows around the swap
                new_counts = [0] * num_perms
                L_new = len(word)
                for s in range(max(0, i - n + 1), min(i + 1, L_new - n) + 1):
                    w = tuple(word[s:s + n])
                    pid = perm_to_id.get(w)
                    if pid is not None:
                        new_counts[pid] += 1
                
                # Check if any permutation is now missing
                missing = False
                for pid in range(num_perms):
                    if counts[pid] >= 1 and new_counts[pid] == 0:
                        # This permutation was covered, but now might not be
                        # Need to check the full word
                        missing = True
                        break
                
                if missing:
                    # Full verification
                    all_covered = [0] * num_perms
                    for s in range(L_new - n + 1):
                        w = tuple(word[s:s + n])
                        pid = perm_to_id.get(w)
                        if pid is not None:
                            all_covered[pid] += 1
                    if all(c >= 1 for c in all_covered):
                        # Swap is valid
                        # Update counts
                        for s in range(max(0, i - n + 1), min(i + 1, L_new - n) + 1):
                            w = tuple(word[s:s + n])
                            pid = perm_to_id.get(w)
                            if pid is not None:
                                counts[pid] = all_covered[pid]
                        found_swap = True
                        break
                    else:
                        # Revert
                        word[i], word[i+1] = word[i+1], word[i]
                else:
                    # Swap is valid
                    # Update counts properly
                    # Remove old windows, add new windows
                    # This is complex, so just recompute
                    all_covered = [0] * num_perms
                    for s in range(L_new - n + 1):
                        w = tuple(word[s:s + n])
                        pid = perm_to_id.get(w)
                        if pid is not None:
                            all_covered[pid] += 1
                    counts = all_covered
                    for pid in range(num_perms):
                        if counts[pid] == 1:
                            essential.add(pid)
                        else:
                            essential.discard(pid)
                    found_swap = True
                    break
            
            if not found_swap:
                break
    
    # Use the best word found
    if best_len < L:
        word = best_word
    
    return "".join(map(str, word))
