"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import Counter

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Classic recursive construction of superpermutation
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # For n <= 4, the classic construction is optimal
    if n <= 4:
        return "".join(map(str, word))
    
    # For n >= 5, attempt to shorten the word using structured deletion
    # Precompute all permutations of 1..n for validity checking
    all_perms_set = set(itertools.permutations(range(1, n + 1)))
    
    # Count how many times each permutation appears as a window
    perm_counts = Counter()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        perm_counts[window] += 1
    
    # Helper to check if a word is valid
    def is_valid(word_list):
        if len(word_list) < n:
            return False
        seen = set()
        for i in range(len(word_list) - n + 1):
            seen.add(tuple(word_list[i:i + n]))
        return all_perms_set.issubset(seen)
    
    # Structured deletion: identify block boundaries in the classic construction
    # In the classic construction, each "n" symbol that is NOT 1..n-1's last symbol
    # marks a block transition. We can try to merge adjacent blocks.
    
    # First, let's do aggressive individual position deletion with efficient checking
    def greedy_delete(word, perm_counts, max_iterations=100000):
        """Aggressively delete positions that don't uncover any permutation."""
        deleted_count = 0
        iteration = 0
        while iteration < max_iterations:
            if time.time() - start_time > deadline - 5:
                break
            
            iteration += 1
            found_deletion = False
            
            # Scan from left to right, try to delete each position
            for i in range(len(word)):
                if time.time() - start_time > deadline - 5:
                    break
                
                # Determine which windows are affected by removing position i
                s_min = max(0, i - n + 1)
                s_max = min(i, len(word) - n)
                
                # Check if removing position i would uncover any permutation
                safe = True
                for s in range(s_min, s_max + 1):
                    window = tuple(word[s:s + n])
                    if window in all_perms_set and perm_counts[window] == 1:
                        safe = False
                        break
                
                if safe:
                    # Remove position i and update counts
                    word.pop(i)
                    deleted_count += 1
                    found_deletion = True
                    
                    # Update perm_counts: remove windows that contained position i
                    for s in range(s_min, s_max + 1):
                        if s < len(word) - n + 1:
                            window = tuple(word[s:s + n])
                            if window in perm_counts:
                                perm_counts[window] -= 1
                                if perm_counts[window] <= 0:
                                    del perm_counts[window]
                    
                    # Add new windows that formed after deletion
                    new_s_min = max(0, i - n + 1)
                    new_s_max = min(i, len(word) - n)
                    for s in range(new_s_min, new_s_max + 1):
                        if s < len(word) - n + 1:
                            window = tuple(word[s:s + n])
                            perm_counts[window] += 1
                    
                    break  # Restart scan as indices changed
            
            if not found_deletion:
                break
        
        return deleted_count
    
    # Apply greedy deletion
    greedy_delete(word, perm_counts, max_iterations=50000)
    
    # Second pass: more aggressive deletion
    if time.time() - start_time < deadline - 10:
        greedy_delete(word, perm_counts, max_iterations=50000)
    
    # Third pass if time permits
    if time.time() - start_time < deadline - 5:
        greedy_delete(word, perm_counts, max_iterations=20000)
    
    # Verify validity before returning
    if not is_valid(word):
        # Fallback: return the original classic construction
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
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
