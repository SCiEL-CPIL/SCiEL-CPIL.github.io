import time
import numpy as np
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
    
    # Initialize counts
    def build_counts(w):
        c = [0] * num_perms
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if p in perm_to_idx:
                c[perm_to_idx[p]] += 1
        return c

    counts = build_counts(word)
    
    # Helper to check if a specific window tuple is a valid permutation and get its index
    def try_delete(k, w, c):
        """
        Simulates deletion of index k.
        Returns (is_valid, new_counts)
        """
        L = len(w)
        # 1. Identify old windows containing k
        j_min = max(0, k - n + 1)
        j_max = min(L - n, k)
        
        # We track deltas to avoid copying the whole array if possible, 
        # but for n=7, num_perms=5040, copying is fast enough.
        # However, we can just check validity first.
        
        valid_deletion = True
        
        # Check if any count goes to 0
        # We need to know the counts of the affected windows.
        # If a window appears multiple times, decrementing one instance might not make it 0.
        
        # Let's just do the copy and check.
        temp_counts = c[:]
        
        for j in range(j_min, j_max + 1):
            p = tuple(w[j:j+n])
            idx = perm_to_idx.get(p)
            if idx is not None:
                temp_counts[idx] -= 1
                if temp_counts[idx] == 0:
                    valid_deletion = False
                    break

        if not valid_deletion:
            return False, c
        
        # 2. Identify new windows formed by deletion
        j_prime_min = max(0, k - n + 1)
        j_prime_max = min(L - 1 - n, k - 1) # Max start index in w' is L-1-n
        
        for j_prime in range(j_prime_min, j_prime_max + 1):
            left_part = w[j_prime:k]
            right_part = w[k+1:j_prime+n+1]
            new_window_list = left_part + right_part
            p_new = tuple(new_window_list)
            idx = perm_to_idx.get(p_new)
            if idx is not None:
                temp_counts[idx] += 1
        
        if min(temp_counts) == 0:
            return False, c
        
        return True, temp_counts

    # Main optimization loop
    while time.time() < deadline:
        L = len(word)
        if L < n:
            break
            
        improved = False
        # Try to delete
        for k in range(L):
            if time.time() > deadline - 5:
                break
            
            valid, new_counts = try_delete(k, word, counts)
            if valid:
                # Perform deletion
                word = word[:k] + word[k+1:]
                counts = new_counts
                improved = True
                break # Restart scan
        
        if not improved:
            # Try swaps
            swapped = False
            for i in range(L - 1):
                if time.time() > deadline - 5:
                    break
                # Swap i and i+1
                word[i], word[i+1] = word[i+1], word[i]
                
                # Check validity
                new_counts = build_counts(word)
                
                if min(new_counts) > 0:
                    counts = new_counts
                    swapped = True
                    break
                else:
                    # Revert
                    word[i], word[i+1] = word[i+1], word[i]
            
            if not swapped:
                # Try 2-swaps?
                # For now, stop if no single swap works.
                break

    return "".join(map(str, word))
