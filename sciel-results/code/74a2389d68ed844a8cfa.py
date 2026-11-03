import time
import random
from itertools import permutations
from collections import defaultdict

def construct(n):
    deadline = time.time() + 145  # 5s safety margin before 150s kill

    if n <= 1:
        return "1" if n == 1 else ""

    # Classic recursive construction: length = 1! + 2! + ... + n!
    # For n=7, this gives 1+2+6+24+120+720+5040 = 5913
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
            best_overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, 0, -1):
                if new[-o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new

    if n <= 6:
        return "".join(map(str, word))

    # For n >= 7, apply local search to shorten the word
    all_perms = set(permutations(range(1, n + 1)))
    current = word[:]
    n_local = n
    L = len(current)
    total_perms = len(all_perms)

    # Build reference count map: window -> count of occurrences
    ref_count = defaultdict(int)
    for i in range(L - n_local + 1):
        ref_count[tuple(current[i:i + n_local])] += 1

    # Count how many required permutations are covered
    covered = sum(1 for p in all_perms if ref_count[p] > 0)

    # Verify classic construction is valid
    if covered < total_perms:
        return "".join(map(str, current))

    # Fast local search: try deleting each character using reference counting
    improved = True
    start_time = time.time()
    
    while improved and time.time() - start_time < 140:
        improved = False
        L = len(current)
        num_windows = L - n_local + 1
        
        # Try deleting from left to right
        for i in range(L):
            if time.time() - start_time > 140:
                break
            
            # Check if deleting position i would cause any permutation to become uncovered
            # Windows affected: those that include position i
            # Window starting at j covers positions j..j+n-1
            # Position i is in window j iff j <= i <= j+n-1, i.e., i-n+1 <= j <= i
            # Also j must be in [0, num_windows-1]
            
            j_start = max(0, i - n_local + 1)
            j_end = min(i, num_windows - 1)
            
            # Check if any affected window has ref_count == 1 and is a required permutation
            can_delete = True
            for j in range(j_start, j_end + 1):
                w = tuple(current[j:j + n_local])
                if ref_count[w] == 1 and w in all_perms:
                    can_delete = False
                    break
            
            if can_delete:
                # Save old windows that will be removed
                old_windows = []
                for j in range(j_start, j_end + 1):
                    old_windows.append(tuple(current[j:j + n_local]))
                
                # Perform deletion
                current.pop(i)
                L -= 1
                num_windows -= 1
                improved = True
                
                # Remove old window counts
                for w in old_windows:
                    ref_count[w] -= 1
                    if ref_count[w] == 0:
                        del ref_count[w]
                
                # Add new windows in the affected region
                new_j_start = max(0, i - n_local + 1)
                new_j_end = min(i, L - n_local)
                
                for j in range(new_j_start, new_j_end + 1):
                    w = tuple(current[j:j + n_local])
                    ref_count[w] += 1

    return "".join(map(str, current))
