import time
from itertools import permutations
import sys
import random

def construct(n):
    # Safety margin: 150s limit, aim to finish by 140s
    deadline = time.time() + 140

    # 1. Generate standard superpermutation
    # Classic recursive construction: length 1! + 2! + ... + n!
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

    # 2. Setup for optimized local search
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts array
    counts = [0] * num_perms
    L = len(word)
    
    # Count initial permutations
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        pid = perm_to_id.get(p)
        if pid is not None:
            counts[pid] += 1
            
    # Check if valid initially
    if any(c == 0 for c in counts):
        return "".join(map(str, word))
    
    # Track number of permutations with count > 0
    covered = sum(1 for c in counts if c > 0)
    
    def try_delete(i):
        """
        Attempts to delete word[i]. 
        Returns True if successful (word and counts updated), False otherwise.
        """
        nonlocal covered
        L_curr = len(word)
        if L_curr <= n:
            return False
        
        # Identify affected old windows
        start_min = max(0, i - n + 1)
        start_max = min(i, L_curr - n)
        
        old_pids = []
        for k in range(start_min, start_max + 1):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                old_pids.append(pid)
                if counts[pid] == 1:
                    covered -= 1
                counts[pid] -= 1
        
        # Identify affected new windows
        new_starts_min = max(0, i - n + 1)
        new_starts_max = min(i - 1, L_curr - 1 - n)
        
        new_pids = []
        for k in range(new_starts_min, new_starts_max + 1):
            w1 = word[k:i]
            w2 = word[i+1:k+n]
            p = tuple(w1 + w2)
            pid = perm_to_id.get(p)
            if pid is not None:
                if counts[pid] == 0:
                    covered += 1
                counts[pid] += 1
                new_pids.append(pid)

        # Check validity: all covered permutations must still be covered
        if covered == num_perms:
            word.pop(i)
            return True
        else:
            # Revert counts
            for pid in old_pids:
                counts[pid] += 1
                if counts[pid] == 1:
                    covered += 1
            for pid in new_pids:
                counts[pid] -= 1
                if counts[pid] == 0:
                    covered -= 1
            return False

    # Local Search Loop with multiple strategies
    best_len = len(word)
    improved = True
    pass_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        i = 0
        L_curr = len(word)
        pass_count += 1
        # Safety check: if L_curr is small, we might finish fast
        while i < L_curr:
            if time.time() > deadline - 5: # 5s safety buffer
                break
            
            if try_delete(i):
                improved = True
                L_curr -= 1
                if L_curr < best_len:
                    best_len = L_curr
                # Do not increment i, as the element at i is now the one that was at i+1
            else:
                i += 1
        
        # If no improvement in a full pass, try a randomized restart
        if not improved and time.time() < deadline - 15:
            # Try deletions in random order to escape local minima
            random.seed(pass_count * 1000 + 42)
            positions = list(range(len(word)))
            random.shuffle(positions)
            
            # Try all positions in random order
            for pos in positions:
                if time.time() > deadline - 15:
                    break
                if pos < len(word):
                    if try_delete(pos):
                        improved = True
                        if len(word) < best_len:
                            best_len = len(word)

    return "".join(map(str, word))

if __name__ == "__main__":
    # Quick self-test for n=4,5,6
    # print(construct(4))
    pass
