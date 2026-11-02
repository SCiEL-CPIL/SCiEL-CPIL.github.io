import time
import itertools
import random
from collections import Counter

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Classic Recursive Construction
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
    
    if n <= 4:
        return "".join(map(str, word))
    
    # 2. Optimization
    total_perms = 1
    for i in range(2, n + 1):
        total_perms *= i
        
    # Pre-compute all valid permutations for fast lookup
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        # Optimization: Use a set to track seen perms
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if window in all_perms:
                seen.add(window)
            if len(seen) == total_perms:
                return True
        return len(seen) == total_perms

    # Initial check
    if not is_valid(word):
        return "".join(map(str, word))
        
    best_word = word[:]
    best_len = len(best_word)
    
    # Phase 1: Greedy Deletion Pass (Left to Right)
    # This is fast and removes obvious redundancies.
    def greedy_delete_pass(w):
        i = 0
        while i < len(w):
            if time.time() > deadline:
                break
            # Try to delete i
            candidate = w[:i] + w[i+1:]
            if is_valid(candidate):
                w = candidate
                # i stays same, check new element at i
            else:
                i += 1
        return w

    word = greedy_delete_pass(word)
    best_word = word[:]
    best_len = len(best_word)
    
    # Phase 2: Random Local Search
    # If we are stuck (no more deletions), try perturbations.
    # We will run this until time limit.
    
    # To speed up is_valid, we can use a Counter-based incremental check?
    # No, O(L) is fast enough for ~1000-5000 iterations in 100s.
    # 1000 iterations * 50ms = 50s.
    # Let's aim for efficient checks.
    
    # We can also try swapping adjacent elements to break local minima.
    
    while time.time() < deadline - 5:
        op = random.random()
        candidate = best_word[:]
        
        if op < 0.7:
            # Delete
            if len(candidate) <= n:
                continue
            idx = random.randint(0, len(candidate) - 1)
            del candidate[idx]
        elif op < 0.9:
            # Swap adjacent
            if len(candidate) < 2:
                continue
            idx = random.randint(0, len(candidate) - 2)
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
        else:
            # Move
            if len(candidate) < 2:
                continue
            idx = random.randint(0, len(candidate) - 1)
            new_idx = random.randint(0, len(candidate) - 1)
            if idx == new_idx:
                continue
            val = candidate.pop(idx)
            candidate.insert(new_idx, val)
            
        if is_valid(candidate):
            if len(candidate) < best_len:
                best_word = candidate
                best_len = len(candidate)
            # Even if not shorter, keep it? 
            # If same length, we might be in a different local minimum.
            # But we want to minimize length.
            # If we swap and length is same, we might enable future deletions.
            # So accept if valid, regardless of length change?
            # Yes, accept valid states.
            best_word = candidate
            best_len = len(candidate)
            
    return "".join(map(str, best_word))
