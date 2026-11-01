import time
import random
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 110

    # 1. Generate the naive superpermutation
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    best_word = word[:]
    total_perms = 1
    for i in range(2, n + 1):
        total_perms *= i
    
    # Precompute all permutations
    all_perms = list(itertools.permutations(range(1, n + 1)))
    all_perms_set = set(all_perms)
    
    # Helper to compute permutation counts
    def compute_counts(w):
        counts = defaultdict(int)
        for i in range(len(w) - n + 1):
            t = tuple(w[i:i+n])
            if t in all_perms_set:
                counts[t] += 1
        return counts
    
    def is_valid_from_counts(counts):
        for p in all_perms:
            if counts[p] < 1:
                return False
        return True
    
    current_len = len(best_word)
    current_counts = compute_counts(best_word)
    
    if not is_valid_from_counts(current_counts):
        return "".join(map(str, best_word))
    
    # Main search loop
    while time.time() < deadline:
        found_improvement = False
        
        # Phase 1: Try to find redundant positions to remove
        for idx in range(len(best_word)):
            if len(best_word) <= n:
                break
            start_lo = max(0, idx - n + 1)
            start_hi = min(idx, len(best_word) - n)
            
            can_remove = True
            for s in range(start_lo, start_hi + 1):
                t = tuple(best_word[s:s+n])
                if t in all_perms_set and current_counts[t] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                candidate = best_word[:]
                del candidate[idx]
                new_counts = compute_counts(candidate)
                if is_valid_from_counts(new_counts):
                    best_word = candidate
                    current_len = len(best_word)
                    current_counts = new_counts
                    found_improvement = True
                    break
        
        # Phase 2: If no deletion found, try random moves
        if not found_improvement:
            for _ in range(2000):
                if time.time() > deadline:
                    break
                if len(best_word) <= n:
                    break
                idx = random.randint(0, len(best_word) - 1)
                new_idx = random.randint(0, len(best_word) - 1)
                if idx == new_idx:
                    continue
                candidate = best_word[:]
                val = candidate.pop(idx)
                candidate.insert(new_idx, val)
                
                if len(candidate) < current_len:
                    new_counts = compute_counts(candidate)
                    if is_valid_from_counts(new_counts):
                        best_word = candidate
                        current_len = len(best_word)
                        current_counts = new_counts
                        break
    
    return "".join(map(str, best_word))
