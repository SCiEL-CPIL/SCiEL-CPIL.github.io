import time
import itertools
import random
from collections import Counter

def _classic_superperm(n):
    """Classic recursive construction: length 1!+2!+...+n!."""
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
    return word

def _is_valid_superperm(word, n, all_perms):
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        seen.add(tuple(word[i:i + n]))
    return all_perms.issubset(seen)

def _init_counts(word, n, all_perms):
    """Initialize permutation counts for valid windows only."""
    perm_counts = Counter()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if w in all_perms:
            perm_counts[w] += 1
    return perm_counts

def _try_delete(word, idx, perm_counts, all_perms, n):
    """
    Check if deleting word[idx] is safe.
    Returns (is_safe, new_word, new_counts) or (False, None, None).
    """
    L = len(word)
    if L < n:
        return False, None, None
    
    s_min = max(0, idx - n + 1)
    s_max = min(idx, L - n)
    
    if s_min > s_max:
        # No valid windows affected (shouldn't happen for idx < L)
        new_word = word[:idx] + word[idx+1:]
        return True, new_word, perm_counts
    
    # Check safety: all affected windows must have count > 1
    for s in range(s_min, s_max + 1):
        w = tuple(word[s:s + n])
        if w in all_perms and perm_counts.get(w, 0) <= 1:
            return False, None, None
    
    # Safe to delete
    new_word = word[:idx] + word[idx+1:]
    new_L = len(new_word)
    
    # Update counts: remove old windows, add new windows
    for s in range(s_min, s_max + 1):
        w = tuple(word[s:s + n])
        if w in all_perms:
            perm_counts[w] -= 1
    
    new_s_min = max(0, idx - n + 1)
    new_s_max = min(idx - 1, new_L - n)
    
    for s in range(new_s_min, new_s_max + 1):
        w = tuple(new_word[s:s + n])
        if w in all_perms:
            perm_counts[w] += 1
    
    return True, new_word, perm_counts

def _greedy_pass(word, n, all_perms, strategy, deadline, rng):
    """
    Run a single greedy deletion pass with the given strategy.
    Returns the resulting word.
    """
    current_word = list(word)
    perm_counts = _init_counts(current_word, n, all_perms)
    
    L = len(current_word)
    if L < n:
        return current_word
    
    pass_count = 0
    max_passes = 200  # Safety limit
    
    while time.time() < deadline and pass_count < max_passes:
        pass_count += 1
        changed = False
        
        # Generate indices based on strategy
        if strategy == "left_to_right":
            indices = list(range(len(current_word)))
        elif strategy == "right_to_left":
            indices = list(range(len(current_word) - 1, -1, -1))
        elif strategy == "random":
            indices = list(range(len(current_word)))
            rng.shuffle(indices)
        elif strategy == "alternating":
            # Alternate direction each pass
            if pass_count % 2 == 1:
                indices = list(range(len(current_word)))
            else:
                indices = list(range(len(current_word) - 1, -1, -1))
        else:
            indices = list(range(len(current_word)))
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            if idx >= len(current_word):
                continue
            
            safe, new_word, new_counts = _try_delete(
                current_word, idx, perm_counts, all_perms, n
            )
            if safe:
                current_word = new_word
                perm_counts = new_counts
                changed = True
                # For deterministic strategies, restart scan from beginning
                # For random, just continue with remaining indices
                if strategy in ("left_to_right", "right_to_left", "alternating"):
                    break
        
        if not changed:
            break
    
    return current_word

def construct(n):
    if n == 1:
        return "1"
    if n <= 4:
        word = _classic_superperm(n)
        return "".join(map(str, word))
    
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s limit
    
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Generate classic superpermutation
    word = _classic_superperm(n)
    
    best_word = list(word)
    best_len = len(word)
    
    # Run multiple strategies with different random seeds
    strategies = ["left_to_right", "right_to_left", "random", "alternating"]
    num_random_seeds = 3
    
    # First: run deterministic strategies (fast, good baseline)
    for strategy in ["left_to_right", "right_to_left"]:
        if time.time() > deadline - 15:
            break
        
        polished = _greedy_pass(word, n, all_perms, strategy, deadline - 10, random)
        
        if _is_valid_superperm(polished, n, all_perms):
            if len(polished) < best_len:
                best_len = len(polished)
                best_word = polished
                if n == 7 and best_len <= 5905:
                    break
    
    # Then: run random strategies with different seeds
    for seed in range(num_random_seeds):
        if time.time() > deadline - 15:
            break
        
        rng = random.Random(seed * 7919 + 42)
        polished = _greedy_pass(best_word, n, all_perms, "random", deadline - 10, rng)
        
        if _is_valid_superperm(polished, n, all_perms):
            if len(polished) < best_len:
                best_len = len(polished)
                best_word = polished
                if n == 7 and best_len <= 5905:
                    break
    
    # Final: run alternating strategy on best found
    if time.time() < deadline - 10:
        rng = random.Random(12345)
        polished = _greedy_pass(best_word, n, all_perms, "alternating", deadline - 5, rng)
        
        if _is_valid_superperm(polished, n, all_perms):
            if len(polished) < best_len:
                best_len = len(polished)
                best_word = polished
    
    return "".join(map(str, best_word))
