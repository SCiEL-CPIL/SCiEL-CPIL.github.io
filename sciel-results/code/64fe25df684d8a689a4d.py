"""Superpermutation construction with heuristic improvement via local search."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import permutations

def _initial_superpermutation(n):
    """Classic recursive construction: length = 1! + 2! + ... + n!."""
    word = [1]
    for m in range(2, n + 1):
        # Collect unique (m-1)-permutations in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        # Insert m: for each permutation p, extend p -> p, m, p
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between end of new and start of block
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _perm_to_mask(perm, n):
    """Convert a permutation tuple to a bitmask for fast lookup."""
    # For n<=7, we can encode each permutation as an integer
    # Using factorial number system or simple encoding
    # For n=7, 5040 perms, use a simple hash
    return hash(perm)

def _check_superpermutation_fast(word, n):
    """Fast check if word contains all n! permutations of 1..n using set of tuples."""
    target = math.factorial(n)
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if len(set(w)) == n:
            seen.add(w)
            if len(seen) == target:
                return True
    return False

def _count_missing(word, n, target_perms=None):
    """Count how many permutations are missing from word. Returns (count, set_of_seen)."""
    target = math.factorial(n)
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if len(set(w)) == n:
            seen.add(w)
    return target - len(seen), seen

def _remove_redundant_fast(word, n, deadline):
    """
    Try to remove redundant symbols from the superpermutation.
    Uses a more efficient approach: for each position, check if removing it
    still covers all permutations.
    """
    word = list(word)
    target = math.factorial(n)
    
    while time.time() < deadline:
        improved = False
        # Try removing from each position, right to left
        for i in range(len(word) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Temporarily remove position i
            new_word = word[:i] + word[i+1:]
            if len(new_word) < n:
                continue
            # Check if new_word is still a valid superpermutation
            seen = set()
            valid = False
            for j in range(len(new_word) - n + 1):
                w = tuple(new_word[j:j + n])
                if len(set(w)) == n:
                    seen.add(w)
                    if len(seen) == target:
                        valid = True
                        break
            if valid:
                word = new_word
                improved = True
        
        if not improved:
            break
    
    return word

def _try_remove_window(word, n, deadline, start_idx, window_size):
    """Try to remove a contiguous block of symbols and check validity."""
    word = list(word)
    target = math.factorial(n)
    
    for i in range(start_idx, min(start_idx + window_size, len(word) - n)):
        if time.time() > deadline:
            break
        new_word = word[:i] + word[i+1:]
        if len(new_word) < n:
            continue
        seen = set()
        valid = False
        for j in range(len(new_word) - n + 1):
            w = tuple(new_word[j:j + n])
            if len(set(w)) == n:
                seen.add(w)
                if len(seen) == target:
                    valid = True
                    break
        if valid:
            return new_word
    return word

def _local_search(word, n, deadline):
    """
    Local search: try various operations to reduce length.
    1. Remove redundant single symbols
    2. Try adjacent swaps that enable further reductions
    3. Try removing small blocks
    """
    word = list(word)
    target = math.factorial(n)
    
    # First, aggressively remove redundant symbols
    word = _remove_redundant_fast(word, n, deadline)
    
    # Try multiple passes of different strategies
    max_rounds = 10
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Strategy 1: Try removing single symbols at various positions
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(word) - 1, -1, -1):
                if time.time() > deadline:
                    break
                new_word = word[:i] + word[i+1:]
                if len(new_word) < n:
                    continue
                seen = set()
                valid = False
                for j in range(len(new_word) - n + 1):
                    w = tuple(new_word[j:j + n])
                    if len(set(w)) == n:
                        seen.add(w)
                        if len(seen) == target:
                            valid = True
                            break
                if valid:
                    word = new_word
                    improved = True
        
        # Strategy 2: Try adjacent swaps that might enable removals
        if time.time() > deadline:
            break
        found_swap = False
        for i in range(len(word) - 1):
            if time.time() > deadline:
                break
            word[i], word[i+1] = word[i+1], word[i]
            # Check if this swap enables a removal
            new_word = _remove_redundant_fast(word, n, deadline)
            if len(new_word) < len(word):
                word = new_word
                found_swap = True
            else:
                # Revert swap
                word[i], word[i+1] = word[i+1], word[i]
        
        if not found_swap and not improved:
            break
    
    return word

def construct(n):
    start_time = time.time()
    # Use 115s deadline to leave 5s safety margin
    deadline = start_time + 115
    
    # Generate initial superpermutation
    word = _initial_superpermutation(n)
    
    # Try to improve via local search
    word = _local_search(word, n, deadline)
    
    # Final safety check: if time is running out, return what we have
    if time.time() > deadline - 2:
        return "".join(map(str, word))
    
    # One more pass of removal if we have time
    word = _remove_redundant_fast(word, n, deadline)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
