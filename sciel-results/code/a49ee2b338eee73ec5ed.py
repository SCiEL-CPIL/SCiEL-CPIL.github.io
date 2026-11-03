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

def _check_superpermutation(word, n):
    """Check if word contains all n! permutations of 1..n."""
    target = math.factorial(n)
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if len(set(w)) == n:
            seen.add(w)
            if len(seen) == target:
                return True
    return False

def _remove_redundant(word, n, deadline):
    """
    Try to remove redundant symbols from the superpermutation.
    A symbol at position i is redundant if removing it still yields a valid superpermutation.
    We iterate: keep removing redundant positions until none found or deadline.
    """
    word = list(word)
    n = len(set(word))  # determine n from content
    target = math.factorial(n)
    
    while time.time() < deadline:
        improved = False
        # Process from right to left so indices stay valid
        for i in range(len(word) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Temporarily remove position i
            new_word = word[:i] + word[i+1:]
            if len(new_word) < n:
                break
            # Check if new_word is still a valid superpermutation
            seen = set()
            valid = True
            for j in range(len(new_word) - n + 1):
                w = tuple(new_word[j:j + n])
                if len(set(w)) == n:
                    seen.add(w)
                    if len(seen) == target:
                        valid = True
                        break
                    if j + n > len(new_word):
                        break
            else:
                valid = len(seen) == target
            
            if valid:
                word = new_word
                improved = True
        
        if not improved:
            break
    
    return word

def _local_search(word, n, deadline):
    """
    Local search: try to swap adjacent elements or insert/delete to reduce length.
    This is a heuristic and may not find the optimum but can reduce length.
    """
    word = list(word)
    target = math.factorial(n)
    
    # First, try to remove redundant symbols
    word = _remove_redundant(word, n, deadline)
    
    # Try adjacent swaps to enable further reductions
    max_iterations = 50
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        found = False
        for i in range(len(word) - 1):
            if time.time() > deadline:
                break
            # Swap positions i and i+1
            word[i], word[i+1] = word[i+1], word[i]
            # Check validity
            seen = set()
            valid = True
            for j in range(len(word) - n + 1):
                w = tuple(word[j:j + n])
                if len(set(w)) == n:
                    seen.add(w)
                    if len(seen) == target:
                        valid = True
                        break
            else:
                valid = len(seen) == target
            
            if valid:
                # Check if we can now remove a redundant symbol
                new_word = _remove_redundant(word, n, deadline)
                if len(new_word) < len(word):
                    word = new_word
                    found = True
                else:
                    # Revert swap
                    word[i], word[i+1] = word[i+1], word[i]
            else:
                # Revert swap
                word[i], word[i+1] = word[i+1], word[i]
        
        if not found:
            break
    
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit, use full time
    
    # Generate initial superpermutation
    word = _initial_superpermutation(n)
    
    # Try to improve via local search
    word = _local_search(word, n, deadline)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
