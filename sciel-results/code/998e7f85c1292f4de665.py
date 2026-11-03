import time
import random
import sys

def _recursive_superperm(n):
    """Classic recursive construction of length sum(1!..n!)."""
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
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _verify(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive n-tuples."""
    from itertools import permutations
    total = 1
    for i in range(2, n + 1):
        total *= i
    found = set()
    for i in range(len(word) - n + 1):
        found.add(tuple(word[i:i + n]))
    if len(found) != total:
        return False, len(found)
    # Verify all are valid permutations
    for t in found:
        if len(set(t)) != n or not all(1 <= x <= n for x in t):
            return False, len(found)
    return True, len(found)

def _local_improve(word, n, deadline):
    """
    Try to shorten the word by finding and removing redundant internal permutations.
    Returns improved word or original if no improvement found.
    """
    word = list(word)
    if len(word) < 2 * n:
        return word
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 5:
            break
        # Find all occurrences of each permutation
        perm_positions = {}
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i + n])
            if len(set(p)) == n and all(1 <= x <= n for x in p):
                perm_positions.setdefault(p, []).append(i)
        
        # Count how many times each permutation appears
        redundant_perms = []
        for p, positions in perm_positions.items():
            if len(positions) > 1:
                redundant_perms.append((p, positions))
        
        # For each redundant permutation, try to remove an internal occurrence
        for p, positions in redundant_perms:
            if time.time() > deadline - 5:
                break
            # Sort positions, skip first and last
            internal = positions[1:-1]
            for pos in internal:
                # Check if removing word[pos] to word[pos+n-1] (keeping first and last char)
                # Actually, to remove a permutation at position pos, we need to see if the
                # prefix before and suffix after can be connected.
                # The permutation occupies word[pos..pos+n-1].
                # If we remove it, we'd need to connect word[:pos] with word[pos+n-1:]
                # but we need to keep the overlap. Let's think differently.
                
                # Alternative: if a permutation appears at pos, and the (n-1)-prefix before it
                # matches the (n-1)-suffix after it, we might be able to shortcut.
                # Actually, the standard approach: if we have ... A P B ... where P is a permutation
                # and A ends with first n-1 of P, and B starts with last n-1 of P, then P is redundant
                # if the (n-1)-window before P and (n-1)-window after P are the same.
                
                if pos == 0 or pos + n > len(word) - 1:
                    continue
                # Window before: word[pos-(n-1) : pos] (if it exists and is a valid (n-1)-tuple)
                # Window after: word[pos+n-1 : pos+2*(n-1)] (if it exists)
                if pos < n - 1:
                    continue
                if pos + n > len(word) - (n - 1):
                    continue
                
                before = tuple(word[pos - (n - 1):pos])
                after = tuple(word[pos + n - 1:pos + 2 * (n - 1)])
                
                if len(set(before)) == n - 1 and len(set(after)) == n - 1:
                    if before == after:
                        # This permutation is redundant! We can remove it.
                        # Remove word[pos : pos + n - 1] (keep the last char as it's the start of 'after')
                        # Actually, we remove n-1 characters: word[pos : pos + n - 1]
                        new_word = word[:pos] + word[pos + n - 1:]
                        # Verify the new word is still valid
                        valid, _ = _verify(new_word, n)
                        if valid:
                            word = new_word
                            improved = True
                            break
            if improved:
                break
    
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    if n <= 7:
        # Use the classic recursive construction which is known to be near-optimal for small n
        word = _recursive_superperm(n)
    else:
        # For larger n, we might need a different approach, but the task focuses on n=7
        # Fallback: use recursive for now
        word = _recursive_superperm(n)
    
    # Try local improvements
    if len(word) > n:
        word = _local_improve(word, n, deadline)
    
    return "".join(map(str, word))
