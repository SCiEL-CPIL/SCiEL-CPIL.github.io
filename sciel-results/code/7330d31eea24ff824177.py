"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import permutations

# EVOLVE-BLOCK-START

def _classic_superperm(n):
    """Build the classic superpermutation of length sum(k! for k=1..n)."""
    word = [1]
    for m in range(2, n + 1):
        # Collect all (m-1)-permutations in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between end of new and start of block
            overlap = 0
            max_ov = min(len(new), len(block) - 1)
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _is_valid_superperm(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive windows."""
    expected = math.factorial(n)
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            seen.add(w)
            if len(seen) == expected:
                return True
    return len(seen) == expected

def _count_unique_perms(word, n):
    """Count unique n-permutations in word."""
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            seen.add(w)
    return len(seen)

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # For n <= 4, classic construction is optimal
    if n <= 4:
        word = _classic_superperm(n)
        return "".join(map(str, word))
    
    # Build classic superpermutation as baseline
    best_word = _classic_superperm(n)
    best_len = len(best_word)
    
    # Verify baseline
    if not _is_valid_superperm(best_word, n):
        # Should never happen, but safety
        return "".join(map(str, best_word))
    
    # For n=5, optimal is 152 (classic gives 153)
    # For n=6, optimal is 872 (classic gives 1+2+6+24+120+720=873)
    # For n=7, optimal is 5905 (classic gives 5913)
    
    # Strategy: local search - try to remove redundant symbols
    # A symbol at position i is "removable" if removing it doesn't lose any permutation
    
    if n >= 5:
        # Local search: iteratively try to remove symbols
        # Start from the classic construction and try to shorten it
        
        # First, let's try a smarter construction for n=5
        if n == 5:
            # Known optimal length is 152
            # Try to find it via local search from classic (153)
            word = best_word[:]
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(1, len(word) - 1):
                    if time.time() > deadline - 5.0:
                        break
                    # Try removing word[i]
                    candidate = word[:i] + word[i+1:]
                    if len(candidate) < len(word) and _is_valid_superperm(candidate, n):
                        word = candidate
                        improved = True
                        break
            if len(word) < best_len:
                best_word = word
                best_len = len(word)
        
        # For n=6 and n=7, try local search
        if n >= 6:
            word = best_word[:]
            # Local search with multiple passes
            max_passes = 50
            for _pass in range(max_passes):
                if time.time() > deadline - 10.0:
                    break
                improved = False
                # Try removing each position
                for i in range(1, len(word) - 1):
                    if time.time() > deadline - 10.0:
                        break
                    # Quick check: if word[i-1] == word[i+1], removal is likely safe
                    # But we need to verify
                    candidate = word[:i] + word[i+1:]
                    if _is_valid_superperm(candidate, n):
                        word = candidate
                        improved = True
                        break
                if not improved:
                    break
            if len(word) < best_len:
                best_word = word
                best_len = len(word)
        
        # Additional: try swapping adjacent symbols to enable further removals
        if n >= 6 and time.time() < deadline - 20.0:
            word = best_word[:]
            # Try local moves: swap, then try to remove
            for _pass in range(30):
                if time.time() > deadline - 20.0:
                    break
                improved = False
                for i in range(1, len(word) - 2):
                    if time.time() > deadline - 20.0:
                        break
                    # Try swapping word[i] and word[i+1]
                    candidate = word[:]
                    candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
                    # After swap, try to remove a symbol
                    for j in range(1, len(candidate) - 1):
                        if time.time() > deadline - 20.0:
                            break
                        candidate2 = candidate[:j] + candidate[j+1:]
                        if _is_valid_superperm(candidate2, n):
                            word = candidate2
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            if len(word) < best_len:
                best_word = word
                best_len = len(word)
    
    return "".join(map(str, best_word))

# EVOLVE-BLOCK-END
