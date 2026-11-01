"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import permutations

# EVOLVE-BLOCK-START

def construct(n):
    """
    Construct a superpermutation for n symbols.
    Returns a string of digits 1..n.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    if n <= 1:
        return "1"
    
    # For small n, use the classic recursive construction which is optimal
    if n <= 4:
        return _classic_superpermutation(n)
    
    # For n >= 5, use a greedy heuristic with maximum overlap
    # This should be faster and potentially produce shorter results
    
    if n == 5:
        # 5! = 120, classic length is 1+2+6+24+120 = 153
        # Known shortest is 153
        return _classic_superpermutation(5)
    
    if n == 6:
        # 6! = 720, classic length is 1+2+6+24+120+720 = 873
        # Known shortest is 873 (or slightly less with better construction)
        # Let's try a greedy approach
        return _greedy_superpermutation(n, deadline)
    
    if n == 7:
        # 7! = 5040, classic length is 1+2+6+24+120+720+5040 = 5913
        # Known shortest is 5905 (Teodorescu 2026)
        # We need to find something close to 5905
        return _greedy_superpermutation(n, deadline)
    
    # For larger n, use greedy
    return _greedy_superpermutation(n, deadline)


def _classic_superpermutation(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
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
            # Find maximum overlap
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))


def _greedy_superpermutation(n, deadline):
    """
    Greedy heuristic: maintain set of unvisited permutations,
    at each step append the permutation with maximum suffix overlap.
    """
    start_time = time.time()
    
    # Start with permutation (1, 2, ..., n)
    word = list(range(1, n + 1))
    
    # Track which permutations have been seen
    all_perms = set(permutations(range(1, n + 1)))
    seen_perms = set()
    
    # Add the initial permutation
    initial_perm = tuple(word)
    seen_perms.add(initial_perm)
    remaining = all_perms - seen_perms
    
    # For efficiency, we'll track the last (n-1) symbols as a tuple
    # and use it to find overlapping permutations
    
    while remaining:
        if time.time() > deadline - 5:
            break
        
        current_suffix = tuple(word[-(n - 1):])
        
        # Find all permutations that can be formed by extending the current suffix
        # A permutation p = (p_0, ..., p_{n-1}) can follow if p[1:] == current_suffix
        # So we need p such that p[1:] == current_suffix, which means p = (x, current_suffix[0], ..., current_suffix[n-2])
        # where x is the missing symbol
        
        best_perm = None
        best_overlap = 0
        
        # The only possible next symbol is the one not in current_suffix
        suffix_set = set(current_suffix)
        missing = [x for x in range(1, n + 1) if x not in suffix_set]
        
        if len(missing) == 1:
            candidate = (missing[0],) + current_suffix
            if candidate in remaining:
                # This permutation can be added with overlap n-1
                word.append(missing[0])
                seen_perms.add(candidate)
                remaining.discard(candidate)
                # Check for new permutations formed
                # The new last n symbols form a permutation
                new_perm = tuple(word[-n:])
                if new_perm in remaining:
                    seen_perms.add(new_perm)
                    remaining.discard(new_perm)
                continue
        
        # If no perfect overlap, try to find the best partial overlap
        # This is more complex; for now, just pick any remaining permutation
        # that has the longest suffix match
        
        # For efficiency with large n, we need a better strategy
        # Let's use a different approach: build the word by adding symbols one at a time
        
        # Actually, let's restart with a simpler greedy:
        # At each step, try adding each symbol 1..n and see which creates the most new permutations
        best_symbol = None
        best_new_count = -1
        
        for sym in range(1, n + 1):
            # Simulate adding sym
            new_word_suffix = word[-(n - 1):] + [sym]
            if len(new_word_suffix) > n:
                new_word_suffix = new_word_suffix[-n:]
            
            # Count how many new permutations this would create
            new_count = 0
            # The new permutation is the last n symbols
            perm = tuple(new_word_suffix)
            if perm in remaining:
                new_count += 1
            
            # Also check if any other windows are formed (only the last one is new)
            
            if new_count > best_new_count:
                best_new_count = new_count
                best_symbol = sym
        
        if best_symbol is None:
            break
        
        word.append(best_symbol)
        
        # Update seen permutations
        new_perm = tuple(word[-n:])
        if new_perm in remaining:
            seen_perms.add(new_perm)
            remaining.discard(new_perm)
    
    return "".join(map(str, word))


# EVOLVE-BLOCK-END
