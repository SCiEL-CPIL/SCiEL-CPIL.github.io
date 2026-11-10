"""Superpermutation construction with local search optimization."""
import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110

    # Step 1: Build the classic superpermutation
    word = _build_classic(n)
    
    if n <= 4:
        return word
    
    # For n=7, the classic construction gives a valid superpermutation
    # but it's not optimal. Let's try to optimize it.
    
    # Step 2: Try to reduce length via local search
    # The idea: if we can remove some symbols while maintaining validity, we do so.
    
    if time.time() > deadline - 30:
        return word
    
    # Convert to list of ints for easier manipulation
    symbols = [ord(c) - 48 for c in word]
    
    # Quick validity check function
    def is_valid(syms):
        if len(syms) < n:
            return False
        seen = set()
        for i in range(len(syms) - n + 1):
            window = tuple(syms[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return len(seen) == n
    
    # This is expensive for n=7 (5040 permutations), so let's be smarter
    # Use a bitset approach for tracking which permutations we've seen
    
    def count_permutations(syms):
        """Count distinct permutations of 1..n found as consecutive windows."""
        if len(syms) < n:
            return 0
        # Map each permutation to an index using Lehmer code or similar
        # For n=7, there are 5040 permutations
        seen = set()
        for i in range(len(syms) - n + 1):
            window = tuple(syms[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return len(seen)
    
    # Precompute all permutations for faster checking
    all_perms = set(permutations(range(1, n+1)))
    
    def is_valid_fast(syms):
        if len(syms) < n:
            return False
        seen = set()
        for i in range(len(syms) - n + 1):
            window = tuple(syms[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return seen == all_perms
    
    # Try removing symbols one at a time
    # Start from the end and work backwards
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Try removing each symbol
        for i in range(len(symbols) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            # Try removing symbol at position i
            new_syms = symbols[:i] + symbols[i+1:]
            if is_valid_fast(new_syms):
                symbols = new_syms
                improved = True
    
    return ''.join(str(s) for s in symbols)

def _build_classic(n):
    """Build the classic recursive superpermutation."""
    if n == 1:
        return "1"
    
    # Get the superpermutation for n-1
    prev = _build_classic(n - 1)
    
    # Extract all (n-1)-length permutations from prev
    perms = []
    seen = set()
    for i in range(len(prev) - (n-1) + 1):
        window = prev[i:i+n-1]
        if len(set(window)) == n-1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    # Build new superpermutation
    result = ""
    for p in perms:
        block = p + str(n) + p
        # Find maximum overlap
        max_ov = min(len(result), len(block))
        ov = 0
        for o in range(max_ov, 0, -1):
            if result[-o:] == block[:o]:
                ov = o
                break
        result += block[ov:]
    
    return result
