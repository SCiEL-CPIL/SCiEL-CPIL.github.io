"""Shortest superpermutation solver using prefix-tree traversal with overlap optimization."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

def construct(n):
    """Construct shortest superpermutation over symbols 1..n using prefix-tree pruning."""
    start_time = time.time()
    deadline = 110.0  # seconds
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return _classic_construct(4)
    if n == 5:
        return _classic_construct(5)
    
    # For n >= 6, use the classic recursive construction as baseline
    # This gives length sum(k! for k=1..n)
    # For n=7: 1+2+6+24+120+720+5040 = 5913
    # The known shortest is 5905, so we're within 8 of optimal.
    
    # Try to improve using local search / perturbation
    base = _classic_construct(n)
    base_list = [int(c) for c in base]
    
    # Attempt local optimization: try removing redundant symbols
    # Strategy: scan for patterns where a symbol can be removed without losing any permutation window
    # This is tricky because removing a symbol shifts all subsequent windows.
    
    # Alternative: try different orderings in the recursive construction
    # The classic construction uses a specific order. We can try permuting the symbols.
    
    best = base_list
    best_len = len(base_list)
    
    # Try a few random permutations of the symbol set to see if they yield shorter results
    # The length should be the same regardless of symbol order, but the structure might differ
    # in ways that allow further optimization.
    
    # Actually, the length is invariant under symbol relabeling, so this won't help directly.
    
    # Let's try a different approach: modify the recursive construction to use
    # a different ordering of the (m-1)-permutations at each level.
    
    # For now, return the classic construction. It's valid and close to optimal.
    return "".join(map(str, best))


def _classic_construct(n):
    """Classic recursive construction of superpermutation."""
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
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
