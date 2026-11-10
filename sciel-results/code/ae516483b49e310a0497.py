"""Superpermutation solver using de Bruijn graph Hamiltonian path search."""
import time
import math
import random
import sys
from functools import lru_cache

# Global deadline guard
_START_TIME = None
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0  # seconds

def _remaining_time():
    if _START_TIME is None:
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def construct(n):
    global _START_TIME
    _START_TIME = time.time()
    
    # For small n, use exact known constructions
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Known shortest for n=4 is length 33
        return _classic_construction(n)
    
    if n == 5:
        # Known shortest for n=5 is 152
        # Try to find it, fall back to classic (153)
        result = _try_de_bruijn_search(n)
        if result is not None:
            return result
        return _classic_construction(n)
    
    if n >= 6:
        # For larger n, use classic construction
        # Classic gives 1!+2!+...+n!
        # For n=7: 1+2+6+24+120+720+5040 = 5913
        # Known best is 5905, lower bound 5884
        result = _classic_construction(n)
        # Try local search improvements if time permits
        if _remaining_time() > 10.0:
            improved = _local_search_improve(result, n)
            if improved is not None and len(improved) < len(result):
                return improved
        return result
    
    if n == 5:
        # Known shortest for n=5 is 152
        # Try to find it, fall back to classic (153)
        result = _try_de_bruijn_search(n)
        if result is not None:
            return result
        return _classic_construction(n)
    
    return _classic_construction(n)

def _classic_construction(n):
    """The classic recursive construction of length 1! + 2! + ... + n!."""
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
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _get_perm_set(word, n):
    """Get the set of all length-n permutations in word."""
    perms = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n:
            perms.add(window)
    return perms

def _try_de_bruijn_search(n):
    """
    Find a shortest superpermutation by finding a Hamiltonian path in the 
    de Bruijn graph where nodes are (n-1)-permutations and edges represent
    appending a symbol to form a new n-permutation.
    
    For n=5: 4! = 24 nodes, need to visit all 5! = 120 edges (each edge 
    corresponds to an n-permutation).
    
    Actually, let's model it correctly:
    - A superpermutation of length L has L-n+1 windows of length n
    - Each window is a permutation of 1..n
    - Consecutive windows overlap in n-1 positions
    - So we need a sequence of n! permutations where consecutive ones 
      overlap in n-1 positions
    
    This is equivalent to finding an Eulerian path in a graph where:
    - Nodes are (n-1)-permutations (of n-1 distinct elements from 1..n)
    - An edge from node u=(a1,...,a_{n-1}) to node v=(b1,...,b_{n-1}) exists 
      if u[1:] == v[:n-2] ... wait, that's not right.
    
    Let me reconsider. If we have a word W, the windows are W[0:n], W[1:n+1], ...
    Window i: W[i:i+n]
    Window i+1: W[i+1:i+n+1]
    The overlap is W[i+1:i+n] which is the last n-1 chars of window i and 
    first n-1 chars of window i+1.
    
    So if window i = (a1, a2, ..., an), then window i+1 must start with 
    (a2, ..., an) and end with some new symbol x.
    So window i+1 = (a2, ..., an, x) where x is the symbol not in {a2,...,an}.
    
    This means each window determines the next window uniquely! Given window i,
    there's exactly one possible next window (by dropping the first element and 
    appending the missing symbol).
    
    Wait, that can't be right because then the superpermutation would be 
    completely determined by its first window, giving a fixed length.
    
    Let me re-examine. The key insight: in a superpermutation, not every 
    consecutive pair of windows must be "valid" in the sense that the second 
    is determined by the first. The word just needs to contain all permutations 
    as substrings, but consecutive windows in the word don't have to form 
    valid transitions in the permutation graph.
    
    Actually, I think I was overcomplicating this. Let me use a different 
    approach: greedy construction with backtracking for n=5.
    """
    if n > 5:
        return None
    
    # For n=5, use a more direct approach
    # The classic construction gives 153, optimal is 152
    # Let's try to find a 152-length solution
    
    best_result = None
    best_len = float('inf')
    
    # Try multiple random starting permutations
    for trial in range(200):
        if _remaining_time() < 5.0:
            break
        
        # Start with a random permutation
        start = list(range(1, n + 1))
        random.shuffle(start)
        
        word = list(start)
        visited = {tuple(start)}
        target = math.factorial(n)
        
        # Greedily extend
        while len(visited) < target:
            if _remaining_time() < 5.0:
                break
            
            last_n_minus_1 = tuple(word[-(n-1):])
            
            # The only symbol we can append is the one not in last_n_minus_1
            missing = [x for x in range(1, n+1) if x not in last_n_minus_1]
            
            if not missing:
                break
            
            x = missing[0]  # Only one choice
            new_window = last_n_minus_1[1:] + (x,)
            
            # Check if this is a valid permutation (all distinct)
            if len(set(new_window)) != n - 1:
                break
            
            full_window = last_n_minus_1 + (x,)
            
            if full_window in visited:
                # Already seen this permutation, but we still need to continue
                # to potentially find new ones later
                pass
            
            word.append(x)
            visited.add(full_window)
        
        if len(visited) == target and len(word) < best_len:
            best_len = len(word)
            best_result = "".join(map(str, word))
            
            if best_len == 152:  # Optimal for n=5
                break
    
    return best_result

def _local_search_improve(word_str, n):
    """
    Try to improve the classic construction using local search.
    For n>=6, the classic construction may not be optimal.
    """
    if n < 6:
        return None
    
    word = [int(c) for c in word_str]
    original_len = len(word)
    
    # Verify the word is valid
    if not _is_valid_superpermutation(word, n):
        return None
    
    target_perms = math.factorial(n)
    
    # Try to shorten by removing redundant symbols
    # A symbol at position i is redundant if removing it still gives a valid superpermutation
    # We use an incremental approach: track which permutations are covered by which windows
    
    best_word = word[:]
    best_len = len(word)
    
    # Multiple passes of deletion
    for iteration in range(20):
        if _remaining_time() < 5.0:
            break
        
        improved = False
        # Try deleting each position (from end to start to keep indices valid)
        for i in range(len(best_word) - 1, -1, -1):
            if _remaining_time() < 5.0:
                break
            
            # Skip if this would make the word too short
            if len(best_word) - 1 < n:
                continue
            
            # Quick check: is this position part of any unique permutation?
            # We'll do a full check but with early termination
            
            # Create a new word without position i
            test_word = best_word[:i] + best_word[i+1:]
            
            # Check if still valid - optimized check
            if _is_valid_superpermutation_fast(test_word, n, target_perms):
                best_word = test_word
                improved = True
        
        if not improved:
            break
    
    if len(best_word) < original_len:
        return "".join(map(str, best_word))
    
    return None

def _is_valid_superpermutation_fast(word, n, target_perms):
    """Fast validity check with early termination."""
    if len(word) < n:
        return False
    
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n:
            seen.add(window)
            if len(seen) == target_perms:
                return True
    
    return len(seen) == target_perms

def _is_valid_superpermutation(word, n):
    """Check if word contains all permutations of 1..n as substrings."""
    if len(word) < n:
        return False
    
    perms = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n:
            perms.add(window)
    
    return len(perms) == math.factorial(n)

# Test
if __name__ == "__main__":
    for n in range(1, 8):
        start = time.time()
        result = construct(n)
        elapsed = time.time() - start
        print(f"n={n}: length={len(result)}, time={elapsed:.2f}s")
        if n <= 5:
            print(f"  word={result}")
