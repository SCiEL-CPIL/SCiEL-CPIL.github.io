"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
# EVOLVE-BLOCK-START

def _get_perm_set(word, n):
    """Extract the set of all valid n-permutations from word."""
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
    return seen

def _check_valid(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive windows."""
    target = math.factorial(n)
    seen = _get_perm_set(word, n)
    return len(seen) == target

def _try_remove_fast(word, n, deadline):
    """Try to remove symbols using incremental permutation tracking."""
    word = list(word)
    target = math.factorial(n)
    perm_count = {}  # tuple -> count of occurrences
    # Build initial counts
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            perm_count[window] = perm_count.get(window, 0) + 1
    
    def rebuild_counts(w):
        pc = {}
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                pc[window] = pc.get(window, 0) + 1
        return pc
    
    i = 0
    while i < len(word) - n + 1:
        if time.time() > deadline:
            break
        # Try removing word[i]
        test = word[:i] + word[i+1:]
        new_counts = rebuild_counts(test)
        # Check if all permutations still have count >= 1
        if all(c >= 1 for c in new_counts.values()) and len(new_counts) == target:
            word = test
            perm_count = new_counts
            # Don't increment i
        else:
            i += 1
    return word

def construct(n):
    deadline = time.time() + 140  # 140s safety margin for 150s limit
    # Classic recursive construction
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    best_word = word
    best_len = len(word)
    
    # Try to shorten by removing symbols
    if n <= 7:
        # Single pass removal with fast checking
        word2 = _try_remove_fast(word, n, deadline)
        if len(word2) < best_len:
            best_word = word2
            best_len = len(word2)
        
        # Multiple passes if time allows
        while time.time() < deadline - 5:
            word3 = _try_remove_fast(best_word, n, deadline)
            if len(word3) < best_len:
                best_word = word3
                best_len = len(word3)
            else:
                break
    
    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
