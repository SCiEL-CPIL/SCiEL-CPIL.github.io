"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

def construct(n):
    """
    Construct a superpermutation over symbols 1..n.
    
    Strategy:
    1. Start with the classic recursive construction (length = sum_{k=1}^n k!).
    2. Use local search to shorten the word while maintaining the property
       that every permutation appears as a consecutive window.
    
    For n <= 7, we can do extensive local search.
    """
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # Step 1: Classic recursive construction
    word = _classic_superperm(n)
    
    # For small n, the classic construction is already optimal or near-optimal
    # For n=7, classic gives 5913, we need to get closer to 5905
    
    if n <= 4:
        # For small n, we can verify and potentially optimize
        return _optimize_small_n(word, n, start_time, deadline)
    
    # For larger n, use local search
    return _local_search(word, n, start_time, deadline)


def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
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


def _get_missing_perms(word, n):
    """Return set of permutations not present in word."""
    found = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            found.add(window)
    all_perms = set(permutations(range(1, n + 1)))
    return all_perms - found


def _count_perms(word, n):
    """Count how many distinct permutations appear in word."""
    found = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            found.add(window)
    return len(found)


def _local_search(word, n, start_time, deadline):
    """
    Local search to shorten the superpermutation.
    
    Operations:
    1. Delete a symbol (if the resulting word still contains all perms)
    2. Swap adjacent symbols (if still valid)
    3. Move a symbol (if still valid)
    """
    word = list(word)
    current_len = len(word)
    best_word = list(word)
    best_len = current_len
    
    # Precompute which permutations are present
    def check_valid(w):
        """Check if w contains all n! permutations."""
        found = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                found.add(window)
            if len(found) == n ** n:  # early exit if we have all
                break
        return len(found) == n ** n or len(found) >= n ** n
    
    # Actually, n! is the number of permutations, not n^n
    target_count = _factorial(n)
    
    def check_valid(w):
        """Check if w contains all n! permutations."""
        found = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                found.add(window)
            if len(found) == target_count:
                return True
        return len(found) == target_count
    
    # Try deletions first - most impactful
    random.seed(42)
    indices = list(range(len(word)))
    random.shuffle(indices)
    
    for idx in indices:
        if time.time() > deadline - 10:
            break
        
        # Try deleting position idx
        new_word = word[:idx] + word[idx + 1:]
        if len(new_word) < n:
            continue
        
        if check_valid(new_word):
            word = new_word
            best_word = list(word)
            best_len = len(word)
            # Restart search with shorter word
            indices = list(range(len(word)))
            random.shuffle(indices)
            continue
    
    # Try swaps
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        if len(word) < 2:
            break
        
        i = random.randint(0, len(word) - 2)
        new_word = list(word)
        new_word[i], new_word[i + 1] = new_word[i + 1], new_word[i]
        
        if check_valid(new_word):
            word = new_word
            best_word = list(word)
            best_len = len(word)
    
    # Try insertions of existing patterns (more complex)
    # For now, return best found
    return "".join(map(str, best_word))


def _factorial(n):
    """Compute n!"""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _optimize_small_n(word, n, start_time, deadline):
    """Optimize for small n where we can be exhaustive."""
    word = list(word)
    best_word = list(word)
    best_len = len(word)
    
    target_count = _factorial(n)
    
    def check_valid(w):
        found = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                found.add(window)
            if len(found) == target_count:
                return True
        return len(found) == target_count
    
    # Exhaustive deletion search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(word)):
            new_word = word[:i] + word[i + 1:]
            if check_valid(new_word):
                word = new_word
                best_word = list(word)
                best_len = len(word)
                improved = True
                break
    
    return "".join(map(str, best_word))


# EVOLVE-BLOCK-END
