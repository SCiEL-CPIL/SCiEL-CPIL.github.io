"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

START_TIME = time.time()
DEADLINE = 55.0  # safety margin before 90s kill

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

# Bitmask-based validity checking
# Max sum for n elements from [1,256]: n*256, for n=60 that's 15360
# We use 16384 bits to be safe
MAX_SUM = 16384
NUM_WORDS = (MAX_SUM + 63) // 64  # 256 uint64 words

def check_validity_bitmask(A):
    """Fast validity check using numpy uint64 bitmasks."""
    n = len(A)
    if n == 0:
        return True
    
    # sums_by_k[k] = array of uint64 words representing which sums are achievable
    sums_by_k = [np.zeros(NUM_WORDS, dtype=np.uint64) for _ in range(n + 1)]
    sums_by_k[0][0] = 1  # sum 0 is achievable with 0 elements
    
    for a in A:
        # Update in reverse to avoid using a twice
        for k in range(n - 1, -1, -1):
            if np.any(sums_by_k[k]):
                shifted = np.zeros(NUM_WORDS, dtype=np.uint64)
                word_shift = a // 64
                bit_shift = a % 64
                for w in range(NUM_WORDS - word_shift):
                    val = sums_by_k[k][w]
                    if val:
                        shifted[w + word_shift] |= (val << bit_shift)
                        if bit_shift > 0 and w + word_shift + 1 < NUM_WORDS:
                            shifted[w + word_shift + 1] |= (val >> (64 - bit_shift))
                sums_by_k[k + 1] |= shifted
    
    # Check pairwise disjointness
    seen = np.zeros(NUM_WORDS, dtype=np.uint64)
    for k in range(n + 1):
        if np.any(sums_by_k[k] & seen):
            return False
        seen |= sums_by_k[k]
    return True

def greedy_construct():
    """Greedy: try to add elements in ascending order."""
    A = []
    used = set()
    
    for a in range(1, 257):
        if a in used:
            continue
        if is_done():
            break
        A.append(a)
        used.add(a)
        if not check_validity_bitmask(A):
            A.pop()
            used.discard(a)
    
    return A

def greedy_construct_descending():
    """Greedy with descending order."""
    A = []
    used = set()
    
    for a in range(256, 0, -1):
        if a in used:
            continue
        if is_done():
            break
        A.append(a)
        used.add(a)
        if not check_validity_bitmask(A):
            A.pop()
            used.discard(a)
    
    return A

def backtracking_search(A, used, depth, max_depth):
    """Backtracking search to find larger valid sets."""
    if is_done():
        return A
    
    if depth >= max_depth:
        return A
    
    best = list(A)
    
    # Try adding each unused element
    candidates = [a for a in range(1, 257) if a not in used]
    
    for a in candidates:
        if is_done():
            break
        new_A = A + [a]
        new_used = used | {a}
        if check_validity_bitmask(new_A):
            result = backtracking_search(new_A, new_used, depth + 1, max_depth)
            if len(result) > len(best):
                best = result
    
    return best

def local_search_improve(A):
    """Try to improve A by swapping elements and adding new ones."""
    A = list(A)
    used = set(A)
    improved = True
    
    while improved and not is_done():
        improved = False
        
        # First try to add a new element
        for add_val in range(1, 257):
            if add_val in used:
                continue
            A.append(add_val)
            used.add(add_val)
            if check_validity_bitmask(A):
                improved = True
                break
            else:
                A.pop()
                used.discard(add_val)
        
        if is_done():
            break
        
        # Then try swapping
        for i in range(len(A)):
            if is_done():
                break
            for new_val in range(1, 257):
                if new_val in used:
                    continue
                old_val = A[i]
                A[i] = new_val
                used.discard(old_val)
                used.add(new_val)
                
                if check_validity_bitmask(A):
                    # Try to add another element after swap
                    for add_val in range(1, 257):
                        if add_val in used:
                            continue
                        A.append(add_val)
                        used.add(add_val)
                        if check_validity_bitmask(A):
                            improved = True
                            break
                        else:
                            A.pop()
                            used.discard(add_val)
                    break
                else:
                    A[i] = old_val
                    used.discard(new_val)
                    used.add(old_val)
    
    return A

def random_search(num_trials=200):
    """Try random subsets and see which are valid."""
    best = []
    
    for trial in range(num_trials):
        if is_done():
            break
        
        # Try various sizes
        size = random.randint(8, 35)
        A = random.sample(range(1, 257), size)
        
        if check_validity_bitmask(A):
            if len(A) > len(best):
                best = A
    
    return best

def construct():
    """Main entry point: try multiple strategies and return the best valid set."""
    best = []
    
    # Strategy 1: Greedy ascending
    A1 = greedy_construct()
    if len(A1) > len(best):
        best = A1
    
    if is_done():
        return best
    
    # Strategy 2: Greedy descending
    A2 = greedy_construct_descending()
    if len(A2) > len(best):
        best = A2
    
    if is_done():
        return best
    
    # Strategy 3: Local search improvement on best so far
    A3 = local_search_improve(best)
    if len(A3) > len(best):
        best = A3
    
    if is_done():
        return best
    
    # Strategy 4: Random search
    A4 = random_search(100)
    if len(A4) > len(best):
        best = A4
    
    if is_done():
        return best
    
    # Strategy 5: Local search on the best found
    A5 = local_search_improve(best)
    if len(A5) > len(best):
        best = A5
    
    # Ensure we return a valid result
    if not check_validity_bitmask(best):
        # Fallback to a known valid small set
        best = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Ensure all are Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
