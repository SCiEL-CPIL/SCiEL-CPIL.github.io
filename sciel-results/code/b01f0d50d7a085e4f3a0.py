"""Improved ternary covering code K_3(7,1) via greedy construction with pruning."""
import time
import random
import itertools
import numpy as np

# Deadline guard: leave a safety margin before the 150s limit
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds

def _hamming_dist(a, b):
    return sum(x != y for x, y in zip(a, b))

def construct(q, n, radius):
    """Construct a covering code of radius `radius` for {0..q-1}^n.
    
    For the specific case q=3, n=7, radius=1, uses a greedy algorithm with
    structured initialization. Falls back to direct product construction for
    other parameters.
    """
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = 150.0

    # Handle the general case with direct product if not the specific task
    if not (q == 3 and n == 7 and radius == 1):
        # Direct sum construction: use tetracode for first 4, full for rest
        if n >= 4:
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] 
                         for a in range(q) for b in range(q)]
            return [h + list(tail) for h in tetracode 
                    for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # For very short codes, just return all words
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    # Specific case: q=3, n=7, radius=1
    # Total space: 3^7 = 2187 words
    # Each codeword covers: 1 + 7*2 = 15 words
    # Lower bound: ceil(2187/15) = 146
    # Best known: 186

    # Strategy: 
    # 1. Start with a good structured seed (tetracode x Z3^3)
    # 2. Greedily try to replace/remove codewords while maintaining coverage
    # 3. Use local search to improve

    # Initial construction: tetracode (9 words) x all of Z3^3 (27 words) = 243 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode 
                    for tail in itertools.product(range(3), repeat=3)]
    
    code = [list(cw) for cw in initial_code]
    
    # Helper: check if a word is covered by the current code
    def is_covered(word, code_list):
        for cw in code_list:
            if _hamming_dist(word, cw) <= radius:
                return True
        return False
    
    # Helper: get all words within radius of a codeword
    def neighbors(cw, q, n, radius):
        nbrs = set()
        # All words at distance 0
        nbrs.add(tuple(cw))
        # All words at distance 1
        for pos in range(n):
            for val in range(q):
                if val != cw[pos]:
                    new_word = cw.copy()
                    new_word[pos] = val
                    nbrs.add(tuple(new_word))
        return nbrs
    
    # Precompute all words in the space
    all_words = list(itertools.product(range(3), repeat=7))
    
    # Try to remove codewords that are redundant
    removed_any = True
    while removed_any:
        removed_any = False
        for i in range(len(code)):
            # Check if removing code[i] still leaves all its neighbors covered
            test_code = code[:i] + code[i+1:]
            if len(test_code) == 0:
                removed_any = True
                code = test_code
                break
            all_ok = True
            for word in neighbors(code[i], 3, 7, 1):
                if not is_covered(word, test_code):
                    all_ok = False
                    break
            if all_ok:
                code = test_code
                removed_any = True
                break
    
    # Now try local search: for each codeword, try moving it to a better position
    random.seed(42)
    improvement_found = True
    while improvement_found:
        improvement_found = False
        for i in range(len(code)):
            # Try moving code[i] to a random position
            for _ in range(20):
                new_cw = code[i].copy()
                pos = random.randint(0, 6)
                val = random.randint(0, 2)
                if val == new_cw[pos]:
                    continue
                new_cw[pos] = val
                if new_cw in [tuple(c) for c in code]:
                    continue
                
                test_code = code.copy()
                test_code[i] = new_cw
                
                # Check if all words previously covered by code[i] are still covered
                ok = True
                for word in neighbors(code[i], 3, 7, 1):
                    if not is_covered(word, test_code):
                        ok = False
                        break
                if ok:
                    # Check if the new position covers some additional words
                    # (not strictly needed, but we accept if valid)
                    code = test_code
                    improvement_found = True
                    break
            if improvement_found:
                break
    
    # Try adding-then-removing: temporarily add a new codeword, then remove an old one
    for _ in range(50):
        if time.time() > _DEADLINE - 10:
            break
        
        # Pick a random new position
        new_cw = [random.randint(0, 2) for _ in range(7)]
        if tuple(new_cw) in [tuple(c) for c in code]:
            continue
        
        test_code = code + [new_cw]
        
        # Now try to remove any codeword
        removed = False
        for i in range(len(test_code)):
            if i == len(test_code) - 1:  # don't remove the new one
                continue
            temp_code = test_code[:i] + test_code[i+1:]
            if len(temp_code) == 0:
                continue
            all_ok = True
            for word in neighbors(test_code[i], 3, 7, 1):
                if not is_covered(word, temp_code):
                    all_ok = False
                    break
            if all_ok:
                code = temp_code
                removed = True
                break
        if removed:
            break
    
    # Verify coverage
    code_tuples = [tuple(c) for c in code]
    for word in all_words:
        covered = False
        for cw in code_tuples:
            if _hamming_dist(word, cw) <= 1:
                covered = True
                break
        if not covered:
            # Fallback: return the initial construction
            return initial_code
    
    return code
