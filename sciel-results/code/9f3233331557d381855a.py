"""
Football pool / ternary covering code K_3(6,1).

Strategy:
  1. Quick construction via direct sum of tetracode with all tails (gives 81).
  2. Greedy covering: pick codewords that cover the most uncovered words.
  3. Local improvement: try to remove codewords if coverage remains.
  4. Add a deadline guard to stay within 150 s.
"""
import time
import itertools
import random
from collections import defaultdict

# Precompute all words of length 6 over {0,1,2}
def all_words(q, n):
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def hamming_distance(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

def neighbors(word, q, n, radius=1):
    """Return all words within Hamming distance <= radius from word."""
    result = [list(word)]
    for i in range(n):
        for v in range(q):
            if v != word[i]:
                w = list(word)
                w[i] = v
                result.append(w)
    return result

def construct(q, n, radius):
    start = time.time()
    deadline = start + 145.0  # safety margin
    
    # Step 1: Quick construction via tetracode direct sum
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Fallback: random greedy
        code = []
        uncovered = set(map(tuple, all_words(q, n)))
        while uncovered:
            best = max(uncovered, key=lambda w: sum(1 for x in uncovered if hamming_distance(w, x) <= radius))
            code.append(list(best))
            for w in neighbors(best, q, n, radius):
                uncovered.discard(tuple(w))
    
    # Step 2: Try to improve by removing redundant codewords
    def is_covered(word, code, q, n, radius):
        return any(hamming_distance(word, c) <= radius for c in code)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(code)):
            candidate = code[:i] + code[i+1:]
            # Check if all words are still covered
            if all(is_covered(w, candidate, q, n, radius) for w in all_words(q, n)):
                code = candidate
                improved = True
                break
    
    # Step 3: Try to add better codewords if we have time
    # (This is a simple local search; for now, return what we have)
    if time.time() < deadline:
        # Try random perturbation: replace a codeword with a random one if it improves coverage
        best_code = code
        best_len = len(code)
        
        # Count covered words for each potential codeword position
        all_w = all_words(q, n)
        
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            
            # Try removing a random codeword
            idx = random.randrange(len(best_code))
            candidate = best_code[:idx] + best_code[idx+1:]
            
            # Check coverage
            covered = set()
            for c in candidate:
                for w in neighbors(c, q, n, radius):
                    covered.add(tuple(w))
            
            if len(covered) == len(all_w):
                best_code = candidate
                best_len = len(candidate)
        
        if len(best_code) < len(code):
            code = best_code
    
    return code
