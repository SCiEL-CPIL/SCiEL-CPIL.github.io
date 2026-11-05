# EVOLVE-BLOCK-START
import itertools
import numpy as np
from collections import defaultdict, deque

def construct(q, n, radius):
    """
    Construct a ternary covering code of length n and radius radius.
    Uses a heuristic greedy set-cover approach with optional local improvement.
    For the specific case q=3, n=6, radius=1, it aims to find a code close to 73.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: use the direct sum construction if applicable,
        # otherwise a simple greedy.
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # Extend to length n by repeating the last coordinate or just padding with 0s?
            # The seed used product for the tail. Let's stick to a simple greedy for general cases.
            pass
        
        # General Greedy Set Cover
        universe = set(itertools.product(range(q), repeat=n))
        candidates = list(universe)
        covered = set()
        code = []
        
        # Pre-compute neighborhoods for efficiency? Too expensive for large n.
        # For n=6, q=3, radius=1, size is 729. Neighborhood size is 1 + 6*2 = 13.
        # Total elements to cover: 729.
        
        # For the specific hard case, we use a specialized heuristic.
        if q == 3 and n == 6 and radius == 1:
            return _construct_ternary_6_1()
            
        # Generic Greedy
        remaining = set(universe)
        while remaining:
            best_cand = None
            max_gain = -1
            # Sample candidates if too many
            cand_list = list(remaining)
            if len(cand_list) > 50:
                import random
                cand_list = random.sample(cand_list, 50)
            
            for c in cand_list:
                # Calculate gain
                gain = 0
                for i in range(n):
                    for val in range(q):
                        if val == c[i]: continue
                        new_word = list(c)
                        new_word[i] = val
                        if tuple(new_word) in remaining:
                            gain += 1
                if c in remaining:
                    gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_cand = c
            
            if best_cand is None:
                break
                
            code.append(list(best_cand))
            # Remove covered
            to_remove = {tuple(best_cand)}
            for i in range(n):
                for val in range(q):
                    if val == best_cand[i]: continue
                    new_word = list(best_cand)
                    new_word[i] = val
                    to_remove.add(tuple(new_word))
            remaining -= to_remove
            
        return code

def _construct_ternary_6_1():
    """
    Specialized heuristic for q=3, n=6, radius=1.
    We use a greedy approach with a priority based on the number of uncovered words
    that are "hard" to cover (i.e., have fewer neighbors already in the code).
    """
    n = 6
    q = 3
    radius = 1
    
    # Represent words as integers 0..728 for speed
    all_words = set(range(3**n))
    covered = set()
    code = []
    
    # Precompute neighbors for each word
    # neighbor[word] = set of words within distance 1
    # This is 729 * 13 integers. Very fast.
    neighbors = {}
    for w in all_words:
        digits = [(w // (3**i)) % 3 for i in range(n)]
        neigh = {w}
        for i in range(n):
            for d in range(3):
                if d == digits[i]: continue
                new_digit = d
                diff = (new_digit - digits[i]) * (3**i)
                neigh.add(w + diff)
        neighbors[w] = neigh
    
    # Greedy selection
    # To improve on basic greedy, we can use a "least constraining" or "most urgent" heuristic.
    # Standard greedy: pick codeword that covers most uncovered words.
    # Improved: pick codeword that covers the most uncovered words, with tie-breaking
    # favoring words that cover "isolated" uncovered words.
    
    # For speed, we will just do a few iterations of greedy with randomization/restart.
    best_code = None
    best_size = float('inf')
    
    import random
    random.seed(42)
    
    for attempt in range(20):
        cur_covered = set()
        cur_code = []
        remaining = set(all_words)
        
        # Pre-compute for each candidate c, the set of uncovered words it covers.
        # We maintain a count for each uncovered word of how many candidates cover it?
        # That's expensive. Let's stick to simple greedy but with a good tie-breaker.
        
        while remaining:
            # Find candidate with max coverage of remaining
            best_c = None
            max_gain = -1
            
            # Optimization: Only consider candidates that are in 'remaining' or close to it?
            # Actually, any word can be a codeword.
            # To speed up, we can sample or iterate all. 729 iterations * 729 checks * 13 ops ~ 7M ops.
            # In Python, this might be slow. 20 attempts * 73 codewords * 729 candidates * 13 checks = ~14M operations per attempt?
            # 20 attempts * 73 * 729 * 13 = 13.9M. This is likely too slow in pure Python within 120s if not optimized.
            
            # Let's optimize the inner loop.
            # Instead of checking all 729 candidates, we can pick a candidate from the remaining set
            # that has the highest "potential".
            
            # Alternative: Use the property that we just need to cover.
            # Let's try a faster greedy:
            # Pick a random uncovered word w.
            # Choose a candidate c from neighbors[w] that covers the most uncovered words.
            
            w = random.choice(list(remaining))
            candidates = neighbors[w]
            
            best_c = None
            max_gain = -1
            
            for c in candidates:
                # Count how many words in neighbors[c] are in remaining
                gain = 0
                for u in neighbors[c]:
                    if u in remaining:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_c = c
            
            if best_c is None:
                break
                
            cur_code.append(best_c)
            # Update remaining
            for u in neighbors[best_c]:
                if u in remaining:
                    remaining.remove(u)
                    cur_covered.add(u)
                    
        if len(cur_code) < best_size:
            best_size = len(cur_code)
            best_code = cur_code.copy()
            
            # Early stop if we hit the known optimum or lower bound
            if best_size <= 71:
                break
            if best_size == 72:
                # Try a few more to see if we can get 71
                continue
                
    # Convert back to list of lists
    result = []
    for w in best_code:
        digits = [(w // (3**i)) % 3 for i in range(n)]
        result.append(digits)
        
    return result

# EVOLVE-BLOCK-END
