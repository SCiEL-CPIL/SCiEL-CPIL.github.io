"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    if q != 3:
        # Fallback for non-ternary: trivial covering (all words)
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    if n <= 4:
        # For small n, just return all words or a perfect code if exists
        if n == 4:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        else:
            return [list(t) for t in itertools.product(range(3), repeat=n)]
    
    # For n=7, radius=1, q=3: use greedy set cover with bitmask optimization
    total_words = 3 ** n  # 2187
    
    # Precompute all words as integers 0..2186
    # word i has digits (i // 3^k) % 3 for k=0..n-1
    # neighbors: for each word, compute all words within Hamming distance <= radius
    
    # Precompute powers of 3
    pow3 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow3[i] = pow3[i-1] * 3
    
    def word_to_int(word):
        """Convert a tuple of digits to integer representation."""
        result = 0
        for i, d in enumerate(word):
            result += d * pow3[i]
        return result
    
    def int_to_word(i):
        """Convert integer to tuple of digits."""
        word = []
        for k in range(n):
            word.append((i // pow3[k]) % 3)
        return tuple(word)
    
    # Precompute neighbors for all words (within Hamming distance <= radius)
    # For radius=1, neighbors are the word itself and words with one digit changed
    neighbors = []
    for i in range(total_words):
        w = int_to_word(i)
        nbrs = [i]
        for pos in range(n):
            for new_digit in range(3):
                if new_digit == w[pos]:
                    continue
                w2 = list(w)
                w2[pos] = new_digit
                nbrs.append(word_to_int(tuple(w2)))
        neighbors.append(set(nbrs))
    
    # Greedy set cover: start with empty code, repeatedly add the codeword
    # that covers the most uncovered words
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    
    # Precompute coverage for each potential codeword
    # For each word c, compute which words it covers
    # We'll use a different approach: for each candidate, count how many uncovered words it covers
    
    while np.any(uncovered):
        if time.time() > deadline - 10.0:
            break
        
        # Find the codeword that covers the most uncovered words
        best_c = -1
        best_count = -1
        
        # Sample candidates for efficiency (if too many, sample)
        all_candidates = list(range(total_words))
        if len(all_candidates) > 500:
            random.seed(int(time.time() * 1000) % (2**31))
            candidates = random.sample(all_candidates, 500)
        else:
            candidates = all_candidates
        
        for c in candidates:
            if not uncovered[c]:
                continue
            # Count uncovered words covered by c
            count = 0
            for nbr in neighbors[c]:
                if uncovered[nbr]:
                    count += 1
            if count > best_count:
                best_count = count
                best_c = c
        
        if best_c == -1 or best_count <= 0:
            break
        
        code.append(best_c)
        # Update uncovered
        for nbr in neighbors[best_c]:
            uncovered[nbr] = False
    
    # If we ran out of time, try to complete with a fallback
    if np.any(uncovered) and time.time() < deadline - 5.0:
        # Add remaining uncovered words as codewords
        for i in range(total_words):
            if uncovered[i]:
                code.append(i)
                for nbr in neighbors[i]:
                    uncovered[nbr] = False
            if time.time() > deadline - 5.0:
                break
    
    # Now try to remove redundant codewords
    if time.time() < deadline - 10.0:
        # Coverage counting approach
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            for nbr in neighbors[c]:
                coverage_count[nbr] += 1
        
        code_set = set(code)
        code_list = list(code_set)
        random.seed(42)
        random.shuffle(code_list)
        
        for c in code_list:
            if time.time() > deadline - 5.0:
                break
            if c not in code_set:
                continue
            
            # Check if removing c leaves any word uncovered
            can_remove = True
            for nbr in neighbors[c]:
                if coverage_count[nbr] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                code_set.remove(c)
                for nbr in neighbors[c]:
                    coverage_count[nbr] -= 1
    
    # Convert back to list of words
    result = [list(int_to_word(c)) for c in sorted(code_set)]
    
    # Verify the result is valid
    covered = np.zeros(total_words, dtype=bool)
    for c in code_set:
        for nbr in neighbors[c]:
            covered[nbr] = True
    
    if not np.all(covered):
        # Fallback to base code if verification fails
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return base_code
    
    return result
# EVOLVE-BLOCK-END
