"""
Football pool problem: ternary covering code K_3(6,1)

Strategy:
1. Use a fast numpy-based coverage representation (bitmask per coordinate/value).
2. Start from the 81-word direct product (tetracode x 3x3).
3. Aggressive local search:
   a. Remove any codeword that is redundant (its ball is covered by others).
   b. Try to replace pairs of codewords with a single one (2->1 reduction).
   c. Randomized restarts with different initial orderings to escape local minima.
4. Use the known 73-word construction as a target; try to beat it.

The key insight: with 729 words and each codeword covering 1 + 2*n = 13 words,
we need at least 729/13 ≈ 56 codewords. The best known is 73.

For speed, we use numpy to compute coverage masks efficiently.
"""

import time
import random
import itertools
import numpy as np

DEADLINE = 130.0  # Safety margin before 150s kill


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet of size q with covering radius radius."""
    start_time = time.time()
    
    # For the specific case q=3, n=6, radius=1, use optimized approach
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1(start_time)
    
    # General fallback: greedy + local search
    return _general_construct(q, n, radius, start_time)


def _time_left(start_time):
    return DEADLINE - (time.time() - start_time)


def _construct_3_6_1(start_time):
    """Optimized construction for q=3, n=6, radius=1."""
    q, n, radius = 3, 6, 1
    total = q ** n  # 729
    
    # Precompute: for each word index and each coordinate, the indices of words
    # that differ only in that coordinate (including the word itself)
    # ball_mask[i] = set of indices covered by word i (as a boolean array)
    
    # Represent each word as an integer in [0, 729)
    # Word i has digits d_0...d_5 where i = d_0*3^5 + d_1*3^4 + ... + d_5
    
    # Precompute power table
    powers = [1]
    for i in range(n):
        powers.append(powers[-1] * q)
    # powers[j] = q^j
    
    # For each coordinate position pos and value v, compute the mask of words
    # that have value v at position pos
    # Actually, for coverage: word w covers all words that differ from w in at most 1 position.
    # So ball(w) = {w} union {words differing from w in exactly position i for each i}
    
    # Precompute: for each word index, the list of indices in its ball
    # This is 729 * 13 = 9477 entries, manageable
    
    # Alternative: use numpy for fast coverage checks
    # coverage_count[i] = number of codewords whose ball contains word i
    
    # Let's precompute the ball membership:
    # ball_of_word[i] = list of word indices covered by word i
    ball_of_word = []
    for i in range(total):
        digits = [(i // powers[j]) % q for j in range(n)]
        ball = [i]
        for pos in range(n):
            for v in range(q):
                if v != digits[pos]:
                    new_idx = i + (v - digits[pos]) * powers[n - 1 - pos]
                    ball.append(new_idx)
        ball_of_word.append(ball)
    
    # Convert to numpy arrays for fast operations
    # coverage_count: how many codewords cover each word
    # For removal test: a codeword c is redundant if for every word w in ball(c),
    # there exists another codeword c' != c such that w is in ball(c')
    # i.e., coverage_count[w] >= 2 for all w in ball(c)
    
    # Start with the 81-word tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=2):
            initial_code.append(h + list(tail))
    
    # Convert to integer indices
    code_idx = [sum(d * powers[n - 1 - i] for i, d in enumerate(word)) for word in initial_code]
    
    # Try multiple strategies with random restarts
    best_code = None
    best_size = len(code_idx)
    
    def _check_validity(codeword_indices):
        """Check if the set of codeword indices covers all words."""
        coverage = np.zeros(total, dtype=np.int32)
        for ci in codeword_indices:
            for w in ball_of_word[ci]:
                coverage[w] += 1
        return np.all(coverage >= 1)
    
    def _try_remove(codeword_indices, remove_idx):
        """Try to remove codeword at remove_idx. Returns new list if valid, else None."""
        new_indices = [ci for j, ci in enumerate(codeword_indices) if j != remove_idx]
        if _check_validity(new_indices):
            return new_indices
        return None
    
    def _local_search_reduce(codeword_indices, timeout=10.0):
        """Local search: repeatedly try to remove redundant codewords."""
        t_start = time.time()
        indices = list(codeword_indices)
        
        while _time_left(start_time) > 5.0:
            if time.time() - t_start > timeout:
                break
            
            improved = False
            # Try removing each codeword
            for i in range(len(indices)):
                new_indices = _try_remove(indices, i)
                if new_indices is not None:
                    indices = new_indices
                    improved = True
                    break
            
            if not improved:
                break
        
        return indices
    
    def _pair_replacement_search(codeword_indices, timeout=10.0):
        """Try to find a single word that can replace two codewords."""
        t_start = time.time()
        indices = list(codeword_indices)
        
        while _time_left(start_time) > 5.0:
            if time.time() - t_start > timeout:
                break
            
            improved = False
            # Try replacing pairs with a single word
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    if time.time() - t_start > timeout:
                        break
                    # Remove both i and j, try all 729 words as replacement
                    remaining = [indices[k] for k in range(len(indices)) if k != i and k != j]
                    # Check if remaining covers everything except possibly some words
                    # that were only covered by indices[i] or indices[j]
                    
                    # Quick check: compute coverage with remaining
                    coverage = np.zeros(total, dtype=np.int32)
                    for ci in remaining:
                        for w in ball_of_word[ci]:
                            coverage[w] += 1
                    
                    # Find words not covered
                    uncovered = np.where(coverage == 0)[0]
                    if len(uncovered) == 0:
                        # Both were redundant - just remove one
                        indices = remaining
                        improved = True
                        break
                    elif len(uncovered) <= 13:
                        # A single new word might cover all uncovered
                        for candidate in range(total):
                            if all(w in ball_of_word[candidate] for w in uncovered):
                                indices = remaining + [candidate]
                                improved = True
                                break
                        if improved:
                            break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return indices
    
    # Strategy 1: Local search reduction from 81
    code = list(code_idx)
    code = _local_search_reduce(code, timeout=15.0)
    if len(code) < best_size:
        best_size = len(code)
        best_code = list(code)
    
    # Strategy 2: Pair replacement
    if _time_left(start_time) > 10.0:
        code = list(best_code) if best_code else list(code_idx)
        code = _pair_replacement_search(code, timeout=20.0)
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
    
    # Strategy 3: Randomized restarts with shuffled initial code
    if _time_left(start_time) > 15.0:
        for restart in range(5):
            if _time_left(start_time) < 10.0:
                break
            
            shuffled = list(code_idx)
            random.shuffle(shuffled)
            code = _local_search_reduce(shuffled, timeout=8.0)
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
            
            # Also try pair replacement on the result
            if _time_left(start_time) > 10.0:
                code2 = _pair_replacement_search(code, timeout=8.0)
                if len(code2) < best_size:
                    best_size = len(code2)
                    best_code = list(code2)
    
    # Strategy 4: Greedy construction from scratch (different ordering)
    if _time_left(start_time) > 10.0:
        code = _greedy_construct_3_6_1(ball_of_word, total)
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
        
        code = _local_search_reduce(code, timeout=8.0)
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
    
    # Strategy 5: Try to find a 73-word code using targeted search
    if _time_left(start_time) > 10.0 and best_size > 73:
        code = _targeted_search_73(ball_of_word, total, start_time)
        if code is not None and len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
    
    # Final: convert best back to words
    if best_code is None:
        best_code = code_idx
    
    # Verify
    if not _check_validity(best_code):
        # Fallback to initial
        best_code = code_idx
    
    # Convert to list of words
    result = []
    for ci in best_code:
        digits = [(ci // powers[j]) % q for j in range(n)]
        result.append(digits)
    
    return result


def _greedy_construct_3_6_1(ball_of_word, total):
    """Greedy set-cover construction for q=3, n=6, radius=1."""
    covered = np.zeros(total, dtype=bool)
    uncovered_count = total
    code = []
    
    while uncovered_count > 0:
        best_word = None
        best_gain = -1
        
        for i in range(total):
            if covered[i]:
                continue
            gain = 0
            for w in ball_of_word[i]:
                if not covered[w]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_word = i
        
        if best_word is None:
            break
        
        code.append(best_word)
        for w in ball_of_word[best_word]:
            if not covered[w]:
                covered[w] = True
                uncovered_count -= 1
    
    return code


def _targeted_search_73(ball_of_word, total, start_time):
    """Try to find a 73-word code via local search starting from 74-word candidates."""
    # This is a heuristic: start from a random 74-word subset of a known good code
    # and try to reduce to 73
    
    # For now, use a simpler approach: take the best code we have and try harder
    # to reduce it by 1
    
    # Generate a random code of size 74
    all_indices = list(range(total))
    
    for attempt in range(20):
        if _time_left(start_time) < 5.0:
            break
        
        # Random 74-word code
        random.shuffle(all_indices)
        code = all_indices[:74]
        
        # Quick validity check
        coverage = np.zeros(total, dtype=np.int32)
        for ci in code:
            for w in ball_of_word[ci]:
                coverage[w] += 1
        
        if not np.all(coverage >= 1):
            continue
        
        # Try to reduce to 73
        improved = True
        while improved and _time_left(start_time) > 5.0:
            improved = False
            for i in range(len(code)):
                new_code = [code[j] for j in range(len(code)) if j != i]
                coverage = np.zeros(total, dtype=np.int32)
                for ci in new_code:
                    for w in ball_of_word[ci]:
                        coverage[w] += 1
                if np.all(coverage >= 1):
                    code = new_code
                    improved = True
                    break
        
        if len(code) <= 73:
            return code
    
    return None


def _general_construct(q, n, radius, start_time):
    """General fallback for other parameters."""
    total = q ** n
    powers = [1]
    for i in range(n):
        powers.append(powers[-1] * q)
    
    # Precompute balls
    ball_of_word = []
    for i in range(total):
        digits = [(i // powers[j]) % q for j in range(n)]
        ball = [i]
        for pos in range(n):
            for v in range(q):
                if v != digits[pos]:
                    new_idx = i + (v - digits[pos]) * powers[n - 1 - pos]
                    ball.append(new_idx)
        ball_of_word.append(ball)
    
    # Greedy construction
    covered = np.zeros(total, dtype=bool)
    uncovered_count = total
    code = []
    
    while uncovered_count > 0:
        best_word = None
        best_gain = -1
        
        for i in range(total):
            if covered[i]:
                continue
            gain = 0
            for w in ball_of_word[i]:
                if not covered[w]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_word = i
        
        if best_word is None:
            break
        
        code.append(best_word)
        for w in ball_of_word[best_word]:
            if not covered[w]:
                covered[w] = True
                uncovered_count -= 1
    
    # Local search reduction
    def _check_validity(indices):
        coverage = np.zeros(total, dtype=np.int32)
        for ci in indices:
            for w in ball_of_word[ci]:
                coverage[w] += 1
        return np.all(coverage >= 1)
    
    def _try_remove(indices, idx):
        new_indices = [ci for j, ci in enumerate(indices) if j != idx]
        if _check_validity(new_indices):
            return new_indices
        return None
    
    improved = True
    while improved and _time_left(start_time) > 5.0:
        improved = False
        for i in range(len(code)):
            new_code = _try_remove(code, i)
            if new_code is not None:
                code = new_code
                improved = True
                break
    
    # Convert to words
    result = []
    for ci in code:
        digits = [(ci // powers[j]) % q for j in range(n)]
        result.append(digits)
    
    return result
