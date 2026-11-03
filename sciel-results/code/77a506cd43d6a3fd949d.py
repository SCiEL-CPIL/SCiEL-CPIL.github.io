"""Ternary covering code K_3(6,1) via greedy set cover with local search polish."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _all_words(q, n):
    """Generate all q^n words as a numpy array of shape (q^n, n)."""
    total = q ** n
    words = np.zeros((total, n), dtype=np.int8)
    for i in range(total):
        v = i
        for j in range(n):
            words[i, j] = v % q
            v //= q
    return words


def _ball_indices(word, q, n, radius):
    """Return list of indices of all words within Hamming distance <= radius of word."""
    # For radius 1, this is 1 + n*(q-1) words
    indices = [0]  # placeholder, we'll compute properly
    # Actually, let's generate the ball and map to indices
    # For efficiency with small radius, enumerate directly
    ball = []
    # The word itself
    ball.append(tuple(word))
    # One-position changes
    for pos in range(n):
        for val in range(q):
            if val != word[pos]:
                w = list(word)
                w[pos] = val
                ball.append(tuple(w))
    return ball


def _word_to_index(word, q, n):
    """Convert a word (tuple or list) to its index in the lexicographic enumeration."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # For the specific case q=3, n=6, radius=1, use specialized approach
    # General fallback: greedy set cover
    
    total_words = q ** n
    if total_words > 100000:
        # Too large for exact enumeration, use heuristic
        # Just return a basic construction
        return _basic_construction(q, n, radius)
    
    # Generate all words
    all_words = _all_words(q, n)
    
    # Precompute ball membership: for each word, which codewords cover it
    # We'll use a greedy approach:
    # 1. Start with all words uncovered
    # 2. Repeatedly pick the word that covers the most uncovered words
    # 3. Add it to the code, mark its ball as covered
    
    # Precompute balls for all words (for small spaces)
    # For q=3, n=6: 729 words, each ball has 1 + 6*2 = 13 words
    # Total ball entries: 729 * 13 = 9477, very manageable
    
    balls = {}  # word_index -> set of word indices covered
    for i in range(total_words):
        word = all_words[i]
        covered = set()
        covered.add(i)
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    w = list(word)
                    w[pos] = val
                    j = _word_to_index(w, q, n)
                    covered.add(j)
        balls[i] = covered
    
    # Greedy set cover
    uncovered = set(range(total_words))
    code = []
    
    while uncovered:
        # Pick the word whose ball covers the most uncovered words
        best_word = None
        best_count = -1
        for i in range(total_words):
            count = len(balls[i] & uncovered)
            if count > best_count:
                best_count = count
                best_word = i
            # Early termination if we can't beat current best
            # (remaining uncovered in this ball can't exceed best_count)
        
        if best_word is None:
            break
            
        code.append(all_words[best_word].tolist())
        uncovered -= balls[best_word]
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Local search polish: try to remove redundant codewords and swap
    # First, check if we have a valid cover
    if not uncovered:  # We have a valid cover
        code = _local_search_polish(code, all_words, balls, total_words, q, n, deadline)
    
    # If we didn't finish the greedy, add more to cover remaining
    if uncovered:
        # Continue greedy from where we left off
        while uncovered and time.time() < deadline - 1.0:
            best_word = None
            best_count = -1
            for i in range(total_words):
                count = len(balls[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = i
            if best_word is None:
                break
            code.append(all_words[best_word].tolist())
            uncovered -= balls[best_word]
    
    # Final validation and deduplication
    seen = set()
    unique_code = []
    for cw in code:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_code.append(cw)
    
    return unique_code


def _basic_construction(q, n, radius):
    """Fallback basic construction for larger spaces."""
    # Use the tetracode-based construction from the seed
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Very basic: just return a few words
        return [[0] * n]


def _local_search_polish(code, all_words, balls, total_words, q, n, deadline):
    """Try to reduce the code size by removing redundant words and swapping."""
    if not code:
        return code
    
    # Convert to set of tuples for fast lookup
    code_set = set(tuple(cw) for cw in code)
    code_indices = set(_word_to_index(cw, q, n) for cw in code)
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Try removing each codeword
        for i, cw in enumerate(list(code)):
            cw_tuple = tuple(cw)
            cw_idx = _word_to_index(cw_tuple, q, n)
            
            # Temporarily remove
            code_set.remove(cw_tuple)
            code_indices.remove(cw_idx)
            
            # Check if all words are still covered
            # A word w is covered if it's within distance 1 of some remaining codeword
            # Check: is every word in the ball of cw covered by other codewords?
            # Actually, we need to check if removing cw leaves any word uncovered.
            # The words that were ONLY covered by cw are those in ball(cw) not in union of balls of other codewords.
            
            # For efficiency, check if there exists a word w such that:
            #   w is in ball(cw) and w is not in ball(cw') for any other cw' in code
            # This is equivalent to: the intersection of ball(cw) with (complement of union of other balls) is non-empty
            
            # Simpler: just verify coverage of the whole space (expensive but correct)
            # For small spaces like 729, this is fast enough
            
            covered = set()
            for other_cw in code_set:
                covered |= balls[_word_to_index(other_cw, q, n)]
            
            if covered == set(range(total_words)):
                # Safe to remove
                code.pop(i)
                improved = True
                break
            else:
                # Restore
                code_set.add(cw_tuple)
                code_indices.add(cw_idx)
    
    # Try swapping: replace one codeword with another that covers more
    for _ in range(20):  # Limit iterations
        if time.time() > deadline - 5.0:
            break
        found_swap = False
        for i, cw in enumerate(list(code)):
            cw_idx = _word_to_index(cw, q, n)
            # Try replacing with a word that covers more of the "hard" regions
            # For simplicity, try random replacements
            for _ in range(50):
                new_idx = random.randint(0, total_words - 1)
                if new_idx == cw_idx:
                    continue
                new_cw = all_words[new_idx].tolist()
                new_cw_tuple = tuple(new_cw)
                
                # Temporarily replace
                old_cw_tuple = tuple(cw)
                code_set.remove(old_cw_tuple)
                code_set.add(new_cw_tuple)
                
                # Check coverage
                covered = set()
                for other_cw in code_set:
                    covered |= balls[_word_to_index(other_cw, q, n)]
                
                if covered == set(range(total_words)):
                    # Valid swap, keep it
                    code[i] = new_cw
                    found_swap = True
                    break
                else:
                    # Restore
                    code_set.remove(new_cw_tuple)
                    code_set.add(old_cw_tuple)
            if found_swap:
                break
        if not found_swap:
            break
    
    return code


# EVOLVE-BLOCK-END
