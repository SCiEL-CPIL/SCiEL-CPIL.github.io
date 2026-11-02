"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _all_neighbors_fast(q, n, radius):
    """Precompute all words within Hamming distance <= radius of each codeword.
    Returns a list of sets (one per codeword) containing all covered word indices."""
    # This is used conceptually; we'll use a different approach for speed
    pass


def _greedy_cover(q, n, radius, max_time=10.0):
    """Fast greedy heuristic to generate a feasible covering code."""
    start_time = time.time()
    total_words = q ** n
    
    # Represent each word as an integer index
    all_words = list(range(total_words))
    uncovered = set(all_words)
    
    codewords = []
    
    # Precompute neighbors for efficiency
    # For radius 1, a word covers itself + n*(q-1) neighbors
    # We'll use a different strategy: greedy set cover
    
    # For efficiency with large spaces, use a different approach
    # Start with a structured initial set
    
    # For q=3, n=8, radius=1: use tetracode-based construction as initial
    if q == 3 and n == 8 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n-4)]
    else:
        # General greedy: add words that cover the most uncovered points
        while uncovered and (time.time() - start_time) < max_time:
            best_word = None
            best_count = -1
            
            # Sample some candidate words to find good ones
            candidates = random.sample(uncovered, min(100, len(uncovered)))
            
            for w in candidates:
                # Count how many uncovered words this covers
                word = [int(w // (q ** (n - i - 1)) % q) for i in range(n)]
                covered = _count_covered(word, q, n, radius, uncovered)
                if covered > best_count:
                    best_count = covered
                    best_word = word
            
            if best_word is not None:
                codewords.append(best_word)
                # Remove covered words from uncovered
                covered_set = _get_covered_set(best_word, q, n, radius)
                uncovered -= covered_set
    
    return codewords


def _word_to_idx(word, q, n):
    """Convert a word (list of ints) to an integer index."""
    idx = 0
    for i, v in enumerate(word):
        idx = idx * q + v
    return idx


def _idx_to_word(idx, q, n):
    """Convert an integer index to a word (list of ints)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _get_neighbors(word, q, n, radius):
    """Get all words within Hamming distance <= radius from word."""
    neighbors = set()
    # Generate all words at distance 0, 1, ..., radius
    for r in range(radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(q), repeat=r):
                new_word = list(word)
                for pos, val in zip(positions, values):
                    new_word[pos] = val
                neighbors.add(tuple(new_word))
    return neighbors


def _count_covered(word, q, n, radius, uncovered_set):
    """Count how many words in uncovered_set are within radius of word."""
    neighbors = _get_neighbors(word, q, n, radius)
    count = 0
    for nb in neighbors:
        idx = 0
        for v in nb:
            idx = idx * q + v
        if idx in uncovered_set:
            count += 1
    return count


def _get_covered_set(word, q, n, radius):
    """Get the set of indices covered by word."""
    neighbors = _get_neighbors(word, q, n, radius)
    covered = set()
    for nb in neighbors:
        idx = 0
        for v in nb:
            idx = idx * q + v
        covered.add(idx)
    return covered


def _local_search(codewords, q, n, radius, max_time=100.0):
    """Local search: try swapping codewords to reduce the total size while maintaining coverage."""
    start_time = time.time()
    deadline = start_time + max_time
    
    current_codewords = [list(w) for w in codewords]
    current_size = len(current_codewords)
    
    # Build coverage map: for each word index, which codewords cover it
    total_words = q ** n
    coverage_count = np.zeros(total_words, dtype=np.int32)
    
    # Initialize coverage
    for cw in current_codewords:
        neighbors = _get_neighbors(cw, q, n, radius)
        for nb in neighbors:
            idx = 0
            for v in nb:
                idx = idx * q + v
            coverage_count[idx] += 1
    
    # Check if all covered
    if np.any(coverage_count == 0):
        return current_codewords
    
    best_codewords = current_codewords[:]
    best_size = current_size
    
    # Local search: try removing a codeword if it doesn't break coverage
    # Then try adding a better one
    random.shuffle(current_codewords)
    
    for i in range(len(current_codewords)):
        if time.time() > deadline - 5.0:
            break
        
        # Try removing codeword i
        cw = current_codewords[i]
        neighbors = _get_neighbors(cw, q, n, radius)
        
        # Check if any word would become uncovered
        would_break = False
        for nb in neighbors:
            idx = 0
            for v in nb:
                idx = idx * q + v
            if coverage_count[idx] == 1:
                would_break = True
                break
        
        if not would_break:
            # Safe to remove
            for nb in neighbors:
                idx = 0
                for v in nb:
                    idx = idx * q + v
                coverage_count[idx] -= 1
            current_codewords.pop(i)
            best_codewords = current_codewords[:]
            best_size = len(current_codewords)
            i = max(0, i - 1)  # Re-check same position
        else:
            # Try swapping: find a codeword that covers the uniquely covered words
            # and potentially allows removal of another codeword
            pass
    
    # Try adding codewords that allow removal of others
    for _ in range(1000):
        if time.time() > deadline - 5.0:
            break
        
        # Randomly pick a codeword to potentially replace
        i = random.randint(0, len(current_codewords) - 1)
        old_cw = current_codewords[i]
        old_neighbors = _get_neighbors(old_cw, q, n, radius)
        
        # Find words uniquely covered by old_cw
        unique_words = []
        for nb in old_neighbors:
            idx = 0
            for v in nb:
                idx = idx * q + v
            if coverage_count[idx] == 1:
                unique_words.append(idx)
        
        if not unique_words:
            # Can remove
            for nb in old_neighbors:
                idx = 0
                for v in nb:
                    idx = idx * q + v
                coverage_count[idx] -= 1
            current_codewords.pop(i)
            best_codewords = current_codewords[:]
            best_size = len(current_codewords)
            continue
        
        # Try to find a new codeword that covers all unique_words
        # Sample some candidates
        found = False
        for _ in range(50):
            # Pick a random word near one of the unique words
            base_idx = random.choice(unique_words)
            base_word = _idx_to_word(base_idx, q, n)
            
            # Perturb
            new_word = base_word[:]
            for _ in range(random.randint(0, 2)):
                pos = random.randint(0, n - 1)
                new_word[pos] = random.randint(0, q - 1)
            
            new_neighbors = _get_neighbors(new_word, q, n, radius)
            
            # Check if covers all unique_words
            covers_all = True
            for uw in unique_words:
                found_uw = False
                for nb in new_neighbors:
                    idx = 0
                    for v in nb:
                        idx = idx * q + v
                    if idx == uw:
                        found_uw = True
                        break
                if not found_uw:
                    covers_all = False
                    break
            
            if covers_all:
                # Check if this new codeword allows removal of another codeword
                # For now, just replace if it doesn't increase size
                current_codewords[i] = new_word
                # Update coverage
                for nb in old_neighbors:
                    idx = 0
                    for v in nb:
                        idx = idx * q + v
                    coverage_count[idx] -= 1
                for nb in new_neighbors:
                    idx = 0
                    for v in nb:
                        idx = idx * q + v
                    coverage_count[idx] += 1
                found = True
                break
        
        if not found:
            pass
    
    return best_codewords


def construct(q, n, radius):
    start_time = time.time()
    total_budget = 120.0
    
    # Generate initial feasible covering
    codewords = _greedy_cover(q, n, radius, max_time=15.0)
    
    # Verify it's actually feasible
    total_words = q ** n
    covered = set()
    for cw in codewords:
        neighbors = _get_neighbors(cw, q, n, radius)
        for nb in neighbors:
            idx = 0
            for v in nb:
                idx = idx * q + v
            covered.add(idx)
    
    if len(covered) < total_words:
        # Not fully covered, need to add more
        while len(covered) < total_words and (time.time() - start_time) < total_budget - 10.0:
            uncovered = set(range(total_words)) - covered
            if not uncovered:
                break
            # Pick a random uncovered word and add a codeword that covers it
            target = random.choice(list(uncovered))
            target_word = _idx_to_word(target, q, n)
            codewords.append(target_word)
            neighbors = _get_neighbors(target_word, q, n, radius)
            for nb in neighbors:
                idx = 0
                for v in nb:
                    idx = idx * q + v
                covered.add(idx)
    
    # Local search to optimize
    remaining_time = total_budget - 15.0 - (time.time() - start_time)
    if remaining_time > 5.0:
        codewords = _local_search(codewords, q, n, radius, max_time=remaining_time - 5.0)
    
    return codewords
# EVOLVE-BLOCK-END
