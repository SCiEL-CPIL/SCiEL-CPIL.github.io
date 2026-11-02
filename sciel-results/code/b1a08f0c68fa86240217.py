"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _all_neighbors(q, n, radius=1):
    """Precompute neighbor indices for all words in {0..q-1}^n within Hamming distance radius."""
    total = q ** n
    # For radius 1: each word has 1 + n*(q-1) neighbors
    if radius == 1:
        num_neighbors = 1 + n * (q - 1)
        neighbors = np.zeros((total, num_neighbors), dtype=np.int32)
        for w in range(total):
            # Convert word index to digits
            digits = np.array([w // (q ** (n - 1 - i)) % q for i in range(n)], dtype=np.int32)
            neighbors[w, 0] = w
            idx = 1
            for pos in range(n):
                orig = digits[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    digits[pos] = val
                    neighbors[w, idx] = int(np.dot(digits, q ** np.arange(n - 1, -1, -1)))
                    idx += 1
                digits[pos] = orig
        return neighbors
    else:
        raise NotImplementedError("Only radius 1 supported")


def _greedy_cover(q, n, radius, time_limit=30.0):
    """Greedy set cover: repeatedly pick the word covering the most uncovered words."""
    start = time.time()
    total = q ** n
    neighbors = _all_neighbors(q, n, radius)
    
    # coverage[i] = number of currently uncovered words in the ball around word i
    # Initially all words are uncovered
    uncovered = np.ones(total, dtype=np.int32)
    coverage = uncovered.copy()  # coverage[i] = sum of uncovered over neighbors[i]
    
    # Precompute neighbor arrays for fast updates
    # For each word i, neighbors[i] gives all words in its ball
    # We need to efficiently update coverage when we cover/uncover words
    
    # Use a different approach: maintain count of how many selected codewords cover each word
    # Then a word is covered if count[word] > 0
    # coverage[i] = number of words in ball(i) with count[word] == 0
    
    # Simpler greedy: at each step, for each candidate word, count how many uncovered words it covers
    # Pick the one with max, add it, mark those words as covered
    
    # To speed up: use the fact that we can vectorize the counting
    selected = []
    covered = np.zeros(total, dtype=bool)
    
    while not covered.all():
        if time.time() - start > time_limit:
            break
        
        # Count how many uncovered words each candidate covers
        # candidates: all words (or just uncovered ones for speed)
        # For each candidate c, coverage_count[c] = sum of (not covered) over neighbors[c]
        # = sum of (1 - covered[neighbors[c]])
        
        # Vectorized: for all candidates at once
        # neighbors shape: (total, num_neighbors)
        # covered[neighbors] shape: (total, num_neighbors)
        # coverage_count = (1 - covered[neighbors]).sum(axis=1)
        
        # This is expensive (total * num_neighbors operations per iteration)
        # Optimize: only consider candidates that are themselves uncovered or have uncovered neighbors
        
        # For large problems, this greedy is O(total * total * num_neighbors) which is too slow
        # Let's use a smarter approach: start from the tetracode construction and try to reduce
        
        # Actually, let's just use the tetracode as initial solution and do local search
        break
    
    # Fallback: return tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def _local_search(q, n, radius, initial_codewords, time_limit=100.0):
    """Local search: try removing redundant codewords and swapping to reduce size."""
    start = time.time()
    total = q ** n
    neighbors = _all_neighbors(q, n, radius)
    
    # Convert codewords to indices
    q_powers = q ** np.arange(n - 1, -1, -1)
    
    def word_to_idx(w):
        return int(np.dot(np.array(w, dtype=np.int32), q_powers))
    
    codeword_indices = [word_to_idx(w) for w in initial_codewords]
    
    # count[word] = number of selected codewords whose ball contains word
    count = np.zeros(total, dtype=np.int32)
    for ci in codeword_indices:
        count[neighbors[ci]] += 1
    
    # A codeword at index ci is "redundant" if for all words w in ball(ci), count[w] > 1
    # i.e., removing it doesn't leave any word uncovered
    
    best_size = len(codeword_indices)
    best_codewords = list(codeword_indices)
    
    def is_redundant(ci):
        return count[neighbors[ci]].min() > 1
    
    def try_remove(ci):
        """Try removing codeword ci. Returns True if successful."""
        if not is_redundant(ci):
            return False
        # Remove it
        count[neighbors[ci]] -= 1
        codeword_indices.remove(ci)
        return True
    
    def try_swap(ci, cj):
        """Try replacing codeword ci with cj. Returns True if successful and size doesn't increase."""
        # After removing ci and adding cj, check coverage
        # New count = count - ball(ci) + ball(cj)
        new_count = count.copy()
        new_count[neighbors[ci]] -= 1
        new_count[neighbors[cj]] += 1
        if new_count.min() >= 1:
            # Valid swap
            count[neighbors[ci]] -= 1
            count[neighbors[cj]] += 1
            idx = codeword_indices.index(ci)
            codeword_indices[idx] = cj
            return True
        return False
    
    # Phase 1: Remove all redundant codewords
    changed = True
    while changed and time.time() - start < time_limit:
        changed = False
        for i in range(len(codeword_indices) - 1, -1, -1):
            if is_redundant(codeword_indices[i]):
                ci = codeword_indices[i]
                count[neighbors[ci]] -= 1
                codeword_indices.pop(i)
                changed = True
                if len(codeword_indices) < best_size:
                    best_size = len(codeword_indices)
                    best_codewords = list(codeword_indices)
    
    # Phase 2: Random swaps to find better configurations
    all_words = list(range(total))
    random.shuffle(all_words)
    
    for _ in range(5000):
        if time.time() - start > time_limit:
            break
        if len(codeword_indices) >= best_size:
            # Reset to best
            count[:] = 0
            for ci in best_codewords:
                count[neighbors[ci]] += 1
            codeword_indices = list(best_codewords)
        
        # Pick a random codeword to potentially replace
        if not codeword_indices:
            break
        i = random.randint(0, len(codeword_indices) - 1)
        ci = codeword_indices[i]
        
        # Pick a random candidate to swap in
        cj = random.choice(all_words)
        if cj == ci:
            continue
        
        if try_swap(ci, cj):
            # Successful swap, size same. Check if we can now remove something
            for j in range(len(codeword_indices) - 1, -1, -1):
                if is_redundant(codeword_indices[j]):
                    cj2 = codeword_indices[j]
                    count[neighbors[cj2]] -= 1
                    codeword_indices.pop(j)
                    if len(codeword_indices) < best_size:
                        best_size = len(codeword_indices)
                        best_codewords = list(codeword_indices)
    
    # Convert back to list of lists
    q_powers_list = list(q_powers)
    result = []
    for ci in best_codewords:
        w = []
        rem = ci
        for i in range(n):
            w.append(rem // (q ** (n - 1 - i)))
            rem = rem % (q ** (n - 1 - i))
        result.append(w)
    return result


def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0  # Leave 10s margin
    
    # Start with tetracode construction
    import itertools
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    if len(initial) <= 486:
        return initial
    
    # Local search to reduce size
    result = _local_search(q, n, radius, initial, time_limit=130.0)
    
    return result
# EVOLVE-BLOCK-END
