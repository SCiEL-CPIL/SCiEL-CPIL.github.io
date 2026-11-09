"""Optimized ternary covering code constructor using numpy for fast pruning."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def _hamming_dist(a, b):
    return sum(x != y for x, y in zip(a, b))


def _is_covered(word, codewords, q, n, radius):
    # Check if word is within radius of any codeword
    for cw in codewords:
        if _hamming_dist(word, cw) <= radius:
            return True
    return False


def _deltas_within_radius(q, n, radius):
    """Generate all delta vectors with Hamming weight <= radius."""
    deltas = []
    # Weight 0
    deltas.append([0] * n)
    
    # Weight 1 to radius
    for weight in range(1, radius + 1):
        for positions in itertools.combinations(range(n), weight):
            for values in itertools.product(range(1, q), repeat=weight):
                delta = [0] * n
                for pos, val in zip(positions, values):
                    delta[pos] = val
                deltas.append(delta)
    return deltas


def _greedy_cover(q, n, radius, time_limit=110.0):
    """
    Greedy set cover approach with numpy-accelerated pruning.
    1. Start with a good initial code (direct sum construction).
    2. Iteratively try to remove codewords if coverage is maintained.
    3. Also try adding new codewords that cover the most uncovered words,
       then prune again.
    """
    start_time = time.time()
    
    # Initial code: direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    if n <= 4:
        # For small n, use a simpler approach
        if n == 1:
            return [[i] for i in range(q)]
        elif n == 2:
            return [list(p) for p in itertools.product(range(q), repeat=2)]
        elif n == 3:
            # Perfect code for q=3, n=3, r=1
            return [[a, b, (a + b) % 3] for a in range(3) for b in range(3)]
        else:
            # n=4: tetracode
            return tetracode[:9] if len(tetracode) >= 9 else tetracode
    
    # Build initial code using direct sum
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    # Randomize the order to help greedy pruning
    random.shuffle(initial_code)
    
    codewords = list(initial_code)
    
    # Precompute deltas
    deltas = _deltas_within_radius(q, n, radius)
    deltas_arr = np.array(deltas, dtype=np.int8)
    
    def _can_remove(idx, codewords_list, deltas_arr, q, n, radius):
        """Check if codeword at idx can be removed while maintaining coverage."""
        cw = np.array(codewords_list[idx], dtype=np.int8)
        other_cws = [np.array(cw_j, dtype=np.int8) for j, cw_j in enumerate(codewords_list) if j != idx]
        
        if not other_cws:
            return False
        
        other_cws_arr = np.array(other_cws, dtype=np.int8)
        
        # For each word in the neighborhood of cw, check if it's covered by others
        for delta in deltas_arr:
            word = (cw + delta) % q
            # Compute Hamming distances to all other codewords
            diffs = np.abs(other_cws_arr - word)
            dists = np.sum(diffs > 0, axis=1)
            if np.min(dists) > radius:
                return False
        return True
    
    # Phase 1: Aggressive pruning with numpy
    max_prune_iters = 200
    for iteration in range(max_prune_iters):
        if time.time() - start_time > time_limit:
            break
        
        if len(codewords) < 2:
            break
        
        improved = False
        indices = list(range(len(codewords)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            
            if idx >= len(codewords):
                continue
            
            if _can_remove(idx, codewords, deltas_arr, q, n, radius):
                del codewords[idx]
                improved = True
                break  # Restart since indices changed
        
        if not improved:
            break
    
    # Phase 2: Try to add codewords that cover uncovered words, then prune
    max_add_iters = 50
    for iteration in range(max_add_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Sample to find uncovered words
        sample_size = min(2000, 3**n // 5)
        uncovered = []
        for _ in range(sample_size):
            word = [random.randint(0, q - 1) for _ in range(n)]
            if not _is_covered(word, codewords, q, n, radius):
                uncovered.append(word)
        
        if not uncovered:
            break
        
        # Generate candidates from uncovered words' neighborhoods
        existing_set = set(tuple(c) for c in codewords)
        candidates = set()
        for uw in uncovered[:30]:
            for delta in deltas:
                cand = tuple((uw[i] + delta[i]) % q for i in range(n))
                if cand not in existing_set:
                    candidates.add(cand)
        
        if not candidates:
            break
        
        # Score candidates by how many uncovered words they cover
        best_candidate = None
        best_score = 0
        for cand in list(candidates)[:100]:
            score = sum(1 for uw in uncovered if _hamming_dist(cand, uw) <= radius)
            if score > best_score:
                best_score = score
                best_candidate = list(cand)
        
        if best_candidate is not None and best_score > 0:
            codewords.append(best_candidate)
        
        # Prune again after adding
        for _ in range(30):
            if time.time() - start_time > time_limit:
                break
            improved = False
            indices = list(range(len(codewords)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() - start_time > time_limit:
                    break
                if idx >= len(codewords):
                    continue
                if _can_remove(idx, codewords, deltas_arr, q, n, radius):
                    del codewords[idx]
                    improved = True
                    break
            if not improved:
                break
    
    return codewords


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q
    with covering radius radius.
    """
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for the 120s limit
    
    # Handle edge cases
    if radius >= n:
        return [[0] * n]
    
    if n == 0:
        return [[]]
    
    if q == 1:
        return [[0] * n]
    
    # For small cases, use exact or near-exact methods
    if n <= 4:
        return _greedy_cover(q, n, radius, time_limit)
    
    # Use greedy optimization with numpy acceleration
    result = _greedy_cover(q, n, radius, time_limit)
    
    # Fallback: if something went wrong, use direct sum
    if time.time() - start_time > time_limit or len(result) == 0:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return result
    
    return result
# EVOLVE-BLOCK-END
