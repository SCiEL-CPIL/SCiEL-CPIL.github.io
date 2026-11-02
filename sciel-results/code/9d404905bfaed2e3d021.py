# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _tetracode(q, n, radius):
    """Direct sum of ternary tetracode: 9 * 3**(n-4) words."""
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    if n == 4:
        return tetracode
    tails = list(itertools.product(range(q), repeat=n - 4))
    return [h + list(tail) for h in tetracode for tail in tails]

def _covered_mask(codewords, q, n, radius):
    """Return a boolean array of size q^n indicating which words are covered."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    
    # Convert codewords to integer indices
    cw_indices = []
    for cw in codewords:
        idx = 0
        for i, v in enumerate(cw):
            idx = idx * q + v
        cw_indices.append(idx)
    
    # Precompute all words as integer indices and their digit representations
    # For efficiency, we'll iterate over each codeword and mark its ball
    
    # To speed up, precompute the list of all words as tuples of digits
    # But that's q^n = 2187 for our case, which is small.
    
    # For each codeword, generate all words within Hamming distance radius
    # and mark them as covered.
    
    # We need to convert integer index to digits and back, or work directly with digits.
    # Let's work with digits. Precompute all words as lists of digits.
    all_words = []
    for idx in range(total):
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        all_words.append(tuple(digits))
    
    # Map from tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # For each codeword, find all words within radius
    for cw in codewords:
        cw_tuple = tuple(cw)
        # Generate all words within Hamming distance radius from cw
        # Use combinations of positions to change
        for d in range(radius + 1):
            for positions in itertools.combinations(range(n), d):
                # For each combination of positions, try all possible changes
                if d == 0:
                    idx = word_to_idx[cw_tuple]
                    covered[idx] = True
                else:
                    # For the chosen positions, try all combinations of new values (different from original)
                    # Actually, for Hamming distance exactly d, we change exactly d positions.
                    # But "within radius" means distance <= radius, so we need distance 0, 1, ..., radius.
                    # For distance exactly d, we change d positions to different values.
                    # But it's easier to just generate all words within distance <= radius.
                    pass
    
    # This approach is getting complex. Let me use a simpler direct approach for small n.
    # Since q^n = 2187 is small, we can just check each word against each codeword.
    
    # Actually, let me just do a direct check: for each word in the space, check if it's within radius of any codeword.
    # This is O(q^n * |C| * n) which for 2187 * 200 * 7 is about 3 million operations, which is fine.
    
    covered = np.zeros(total, dtype=bool)
    for idx, word in enumerate(all_words):
        for cw in codewords:
            dist = sum(1 for i in range(n) if word[i] != cw[i])
            if dist <= radius:
                covered[idx] = True
                break
    
    return covered, all_words, word_to_idx

def _greedy_covering(q, n, radius, initial_codewords=None, time_limit=100):
    """Greedy set cover to build a covering code."""
    start_time = time.time()
    total = q ** n
    
    # Precompute all words
    all_words = []
    for idx in range(total):
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        all_words.append(tuple(digits))
    
    # Precompute distance from each word to each other word? That's 2187*2187 = ~4.8M entries.
    # Each entry is a small integer, so memory is fine.
    # But computing it might take a moment. Let's precompute.
    
    # Actually, let's just compute distances on the fly but cache them.
    # Or precompute a distance matrix.
    
    # Precompute distance matrix: dist[i][j] = Hamming distance between all_words[i] and all_words[j]
    dist_matrix = np.zeros((total, total), dtype=np.uint8)
    for i in range(total):
        for j in range(i + 1, total):
            d = sum(1 for k in range(n) if all_words[i][k] != all_words[j][k])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    
    # Initialize covered set
    covered = np.zeros(total, dtype=bool)
    
    # Start with initial codewords if provided
    codewords = []
    if initial_codewords:
        for cw in initial_codewords:
            idx = None
            for i, w in enumerate(all_words):
                if tuple(cw) == w:
                    idx = i
                    break
            if idx is not None:
                codewords.append(idx)
                # Mark all words within radius as covered
                for j in range(total):
                    if dist_matrix[idx][j] <= radius:
                        covered[j] = True
    
    # Greedy: repeatedly pick the word that covers the most uncovered words
    while True:
        if time.time() - start_time > time_limit:
            break
        
        uncovered = np.where(~covered)[0]
        if len(uncovered) == 0:
            break
        
        # For each candidate word, count how many uncovered words it covers
        best_idx = -1
        best_count = -1
        
        for idx in range(total):
            if idx in codewords:
                continue
            # Count uncovered words within radius of idx
            count = 0
            for j in uncovered:
                if dist_matrix[idx][j] <= radius:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1 or best_count == 0:
            break
        
        codewords.append(best_idx)
        for j in range(total):
            if dist_matrix[best_idx][j] <= radius:
                covered[j] = True
    
    # Convert back to list of lists
    result = [list(all_words[i]) for i in codewords]
    return result

def _remove_redundant(codewords, q, n, radius, time_limit=50):
    """Try to remove redundant codewords."""
    start_time = time.time()
    total = q ** n
    
    all_words = []
    for idx in range(total):
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        all_words.append(tuple(digits))
    
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Convert codewords to indices
    cw_indices = []
    for cw in codewords:
        idx = word_to_idx[tuple(cw)]
        cw_indices.append(idx)
    
    # Precompute distance matrix
    dist_matrix = np.zeros((total, total), dtype=np.uint8)
    for i in range(total):
        for j in range(i + 1, total):
            d = sum(1 for k in range(n) if all_words[i][k] != all_words[j][k])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    
    # Compute coverage count for each word
    coverage_count = np.zeros(total, dtype=np.int32)
    for idx in cw_indices:
        for j in range(total):
            if dist_matrix[idx][j] <= radius:
                coverage_count[j] += 1
    
    # Try to remove codewords
    removed = set()
    for idx in cw_indices:
        if time.time() - start_time > time_limit:
            break
        if idx in removed:
            continue
        
        # Check if removing idx still covers everything
        # A word j is still covered if coverage_count[j] > 1 or if j is covered by another codeword
        # More precisely: j is covered by idx if dist_matrix[idx][j] <= radius
        # If we remove idx, j is still covered if there exists another idx' in cw_indices (not removed) with dist_matrix[idx'][j] <= radius
        
        # For efficiency, check if there's any word j that is ONLY covered by idx
        # i.e., coverage_count[j] == 1 and dist_matrix[idx][j] <= radius
        can_remove = True
        for j in range(total):
            if dist_matrix[idx][j] <= radius and coverage_count[j] == 1:
                can_remove = False
                break
        
        if can_remove:
            removed.add(idx)
            # Update coverage counts
            for j in range(total):
                if dist_matrix[idx][j] <= radius:
                    coverage_count[j] -= 1
    
    # Build result
    result = [list(all_words[i]) for i in cw_indices if i not in removed]
    return result

def construct(q, n, radius):
    """Construct a covering code using greedy method with tetracode initialization."""
    start_time = time.time()
    total_time_limit = 120
    
    # Get initial codewords from tetracode construction
    initial = _tetracode(q, n, radius)
    
    # Use greedy covering to improve
    if time.time() - start_time < total_time_limit - 60:
        improved = _greedy_covering(q, n, radius, initial_codewords=initial, time_limit=total_time_limit - 70)
    else:
        improved = initial
    
    # Remove redundant codewords
    if time.time() - start_time < total_time_limit - 10:
        final = _remove_redundant(improved, q, n, radius, time_limit=total_time_limit - 15)
    else:
        final = improved
    
    # Ensure all codewords are distinct
    seen = set()
    unique = []
    for cw in final:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique.append(cw)
    
    return unique
# EVOLVE-BLOCK-END
