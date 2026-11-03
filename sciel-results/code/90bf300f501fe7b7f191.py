"""Rewritten solver for K_q(n, radius) covering code; entry point: construct(q, n, radius)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _elapsed(t0):
    return time.time() - t0


def _time_left(t0, deadline):
    return max(0.0, deadline - _elapsed(t0))


def _idx_to_word(idx, q, n):
    """Convert integer index to word tuple of length n over alphabet 0..q-1."""
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = idx % q
        idx //= q
    return tuple(w)


def _word_to_idx(word, q, n):
    """Convert word (tuple/list) to integer index."""
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _neighbors_indices(word_tuple, q, n, radius):
    """Yield integer indices of all words within Hamming distance <= radius of word_tuple."""
    # We'll use a recursive/generator approach for radius 1 (most common case)
    # For radius 1: word itself + all single-position changes
    yield _word_to_idx(word_tuple, q, n)
    for pos in range(n):
        orig = word_tuple[pos]
        for v in range(q):
            if v != orig:
                # Build new word
                new_list = list(word_tuple)
                new_list[pos] = v
                yield _word_to_idx(tuple(new_list), q, n)


def _all_neighbors_indices(word_tuple, q, n, radius):
    """Return list of all indices within Hamming distance <= radius of word_tuple."""
    if radius == 0:
        return [_word_to_idx(word_tuple, q, n)]
    if radius == 1:
        return list(_neighbors_indices(word_tuple, q, n, radius))
    # General case (less common)
    import itertools
    result = set()
    for r in range(radius + 1):
        for positions in itertools.combinations(range(n), r):
            if r == 0:
                result.add(_word_to_idx(word_tuple, q, n))
            else:
                for values in itertools.product(range(q), repeat=r):
                    new_list = list(word_tuple)
                    for pos, val in zip(positions, values):
                        new_list[pos] = val
                    result.add(_word_to_idx(tuple(new_list), q, n))
    return list(result)


def _tetracode_initial(q, n, radius):
    """
    Generate initial covering code using tetracode product structure for q=3, n=8, radius=1.
    The ternary tetracode has 9 words of length 4 covering all 81 words within distance 1.
    Product structure: C = Tetracode x {0,1,2}^4 gives 9*81 = 729 words.
    But we can do better with a more clever construction.
    
    Actually, the known best is 486. Let's use a construction based on:
    - Split 8 positions into two groups of 4
    - Use tetracode for each half with a product-like structure
    
    A better approach: use the (8, 486, 1) code structure.
    One known construction: take all words (a,b,c,d,e,f,g,h) where
    a+d+e+h ≡ b+c+f+g (mod 3) -- this gives 3^8/3 = 2187, too many.
    
    Let's use a greedy approach with smart initialization.
    """
    if q != 3 or n != 8 or radius != 1:
        return []
    
    # Use the tetracode product: 9 * 81 = 729 as starting point
    # Tetracode: all (a, b, a+b, a+2b) for a,b in {0,1,2}
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append((a, b, (a + b) % 3, (a + 2 * b) % 3))
    
    # Product: tetracode x all 4-tuples
    codewords = []
    for tc in tetracode:
        for tail in np.ndindex(3, 3, 3, 3):
            word = list(tc) + list(tail)
            codewords.append(tuple(word))
    
    return codewords


def _greedy_cover(q, n, radius, t0, deadline):
    """
    Greedy set cover: repeatedly add the codeword that covers the most uncovered words.
    Returns list of codeword tuples.
    """
    total_words = q ** n
    
    # uncovered as a set of indices
    uncovered = set(range(total_words))
    codewords = []
    
    # Precompute all words as tuples for fast neighbor lookup
    # For q=3, n=8: 6561 words, each covers 17 -> manageable
    
    # Try structured initial code first if applicable
    initial = _tetracode_initial(q, n, radius)
    if initial:
        # Verify coverage and add any missing
        for cw in initial:
            nbs = _all_neighbors_indices(cw, q, n, radius)
            for nb in nbs:
                if nb in uncovered:
                    uncovered.discard(nb)
        codewords.extend(initial)
        
        # If not fully covered, continue greedy
        if uncovered:
            _greedy_fill(uncovered, codewords, q, n, radius, t0, deadline)
        return codewords
    
    # General greedy
    _greedy_fill(uncovered, codewords, q, n, radius, t0, deadline)
    return codewords


def _greedy_fill(uncovered, codewords, q, n, radius, t0, deadline):
    """Fill remaining uncovered words greedily."""
    # For speed, sample candidates from uncovered set
    while uncovered and _time_left(t0, deadline) > 2.0:
        # Sample candidates
        sample_size = min(200, len(uncovered))
        candidates = random.sample(list(uncovered), sample_size)
        
        best_word = None
        best_count = -1
        
        for cand_idx in candidates:
            cand_word = _idx_to_word(cand_idx, q, n)
            nbs = _all_neighbors_indices(cand_word, q, n, radius)
            count = sum(1 for nb in nbs if nb in uncovered)
            if count > best_count:
                best_count = count
                best_word = (cand_idx, cand_word, nbs)
        
        if best_word is None or best_count <= 0:
            break
            
        cand_idx, cand_word, nbs = best_word
        codewords.append(cand_word)
        for nb in nbs:
            uncovered.discard(nb)
    
    # If still uncovered (shouldn't happen for small spaces), brute force add
    while uncovered and _time_left(t0, deadline) > 1.0:
        for idx in list(uncovered)[:100]:
            word = _idx_to_word(idx, q, n)
            nbs = _all_neighbors_indices(word, q, n, radius)
            codewords.append(word)
            for nb in nbs:
                uncovered.discard(nb)


def _build_coverage(codewords, q, n, radius):
    """
    Build coverage array: coverage[i] = number of codewords covering word i.
    Also build reverse mapping: for each word, list of codeword indices covering it.
    """
    total_words = q ** n
    coverage = np.zeros(total_words, dtype=np.int32)
    # For each codeword, precompute its neighbors
    cw_neighbors = []
    for cw in codewords:
        nbs = _all_neighbors_indices(cw, q, n, radius)
        cw_neighbors.append(nbs)
        for nb in nbs:
            coverage[nb] += 1
    return coverage, cw_neighbors


def _local_search(codewords, q, n, radius, t0, deadline):
    """
    Local search: try to remove codewords that don't break coverage,
    then try swaps to reduce size further.
    """
    if not codewords:
        return codewords
    
    total_words = q ** n
    current = list(codewords)
    best = list(current)
    best_size = len(current)
    
    # Build coverage
    coverage, cw_neighbors = _build_coverage(current, q, n, radius)
    
    if np.any(coverage == 0):
        return current  # Not fully covered, shouldn't happen
    
    # Phase 1: Remove redundant codewords
    removed_any = True
    while removed_any and _time_left(t0, deadline) > 5.0:
        removed_any = False
        # Shuffle for diversity
        random.shuffle(current)
        for i in range(len(current)):
            if _time_left(t0, deadline) < 5.0:
                break
            # Check if codeword i is redundant
            nbs = cw_neighbors[i]
            redundant = True
            for nb in nbs:
                if coverage[nb] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                for nb in nbs:
                    coverage[nb] -= 1
                current.pop(i)
                cw_neighbors.pop(i)
                removed_any = True
                if len(current) < best_size:
                    best_size = len(current)
                    best = list(current)
                i = max(0, i - 1)
    
    # Phase 2: Try swaps - replace a codeword with another that allows further removals
    for _ in range(500):
        if _time_left(t0, deadline) < 10.0:
            break
        
        if not current:
            break
            
        i = random.randint(0, len(current) - 1)
        old_cw = current[i]
        old_nbs = cw_neighbors[i]
        
        # Find words uniquely covered by old_cw
        unique_words = [nb for nb in old_nbs if coverage[nb] == 1]
        
        if not unique_words:
            # Can remove
            for nb in old_nbs:
                coverage[nb] -= 1
            current.pop(i)
            cw_neighbors.pop(i)
            if len(current) < best_size:
                best_size = len(current)
                best = list(current)
            continue
        
        # Try to find a replacement that covers all unique words
        found = False
        for _ in range(30):
            # Pick a random word near one of the unique words
            base_idx = random.choice(unique_words)
            base_word = _idx_to_word(base_idx, q, n)
            
            # Perturb by 0-2 positions
            new_word = list(base_word)
            for _ in range(random.randint(0, 2)):
                pos = random.randint(0, n - 1)
                new_word[pos] = random.randint(0, q - 1)
            new_word = tuple(new_word)
            
            # Check if new_word covers all unique_words
            new_nbs = _all_neighbors_indices(new_word, q, n, radius)
            new_nb_set = set(new_nbs)
            
            covers_all = all(uw in new_nb_set for uw in unique_words)
            
            if covers_all:
                # Check if this swap enables removal of another codeword
                # First, simulate the swap
                # Temporarily remove old, add new
                for nb in old_nbs:
                    coverage[nb] -= 1
                for nb in new_nbs:
                    coverage[nb] += 1
                
                current[i] = new_word
                cw_neighbors[i] = new_nbs
                
                # Try to remove other redundant codewords
                further_removal = False
                for j in range(len(current)):
                    if j == i:
                        continue
                    if _time_left(t0, deadline) < 10.0:
                        break
                    nbs_j = cw_neighbors[j]
                    redundant = True
                    for nb in nbs_j:
                        if coverage[nb] == 1:
                            redundant = False
                            break
                    if redundant:
                        for nb in nbs_j:
                            coverage[nb] -= 1
                        current.pop(j)
                        cw_neighbors.pop(j)
                        further_removal = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best = list(current)
                
                if not further_removal:
                    # No further removal, revert if size didn't decrease
                    # Actually we already updated, so keep it
                    pass
                
                found = True
                break
        
        if not found:
            # Revert: we already removed old and added new in the trial, but didn't accept
            # Actually we modified coverage in-place during the trial. Let's fix.
            # This is getting complex. Let's use a cleaner approach.
            # Rebuild coverage from current state
            # Actually, the modifications were made in the `if covers_all` block,
            # and we break out of the inner loop. If found=False, we need to revert.
            # But we only modify inside `if covers_all`, so if not found, no modification happened.
            pass
    
    return best


def _local_search_v2(codewords, q, n, radius, t0, deadline):
    """
    Cleaner local search with proper rollback.
    """
    if not codewords:
        return codewords
    
    total_words = q ** n
    current = list(codewords)
    best = list(current)
    best_size = len(current)
    
    def build_coverage(cw_list):
        cov = np.zeros(total_words, dtype=np.int32)
        cw_nb_list = []
        for cw in cw_list:
            nbs = _all_neighbors_indices(cw, q, n, radius)
            cw_nb_list.append(nbs)
            for nb in nbs:
                cov[nb] += 1
        return cov, cw_nb_list
    
    coverage, cw_neighbors = build_coverage(current)
    
    if np.any(coverage == 0):
        return current
    
    # Phase 1: Remove redundant
    changed = True
    while changed and _time_left(t0, deadline) > 5.0:
        changed = False
        random.shuffle(current)
        i = 0
        while i < len(current):
            if _time_left(t0, deadline) < 5.0:
                break
            nbs = cw_neighbors[i]
            if all(coverage[nb] > 1 for nb in nbs):
                for nb in nbs:
                    coverage[nb] -= 1
                current.pop(i)
                cw_neighbors.pop(i)
                changed = True
                if len(current) < best_size:
                    best_size = len(current)
                    best = list(current)
            else:
                i += 1
    
    # Phase 2: Swap attempts with rollback
    for iteration in range(1000):
        if _time_left(t0, deadline) < 15.0:
            break
        if not current:
            break
        
        i = random.randint(0, len(current) - 1)
        old_cw = current[i]
        old_nbs = cw_neighbors[i]
        
        unique_words = [nb for nb in old_nbs if coverage[nb] == 1]
        
        if not unique_words:
            for nb in old_nbs:
                coverage[nb] -= 1
            current.pop(i)
            cw_neighbors.pop(i)
            if len(current) < best_size:
                best_size = len(current)
                best = list(current)
            continue
        
        # Try replacements
        accepted = False
        for trial in range(20):
            base_idx = random.choice(unique_words)
            base_word = _idx_to_word(base_idx, q, n)
            new_word = list(base_word)
            for _ in range(random.randint(0, 2)):
                pos = random.randint(0, n - 1)
                new_word[pos] = random.randint(0, q - 1)
            new_word = tuple(new_word)
            
            if new_word == old_cw:
                continue
            
            new_nbs = _all_neighbors_indices(new_word, q, n, radius)
            new_nb_set = set(new_nbs)
            
            if not all(uw in new_nb_set for uw in unique_words):
                continue
            
            # Simulate swap
            old_cov_backup = coverage.copy()
            old_current_backup = list(current)
            old_cw_nb_backup = list(cw_neighbors)
            
            # Apply swap
            for nb in old_nbs:
                coverage[nb] -= 1
            for nb in new_nbs:
                coverage[nb] += 1
            current[i] = new_word
            cw_neighbors[i] = new_nbs
            
            # Try to remove redundants
            removals = 0
            for j in range(len(current)):
                if j == i or _time_left(t0, deadline) < 15.0:
                    continue
                nbs_j = cw_neighbors[j]
                if all(coverage[nb] > 1 for nb in nbs_j):
                    for nb in nbs_j:
                        coverage[nb] -= 1
                    current.pop(j)
                    cw_neighbors.pop(j)
                    removals += 1
            
            if removals > 0 or len(current) < best_size:
                accepted = True
                if len(current) < best_size:
                    best_size = len(current)
                    best = list(current)
                break
            else:
                # Revert
                coverage = old_cov_backup
                current = old_current_backup
                cw_neighbors = old_cw_nb_backup
        
    return best


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s limit
    
    # Generate initial code
    codewords = _greedy_cover(q, n, radius, t0, deadline)
    
    # Verify coverage
    total_words = q ** n
    covered = set()
    for cw in codewords:
        nbs = _all_neighbors_indices(cw, q, n, radius)
        covered.update(nbs)
    
    if len(covered) < total_words:
        # Add more codewords to cover remaining
        uncovered = set(range(total_words)) - covered
        while uncovered and _time_left(t0, deadline) > 5.0:
            target = random.choice(list(uncovered))
            target_word = _idx_to_word(target, q, n)
            nbs = _all_neighbors_indices(target_word, q, n, radius)
            codewords.append(target_word)
            covered.update(nbs)
            for nb in nbs:
                uncovered.discard(nb)
    
    # Local search to optimize
    if _time_left(t0, deadline) > 10.0:
        codewords = _local_search_v2(codewords, q, n, radius, t0, deadline)
    
    # Final verification and safety
    if _time_left(t0, deadline) > 3.0:
        covered = set()
        for cw in codewords:
            nbs = _all_neighbors_indices(cw, q, n, radius)
            covered.update(nbs)
        if len(covered) < total_words:
            uncovered = set(range(total_words)) - covered
            for target in list(uncovered)[:1000]:
                target_word = _idx_to_word(target, q, n)
                codewords.append(target_word)
    
    return [list(cw) for cw in codewords]
# EVOLVE-BLOCK-END
