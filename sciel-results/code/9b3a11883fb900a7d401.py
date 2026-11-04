"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 120.0  # safety margin before 150s kill


def _tetracode_solution(q, n, radius):
    """Direct sum of ternary tetracode (9 words len 4, perfect radius 1) with full tails."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words


def _build_cov_matrix(q, n, radius):
    """Build the full coverage matrix: cov_matrix[i] is a bool array of words within radius of word i."""
    space_size = q ** n
    # For each index i, compute the set of indices within Hamming distance <= radius
    # For radius 1: self + all single-position changes
    cov = np.zeros((space_size, space_size), dtype=bool)
    # Precompute digit arrays
    digits = np.zeros((space_size, n), dtype=np.int32)
    for i in range(space_size):
        x = i
        for pos in range(n - 1, -1, -1):
            digits[i, pos] = x % q
            x //= q
    # Self
    cov[np.arange(space_size), np.arange(space_size)] = True
    # Single-position changes (radius 1)
    if radius >= 1:
        for pos in range(n):
            for d in range(q):
                # All words where digit at pos != d, change to d
                mask = digits[:, pos] != d
                if not np.any(mask):
                    continue
                src_indices = np.where(mask)[0]
                # Compute target indices
                # target = src + (d - digits[src, pos]) * q^(n-1-pos)
                weight = q ** (n - 1 - pos)
                targets = src_indices + (d - digits[src_indices, pos]) * weight
                cov[src_indices, targets] = True
    return cov, digits


def _greedy_cover_v2(q, n, radius, time_limit=120.0):
    """Vectorized greedy set cover using uncovered-count array instead of boolean matrix ops."""
    space_size = q ** n
    cov, digits = _build_cov_matrix(q, n, radius)
    
    # uncovered_count[i] = number of uncovered words in the neighborhood of word i
    # Initially all words uncovered, so uncovered_count[i] = popcount(cov[i])
    uncovered_count = cov.sum(axis=1).astype(np.int32)
    
    # Track which words are uncovered
    uncovered = np.ones(space_size, dtype=bool)
    
    selected = []
    remaining = space_size
    
    while remaining > 0:
        if time.time() - _start_time > time_limit - 5:
            break
        # Pick the word covering the most uncovered
        best = int(np.argmax(uncovered_count))
        if uncovered_count[best] <= 0:
            break
        selected.append(best)
        
        # Update: for all words j, subtract the number of newly-covered words in their neighborhood
        # newly_covered = cov[best] & uncovered
        # For each j, subtract sum(cov[j] & newly_covered)
        # This is cov[:, cov[best] & uncovered].sum(axis=1)
        newly_covered_mask = cov[best] & uncovered
        if not np.any(newly_covered_mask):
            break
        # Subtract: for each row j, count how many of its covered words are in newly_covered_mask
        # cov[:, newly_covered_mask_indices].sum(axis=1)
        nc_indices = np.where(newly_covered_mask)[0]
        uncovered_count -= cov[:, nc_indices].sum(axis=1).astype(np.int32)
        uncovered_count = np.maximum(uncovered_count, 0)
        
        uncovered[newly_covered_mask] = False
        remaining -= len(nc_indices)
        
        # Mark best as no longer a candidate
        uncovered_count[best] = -1
    
    # Convert to words
    result = []
    for i in selected:
        result.append(digits[i].tolist())
    return result


def _vectorized_prune(q, n, radius, words, time_limit=120.0):
    """Prune redundant codewords using vectorized coverage union checks."""
    space_size = q ** n
    cov, digits = _build_cov_matrix(q, n, radius)
    k = len(words)
    
    # Map word indices
    word_indices = []
    for w in words:
        # Convert word to index
        idx = 0
        for d in w:
            idx = idx * q + d
        word_indices.append(idx)
    word_indices = np.array(word_indices, dtype=np.int64)
    
    # Coverage of each selected word
    sel_cov = cov[word_indices]  # shape (k, space_size)
    
    # Compute union of all coverage
    union = sel_cov.any(axis=0)
    
    # For each word i, check if its coverage is a subset of union of others
    # union_without_i = union of all except i
    # A word i is redundant if sel_cov[i] is a subset of (union - sel_cov[i] + others)
    # Equivalently: for every position p where sel_cov[i][p], there exists j != i with sel_cov[j][p]
    
    # Vectorized: for each position p, count how many selected words cover it
    cover_counts = sel_cov.sum(axis=0).astype(np.int32)  # shape (space_size,)
    
    # Word i is redundant if for all p where sel_cov[i][p], cover_counts[p] >= 2
    # i.e., sel_cov[i] & (cover_counts < 2) is all False
    redundant = np.all(~(sel_cov & (cover_counts[np.newaxis, :] < 2)), axis=1)
    
    if not np.any(redundant):
        return words
    
    # Remove redundant words, but be careful: removing one may make another non-redundant
    # Do iterative passes
    keep = ~redundant
    changed = True
    while changed:
        if time.time() - _start_time > time_limit - 5:
            break
        changed = False
        keep_indices = np.where(keep)[0]
        if len(keep_indices) == 0:
            break
        sub_cov = sel_cov[keep_indices]
        sub_union = sub_cov.any(axis=0)
        sub_counts = sub_cov.sum(axis=0).astype(np.int32)
        # Check redundancy within kept set
        sub_redundant = np.all(~(sub_cov & (sub_counts[np.newaxis, :] < 2)), axis=1)
        if np.any(sub_redundant):
            changed = True
            for ri in np.where(sub_redundant)[0]:
                keep[keep_indices[ri]] = False
    
    result = [words[i] for i in range(k) if keep[i]]
    return result


def _iterative_prune_v2(q, n, radius, words, time_limit=120.0):
    """Iteratively prune until no more words can be removed."""
    current_words = words
    while True:
        if time.time() - _start_time > time_limit - 5:
            break
        pruned = _vectorized_prune(q, n, radius, current_words, time_limit)
        if len(pruned) >= len(current_words):
            break
        current_words = pruned
    return current_words


def _verify_coverage(q, n, radius, words):
    """Check that all words are within radius of some codeword."""
    space_size = q ** n
    cov, digits = _build_cov_matrix(q, n, radius)
    word_indices = []
    for w in words:
        idx = 0
        for d in w:
            idx = idx * q + d
        word_indices.append(idx)
    word_indices = np.array(word_indices, dtype=np.int64)
    covered = cov[word_indices].any(axis=0)
    return bool(np.all(covered))


def construct(q, n, radius):
    global _start_time
    _start_time = time.time()
    time_limit = _DEADLINE - 5  # 5s safety margin

    # If parameters don't match the ternary length-8 radius-1 case, fall back to tetracode
    if q != 3 or n != 8 or radius != 1:
        return _tetracode_solution(q, n, radius)

    # Start with the known tetracode-based solution (729 words)
    seed_words = _tetracode_solution(q, n, radius)

    # Try vectorized pruning of the seed solution
    pruned = _iterative_prune_v2(q, n, radius, seed_words, time_limit)

    # If time remains, try a greedy-from-scratch which may find a smaller code
    elapsed = time.time() - _start_time
    if elapsed < time_limit - 20 and len(pruned) > 500:
        greedy_fresh = _greedy_cover_v2(q, n, radius, time_limit=time_limit)
        if len(greedy_fresh) < len(pruned):
            pruned = greedy_fresh
        # Try to prune the greedy result as well
        pruned = _iterative_prune_v2(q, n, radius, pruned, time_limit)

    # Verify coverage (sanity check)
    if not _verify_coverage(q, n, radius, pruned):
        # Fallback to seed
        return seed_words

    # Return best
    return pruned
# EVOLVE-BLOCK-END
