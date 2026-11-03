"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _precompute_neighbor_table(q, n, radius):
    """
    Precompute for every word index in [0, q^n) the list of neighbor indices
    within Hamming distance <= radius. Returns a list of arrays.
    """
    total = q ** n
    # Build all words as a 2D array: shape (total, n)
    all_words = np.zeros((total, n), dtype=np.int32)
    for idx in range(total):
        tmp = idx
        for i in range(n):
            all_words[idx, i] = tmp % q
            tmp //= q

    # For each word, compute neighbors within radius
    # For efficiency, we'll compute distance to all words via broadcasting in chunks
    # But for q=3, n=8, total=6561, we can do it in one shot
    # Distance matrix would be 6561x6561 = ~43M entries, manageable

    # Compute pairwise Hamming distances using broadcasting
    # all_words[:, :, None] vs all_words[None, :, :] -> shape (total, total, n)
    # This is 6561*6561*8 ~ 345M bytes for int32, ~1.4GB - too much
    # Instead, compute in chunks

    neighbor_lists = []
    chunk_size = 512
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = all_words[start:end]  # shape (chunk_size, n)
        # Compute distances from chunk to all words
        # diff shape: (chunk_size, total, n)
        diff = (chunk[:, None, :] != all_words[None, :, :])  # boolean
        dists = diff.sum(axis=2)  # shape (chunk_size, total)
        for ci in range(end - start):
            nbs = np.where(dists[ci] <= radius)[0]
            neighbor_lists.append(nbs)

    return neighbor_lists


def _greedy_cover_vectorized(q, n, radius, deadline):
    """
    Vectorized greedy set cover.
    Precomputes neighbor table, then repeatedly picks the word covering most uncovered.
    """
    total = q ** n
    start_time = time.time()

    # Precompute neighbor table
    neighbor_lists = _precompute_neighbor_table(q, n, radius)

    covered = np.zeros(total, dtype=np.uint8)
    codeword_indices = []

    # Precompute for each word: the array of its neighbor indices
    # neighbor_lists[i] is an np.array of neighbor indices for word i

    while time.time() < deadline - 5:
        # For each candidate, count uncovered neighbors
        # Vectorized: for each word i, count = sum(1 for nb in neighbor_lists[i] if covered[nb]==0)
        # We can do this with a loop over candidates but using numpy for the counting

        best_idx = -1
        best_count = -1

        # Process in chunks for memory
        chunk_size = 1024
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            # For each word in chunk, count uncovered neighbors
            # This is a loop but each iteration is a numpy op
            for i in range(start, end):
                nbs = neighbor_lists[i]
                count = np.count_nonzero(covered[nbs] == 0)
                if count > best_count:
                    best_count = count
                    best_idx = i

        if best_idx < 0 or best_count <= 0:
            break

        codeword_indices.append(best_idx)
        nbs = neighbor_lists[best_idx]
        covered[nbs] = 1

        if covered.all():
            break

    # Convert indices to codewords
    codewords = []
    for idx in codeword_indices:
        word = []
        tmp = idx
        for i in range(n):
            word.append(tmp % q)
            tmp //= q
        word.reverse()
        codewords.append(word)

    return codewords


def _is_valid_cover_fast(codeword_indices, neighbor_lists, q, n, radius):
    """Check if the set of codeword indices covers all words."""
    total = q ** n
    covered = np.zeros(total, dtype=np.uint8)
    for idx in codeword_indices:
        covered[neighbor_lists[idx]] = 1
    return covered.all()


def _greedy_removal(codeword_indices, neighbor_lists, q, n, radius, deadline):
    """
    Greedy removal: try to remove codewords one by one if coverage is maintained.
    Uses a coverage count array for efficiency.
    """
    total = q ** n
    start_time = time.time()

    # Build coverage count: for each word, how many codewords cover it
    cov_count = np.zeros(total, dtype=np.int32)
    for idx in codeword_indices:
        cov_count[neighbor_lists[idx]] += 1

    # A word is "critically covered" if cov_count == 1
    # We can remove codeword i if no word it covers has cov_count == 1

    changed = True
    while changed and time.time() < deadline - 3:
        changed = False
        # Try removing each codeword
        for i in range(len(codeword_indices)):
            if time.time() >= deadline - 3:
                break
            idx = codeword_indices[i]
            nbs = neighbor_lists[idx]
            # Check if any word in nbs has cov_count == 1
            if np.any(cov_count[nbs] == 1):
                continue
            # Safe to remove
            cov_count[nbs] -= 1
            codeword_indices.pop(i)
            changed = True
            break  # Restart

    return codeword_indices


def _swap_improvement(codeword_indices, neighbor_lists, q, n, radius, deadline):
    """
    Try to replace a codeword with another that covers more uncovered words
    while maintaining full coverage. This is a 1-swap local search.
    """
    total = q ** n
    start_time = time.time()

    cov_count = np.zeros(total, dtype=np.int32)
    for idx in codeword_indices:
        cov_count[neighbor_lists[idx]] += 1

    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for i in range(len(codeword_indices)):
            if time.time() >= deadline - 3:
                break
            old_idx = codeword_indices[i]
            old_nbs = neighbor_lists[old_idx]
            # Words that would become uncovered if we remove old_idx
            critical = old_nbs[cov_count[old_nbs] == 1]
            if len(critical) == 0:
                # Just remove
                cov_count[old_nbs] -= 1
                codeword_indices.pop(i)
                improved = True
                break

            # Need a replacement that covers all critical words
            # Find candidate: a word not in codeword_indices that covers all critical
            # This is expensive, so sample
            candidates = random.sample(range(total), min(200, total))
            found = False
            for cand in candidates:
                if cand in codeword_indices:
                    continue
                cand_nbs = neighbor_lists[cand]
                # Check if cand covers all critical words
                if np.all(np.isin(critical, cand_nbs)):
                    # Replace
                    cov_count[old_nbs] -= 1
                    cov_count[cand_nbs] += 1
                    codeword_indices[i] = cand
                    improved = True
                    found = True
                    break
            if found:
                break

    return codeword_indices


def construct(q, n, radius):
    """Construct a covering code using vectorized greedy + removal + swap."""
    start_time = time.time()
    deadline = start_time + 115  # Leave 35s safety margin

    total = q ** n

    # Precompute neighbor table (this is the expensive part)
    neighbor_lists = _precompute_neighbor_table(q, n, radius)

    # Initial construction: use a structured starting point if possible
    # For q=3, n=8: use tetracode direct sum as baseline
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Convert to indices
        initial_indices = []
        for word in initial_words:
            idx = 0
            for v in word:
                idx = idx * q + v
            initial_indices.append(idx)
        # Remove duplicates
        initial_indices = list(set(initial_indices))
    else:
        initial_indices = []

    if not initial_indices:
        # Use greedy to build initial solution
        initial_indices = _greedy_cover_vectorized(q, n, radius, deadline)

    # Verify initial solution is valid
    if not _is_valid_cover_fast(initial_indices, neighbor_lists, q, n, radius):
        # Fallback: build from scratch with greedy
        initial_indices = _greedy_cover_vectorized(q, n, radius, deadline)

    # Greedy removal to reduce size
    initial_indices = _greedy_removal(initial_indices, neighbor_lists, q, n, radius, deadline)

    # Swap improvement
    if time.time() < deadline - 5:
        initial_indices = _swap_improvement(initial_indices, neighbor_lists, q, n, radius, deadline)

    # Final verification
    if not _is_valid_cover_fast(initial_indices, neighbor_lists, q, n, radius):
        # Emergency: rebuild
        initial_indices = _greedy_cover_vectorized(q, n, radius, deadline)

    # Convert indices to codewords
    codewords = []
    for idx in initial_indices:
        word = []
        tmp = idx
        for i in range(n):
            word.append(tmp % q)
            tmp //= q
        word.reverse()
        codewords.append(word)

    return codewords
# EVOLVE-BLOCK-END
