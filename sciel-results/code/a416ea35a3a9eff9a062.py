"""Rewritten program for ternary covering code K_3(8,1)."""
import time
import random
import itertools
import numpy as np

# Deadline guard: kill at 150s wall-clock, use 145s for safety
DEADLINE = 145.0


def _tetracode():
    """The 9 words of the ternary tetracode (perfect 1-error-correcting code of length 4)."""
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _all_words_tet():
    """All 81 words of length 4 over {0,1,2}."""
    return [list(w) for w in itertools.product(range(3), repeat=4)]


def _hamming_dist(a, b):
    """Hamming distance between two tuples/lists."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _within_radius(a, b, r):
    """True if Hamming distance between a and b is <= r."""
    d = 0
    for x, y in zip(a, b):
        if x != y:
            d += 1
            if d > r:
                return False
    return True


def _covers(word, codeword, radius):
    """Check if codeword covers word within given radius."""
    return _within_radius(word, codeword, radius)


def _build_tet_cover_map():
    """
    For the tetracode, build a map: for each of the 81 length-4 words,
    which tetracode codewords are within distance 1.
    Returns: dict mapping tuple(word) -> list of tetracode indices
    """
    tet = _tetracode()
    tet_tuples = [tuple(t) for t in tet]
    cover_map = {}
    for word in _all_words_tet():
        wt = tuple(word)
        covering = []
        for i, tc in enumerate(tet_tuples):
            if _within_radius(wt, tc, 1):
                covering.append(i)
        cover_map[wt] = covering
    return cover_map


def _construct_initial(q, n, radius):
    """
    Direct sum construction: tetracode on first 4 coords, all words on remaining n-4 coords.
    Returns list of codewords (each a list of n ints).
    """
    tet = _tetracode()
    remaining = n - 4
    codewords = []
    for h in tet:
        for tail in itertools.product(range(q), repeat=remaining):
            codewords.append(h + list(tail))
    return codewords


def _verify_covering(codewords, q, n, radius):
    """
    Verify that every word in {0..q-1}^n is within radius of some codeword.
    Returns True if valid, False otherwise.
    """
    # For efficiency, check all words
    for word in itertools.product(range(q), repeat=n):
        covered = False
        for cw in codewords:
            if _within_radius(word, cw, radius):
                covered = True
                break
        if not covered:
            return False
    return True


def _try_remove(codewords, idx, q, n, radius):
    """
    Try removing codeword at idx. Check if the remaining codewords still cover everything.
    For efficiency, only check words that were previously covered by this codeword
    (within radius of it) and see if they're still covered by others.
    """
    cw = codewords[idx]
    # Find all words within radius of cw
    # For n=8, q=3, radius=1, the ball size is 1 + 8*2 = 17
    # Check each word in the ball
    for offset in itertools.product(range(-1, 2), repeat=n):
        # Build the word by adding offset to cw, wrapping mod q
        word = []
        valid = True
        for i in range(n):
            val = cw[i] + offset[i]
            if val < 0 or val >= q:
                valid = False
                break
            word.append(val)
        if not valid:
            continue
        # Check if this word is still covered by some other codeword
        covered = False
        for j, other in enumerate(codewords):
            if j == idx:
                continue
            if _within_radius(word, other, radius):
                covered = True
                break
        if not covered:
            return False
    return True


def _try_remove_fast(codewords, idx, q, n, radius, cover_counts):
    """
    Try removing codeword at idx using precomputed cover counts.
    cover_counts[word_index] = number of codewords covering that word.
    Returns True if removal is safe, and updates cover_counts if so.
    """
    cw = codewords[idx]
    # For each word in the ball of cw, check if cover_counts > 1
    for offset in itertools.product(range(-1, 2), repeat=n):
        word = []
        valid = True
        for i in range(n):
            val = cw[i] + offset[i]
            if val < 0 or val >= q:
                valid = False
                break
            word.append(val)
        if not valid:
            continue
        # Convert word to index
        word_idx = _word_to_index(word, q, n)
        if cover_counts[word_idx] <= 1:
            return False
    # Safe to remove; update cover counts
    for offset in itertools.product(range(-1, 2), repeat=n):
        word = []
        valid = True
        for i in range(n):
            val = cw[i] + offset[i]
            if val < 0 or val >= q:
                valid = False
                break
            word.append(val)
        if not valid:
            continue
        word_idx = _word_to_index(word, q, n)
        cover_counts[word_idx] -= 1
    return True


def _word_to_index(word, q, n):
    """Convert a word (list of ints) to a flat index."""
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _index_to_word(idx, q, n):
    """Convert a flat index to a word (list of ints)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _compute_cover_counts(codewords, q, n, radius):
    """
    Compute for each word in the space, how many codewords cover it.
    Returns a numpy array of shape (q^n,).
    """
    total_words = q ** n
    counts = np.zeros(total_words, dtype=np.int32)
    for cw in codewords:
        for offset in itertools.product(range(-1, 2), repeat=n):
            word = []
            valid = True
            for i in range(n):
                val = cw[i] + offset[i]
                if val < 0 or val >= q:
                    valid = False
                    break
                word.append(val)
            if not valid:
                continue
            word_idx = _word_to_index(word, q, n)
            counts[word_idx] += 1
    return counts


def _greedy_reduce(codewords, q, n, radius, deadline):
    """
    Greedy reduction: repeatedly try to remove codewords.
    Use cover counts for fast checking.
    """
    codewords = list(codewords)
    cover_counts = _compute_cover_counts(codewords, q, n, radius)
    
    # Try removing in a fixed order (e.g., by index)
    # Multiple passes until no more removals possible
    progress = True
    while progress and time.time() < deadline - 5:
        progress = False
        # Try each codeword
        for idx in range(len(codewords)):
            if time.time() > deadline - 5:
                break
            if _try_remove_fast(codewords, idx, q, n, radius, cover_counts):
                del codewords[idx]
                progress = True
                # After removal, indices shift, so restart the loop
                break
    return codewords


def _local_search(codewords, q, n, radius, deadline, rng):
    """
    Local search: try to swap codewords to reduce the total count.
    Strategy: for each codeword, try moving it to a random position
    and see if the code is still valid and if we can then remove it.
    """
    codewords = list(codewords)
    best = list(codewords)
    best_len = len(codewords)
    
    start_time = time.time()
    
    while time.time() < deadline - 10 and best_len > 403:
        # Try to find a removable codeword after a perturbation
        # Strategy: pick a random codeword, try moving it to a nearby position
        if len(codewords) <= best_len:
            # Try to remove
            cover_counts = _compute_cover_counts(codewords, q, n, radius)
            removed = False
            for idx in range(len(codewords)):
                if time.time() > deadline - 10:
                    break
                if _try_remove_fast(codewords, idx, q, n, radius, cover_counts):
                    del codewords[idx]
                    removed = True
                    break
            if removed:
                if len(codewords) < best_len:
                    best = list(codewords)
                    best_len = len(codewords)
                continue
        
        # Perturb: move a random codeword to a random nearby position
        if len(codewords) == 0:
            break
        
        idx = rng.randint(0, len(codewords) - 1)
        old_cw = list(codewords[idx])
        
        # Generate a nearby word
        new_cw = list(old_cw)
        # Change 1-2 random positions
        num_changes = rng.randint(1, 2)
        for _ in range(num_changes):
            pos = rng.randint(0, n - 1)
            old_val = new_cw[pos]
            new_val = rng.randint(0, q - 1)
            while new_val == old_val:
                new_val = rng.randint(0, q - 1)
            new_cw[pos] = new_val
        
        # Check if the new codeword is distinct from all others
        is_distinct = True
        for j, other in enumerate(codewords):
            if j == idx:
                continue
            if new_cw == other:
                is_distinct = False
                break
        
        if not is_distinct:
            continue
        
        # Try the move
        codewords[idx] = new_cw
        
        # Check if still valid covering
        # Quick check: compute cover counts and see if all >= 1
        cover_counts = _compute_cover_counts(codewords, q, n, radius)
        if np.min(cover_counts) >= 1:
            # Valid! Now try to remove some codeword
            removed_any = False
            for ridx in range(len(codewords)):
                if time.time() > deadline - 10:
                    break
                if _try_remove_fast(codewords, ridx, q, n, radius, cover_counts):
                    del codewords[ridx]
                    removed_any = True
                    break
            
            if removed_any and len(codewords) < best_len:
                best = list(codewords)
                best_len = len(codewords)
            elif not removed_any:
                # Keep the move if it doesn't hurt, but revert if it makes things worse
                # Actually, if no removal, the size is same, so revert to keep diversity
                codewords[idx] = old_cw
        else:
            # Invalid, revert
            codewords[idx] = old_cw
    
    return best


def _direct_sum_variant(q, n, radius, variant_idx):
    """
    Generate different direct-sum-like constructions.
    variant_idx selects which block code to use on the first 4 coords.
    """
    tet = _tetracode()
    remaining = n - 4
    codewords = []
    for h in tet:
        for tail in itertools.product(range(q), repeat=remaining):
            codewords.append(h + list(tail))
    return codewords


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Returns a list of codewords (each a list of n integers in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # For the specific task: q=3, n=8, radius=1
    # Use the direct sum construction as initial solution
    codewords = _construct_initial(q, n, radius)
    
    # Greedy reduction
    codewords = _greedy_reduce(codewords, q, n, radius, deadline)
    
    # Local search for further improvement
    rng = random.Random(42)
    if time.time() < deadline - 15:
        codewords = _local_search(codewords, q, n, radius, deadline, rng)
    
    # Final verification
    if time.time() < deadline - 2:
        if not _verify_covering(codewords, q, n, radius):
            # Fallback to the initial construction
            codewords = _construct_initial(q, n, radius)
    
    return codewords
