"""Ternary covering code K_3(6,1) via simulated annealing with vectorized coverage."""
import time
import numpy as np

# Fixed deadline guard: return best before ~145s wall-clock (evaluator kills at 150s).
DEADLINE_S = 145.0


def _tetracode():
    """The 9-word ternary tetracode (perfect covering radius 1 on length 4)."""
    return np.array([[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)], dtype=np.uint8)


def _full_space(q, n):
    """All q^n words as a (q**n, n) uint8 array."""
    total = q ** n
    words = np.zeros((total, n), dtype=np.uint8)
    for j in range(n):
        base = q ** (n - 1 - j)
        # tile/repeat pattern for mixed-radix enumeration
        words[:, j] = (np.arange(total) // base) % q
    return words


def _coverage_counts(codewords, space, q, n):
    """
    For each space word, count how many codewords are within Hamming distance <= 1.
    Uses the fact that a codeword c covers c itself and all words at distance 1.
    We accumulate counts by iterating codewords and adding to their closed ball.
    """
    m = len(codewords)
    counts = np.zeros(len(space), dtype=np.int32)
    if m == 0:
        return counts
    # Precompute index of each space word for O(1) lookup? Space is full q^n, so
    # we can compute the linear index of any word.
    # Instead, for each codeword, we add 1 to its own index and to each neighbor's index.
    # To get the linear index of a word w: sum w[j] * q^(n-1-j)
    powers = np.array([q ** (n - 1 - j) for j in range(n)], dtype=np.int64)

    # For speed, process codewords in batches if possible, but m is small (~70-90).
    # For each codeword c:
    #   idx_c = dot(c, powers)
    #   counts[idx_c] += 1
    #   for each position p and each alternative value v in 0..q-1, v != c[p]:
    #       neighbor = c with position p set to v
    #       idx_n = dot(neighbor, powers)
    #       counts[idx_n] += 1
    #
    # We can vectorize over codewords.
    cw = codewords  # (m, n) uint8
    # Compute all codeword indices
    cw_indices = cw.astype(np.int64) @ powers  # (m,)
    np.add.at(counts, cw_indices, 1)

    # For neighbors: for each position p, for each alt value v != cw[:, p]
    # This is tricky to fully vectorize without loops over p and v.
    # n=6, q=3, so 6*2=12 neighbors per codeword. m~80. Total ~960 neighbor updates.
    # Looping over p is fine.
    for p in range(n):
        # For each codeword, the base index without position p
        # Actually, easier: for each codeword i, for each alt v:
        #   neighbor index = cw_indices[i] - cw[i,p]*powers[p] + v*powers[p]
        # We can vectorize over i for a fixed p and v.
        base_idx = cw_indices - cw[:, p].astype(np.int64) * powers[p]  # (m,)
        for v in range(q):
            if v == 0:
                continue  # we'll handle all v, but skip if v equals original? No, we want v != original.
            # Actually, let's just compute for all v and subtract the self case later? 
            # No, simpler: for each v, the neighbor index is base_idx + v * powers[p].
            # But if v == cw[i,p], that's the codeword itself, already counted.
            # So we want to add 1 for each v != cw[i,p].
            # Let's create neighbor_indices for all v, then mask out v == cw[i,p].
            # For fixed p, v:
            nidx = base_idx + v * powers[p]  # (m,)
            # Mask: where cw[i,p] != v
            mask = (cw[:, p].astype(np.int64) != v)
            if mask.any():
                np.add.at(counts, nidx[mask], 1)
    return counts


def _is_valid(codewords, space, q, n):
    """Check if every space word is covered by at least one codeword."""
    if len(codewords) == 0:
        return len(space) == 0
    counts = _coverage_counts(codewords, space, q, n)
    return np.all(counts >= 1)


def construct(q, n, radius):
    start_time = time.time()
    
    # Only handle the specific case q=3, n=6, radius=1 efficiently.
    # For generality, fall back to a simple construction if parameters differ.
    if q != 3 or n != 6 or radius != 1:
        # Fallback: direct sum with tetracode if possible, else greedy
        if q == 3 and n >= 4:
            tet = _tetracode()
            # Extend to length n by adding zeros
            tet_ext = np.zeros((9, n), dtype=np.uint8)
            tet_ext[:, :4] = tet
            # For remaining coordinates, we need to cover all combinations.
            # This fallback is not optimal but valid.
            # Actually, for n > 4, we need more words. Let's use a simple greedy.
            # But for now, just return the extended tetracode (invalid for n>4).
            # Better fallback: random + greedy.
            pass
        # Generic fallback: greedy covering
        space = _full_space(q, n)
        codewords = []
        uncovered = set(range(len(space)))
        while uncovered and time.time() - start_time < DEADLINE_S - 5:
            # Find codeword that covers most uncovered words
            best_cw = None
            best_count = -1
            # Sample candidate codewords from space
            candidates = np.random.choice(len(space), size=min(200, len(space)), replace=False)
            for ci in candidates:
                cw = space[ci]
                # Count how many uncovered words are within distance 1
                # This is expensive; for fallback, just pick randomly
                pass
            # Simplified fallback: just return a random valid set (may be large)
            # For robustness, return the full space if we can't do better quickly.
            # But that's 729 words. Let's try a simple greedy with limited time.
            # Actually, let's just return the tetracode-based construction for n=6.
            # If parameters don't match, return a simple construction.
            if q == 3 and n == 4:
                return _tetracode().tolist()
            # For other cases, return a trivial (but possibly large) valid code
            # by taking all words (always valid).
            return space.tolist()

    space = _full_space(q, n)
    tet = _tetracode()
    
    # Initial codewords: the 9 tetracode words extended to length 6 with zeros.
    # This is not a valid covering code for length 6, but it's a good starting point
    # for simulated annealing. Actually, it's not even close.
    # Better initial: use a known good construction or random.
    # Let's start with a random set of 80 words, then optimize.
    # Or better: start with the 81-word construction from tetracode direct sum with 2 coords.
    # Tetracode (9 words, len 4) x all 3^2=9 words on last 2 coords = 81 words.
    # This is a valid covering code.
    init_cw = np.zeros((81, 6), dtype=np.uint8)
    for i in range(9):
        for j in range(9):
            init_cw[9*i + j, :4] = tet[i]
            init_cw[9*i + j, 4] = j // 3
            init_cw[9*i + j, 5] = j % 3
    
    # Verify initial validity
    if not _is_valid(init_cw, space, q, n):
        # Shouldn't happen, but fallback
        return space.tolist()
    
    best_cw = init_cw.copy()
    best_size = len(best_cw)
    
    # Simulated annealing to reduce size
    # State: a set of codewords. Moves:
    #   1. Remove a random codeword (if still valid)
    #   2. Replace a random codeword with a random other word (if still valid or improves "potential")
    # Objective: minimize number of codewords subject to validity.
    # We can use a penalty-based approach: minimize |C| + lambda * (number of uncovered words)
    # with lambda large. Or just try removals and accept if still valid.
    
    rng = np.random.default_rng(42)
    current_cw = init_cw.copy()
    current_size = len(current_cw)
    
    # Temperature schedule
    T0 = 1.0
    T_min = 1e-4
    # We'll do many iterations, checking time.
    
    # Precompute space indices for fast lookup
    # For a given word, its index in space is dot(word, powers)
    powers = np.array([q ** (n - 1 - j) for j in range(n)], dtype=np.int64)
    
    # Helper to get index of a word
    def word_to_idx(w):
        return int(np.dot(w.astype(np.int64), powers))
    
    # Helper to get word from index
    def idx_to_word(idx):
        w = np.zeros(n, dtype=np.uint8)
        for j in range(n):
            base = q ** (n - 1 - j)
            w[j] = (idx // base) % q
        return w
    
    # Coverage check for a specific word against codewords
    def is_word_covered(word, cw_set):
        """Check if word is within distance 1 of any codeword in cw_set."""
        w_idx = word_to_idx(word)
        # Check self
        for cw in cw_set:
            if np.array_equal(cw, word):
                return True
            # Check distance 1
            # Count mismatches
            mismatches = np.sum(cw != word)
            if mismatches <= 1:
                return True
        return False
    
    # This is too slow. Let's use the vectorized coverage counts.
    # But for SA, we need to check validity after each move.
    # With m~80 and space=729, _coverage_counts is fast enough.
    
    num_iterations = 0
    last_best_update = start_time
    
    while time.time() - start_time < DEADLINE_S - 10:
        # Try a random move
        move_type = rng.random()
        
        if move_type < 0.7 and current_size > 1:
            # Try to remove a random codeword
            idx_remove = rng.integers(0, current_size)
            new_cw = np.delete(current_cw, idx_remove, axis=0)
            if _is_valid(new_cw, space, q, n):
                # Accept removal
                current_cw = new_cw
                current_size = len(current_cw)
                if current_size < best_size:
                    best_cw = current_cw.copy()
                    best_size = current_size
                    last_best_update = time.time()
                num_iterations += 1
            else:
                # Reject
                pass
        elif move_type < 0.9:
            # Try to replace a random codeword with a random other word
            if current_size == 0:
                continue
            idx_replace = rng.integers(0, current_size)
            # Pick a random new word from space
            new_word = space[rng.integers(0, len(space))]
            if np.array_equal(new_word, current_cw[idx_replace]):
                continue
            new_cw = current_cw.copy()
            new_cw[idx_replace] = new_word
            if _is_valid(new_cw, space, q, n):
                # Accept replacement (size unchanged, but may enable future removals)
                current_cw = new_cw
                num_iterations += 1
            else:
                # Reject
                pass
        else:
            # Try to add a random codeword (to escape local minima)
            new_word = space[rng.integers(0, len(space))]
            # Check if it's already in the set
            found = False
            for i in range(current_size):
                if np.array_equal(new_word, current_cw[i]):
                    found = True
                    break
            if not found:
                new_cw = np.vstack([current_cw, new_word.reshape(1, -1)])
                current_cw = new_cw
                current_size = len(current_cw)
                num_iterations += 1
    
    # Final polish: try to remove codewords greedily
    # Sort codewords by how "critical" they are (harder to remove)
    # Just try removing in random order
    for _ in range(100):
        if time.time() - start_time > DEADLINE_S - 5:
            break
        removed_any = False
        for i in range(len(best_cw)):
            if time.time() - start_time > DEADLINE_S - 5:
                break
            new_cw = np.delete(best_cw, i, axis=0)
            if _is_valid(new_cw, space, q, n):
                best_cw = new_cw
                removed_any = True
                break  # Restart scan after a removal
        if not removed_any:
            break
    
    return best_cw.tolist()
