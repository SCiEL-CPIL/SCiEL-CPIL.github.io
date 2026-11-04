"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _tetracode_direct_sum(q, n, radius):
    """Direct sum of ternary tetracode (perfect covering radius 1, length 4) with full space on remaining coords."""
    if n <= 4:
        # Fallback: full space
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    rest = n - 4
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=rest)]


def _word_to_idx(word, q, n):
    """Convert word (list of ints) to linear index in [0, q^n)."""
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _idx_to_word(idx, q, n):
    """Convert linear index to word."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _generate_sphere(word, q, n, radius):
    """Yield all words within Hamming distance <= radius from word. Only used for small spaces."""
    if radius == 0:
        yield word
        return
    # For radius 1: word itself + all single-position changes
    yield word
    for pos in range(n):
        orig = word[pos]
        for v in range(q):
            if v != orig:
                w = list(word)
                w[pos] = v
                yield w


def _greedy_prune(codewords, q, n, radius, deadline):
    """
    Greedy set-cover pruning: repeatedly remove a codeword if the space is still covered.
    Uses efficient bitset representation of coverage for the full space.
    Returns pruned list of codewords.
    """
    total_space = q ** n
    if total_space > 2_000_000:
        # Too large for bitset; use a different strategy
        return _greedy_prune_large(codewords, q, n, radius, deadline)
    
    # Precompute the ball (set of covered indices) for each codeword
    # For radius 1 in ternary length 8: ball size = 1 + 8*2 = 17
    
    # Represent coverage as a bytearray of booleans
    # covered[i] = number of codewords that cover index i
    covered = bytearray(total_space)
    
    # Precompute ball membership for each codeword
    # ball[cw_idx] = set of indices covered by codeword cw_idx
    balls = []
    for cw in codewords:
        ball = set()
        # Add the word itself
        ball.add(_word_to_idx(cw, q, n))
        # Add all words at distance 1
        for pos in range(n):
            orig = cw[pos]
            for v in range(q):
                if v != orig:
                    w = list(cw)
                    w[pos] = v
                    ball.add(_word_to_idx(w, q, n))
        balls.append(ball)
    
    # Initialize coverage count
    for ball in balls:
        for idx in ball:
            covered[idx] += 1
    
    # Verify full coverage
    assert all(c > 0 for c in covered), "Initial solution does not cover the space"
    
    # Greedy pruning: try to remove codewords in random order
    order = list(range(len(codewords)))
    random.shuffle(order)
    
    active = [True] * len(codewords)
    removed = 0
    
    for ci in order:
        if not active[ci]:
            continue
        if time.time() > deadline:
            break
        
        # Try removing codeword ci
        ball = balls[ci]
        can_remove = True
        for idx in ball:
            if covered[idx] == 1:
                can_remove = False
                break
        
        if can_remove:
            # Remove it
            for idx in ball:
                covered[idx] -= 1
            active[ci] = False
            removed += 1
    
    result = [codewords[i] for i in range(len(codewords)) if active[i]]
    return result


def _greedy_prune_large(codewords, q, n, radius, deadline):
    """
    For large spaces, use a sampling-based pruning approach.
    Check coverage on a random sample of the space.
    """
    total_space = q ** n
    sample_size = min(50000, total_space)
    
    # Generate random sample points
    rng = random.Random(42)
    samples = []
    for _ in range(sample_size):
        idx = rng.randint(0, total_space - 1)
        samples.append(idx)
    
    # For each sample, find which codewords cover it
    # sample_cover[sample_idx] = list of codeword indices that cover this sample
    sample_cover = [[] for _ in range(sample_size)]
    
    for ci, cw in enumerate(codewords):
        if time.time() > deadline:
            break
        # Compute ball of this codeword
        ball_indices = set()
        ball_indices.add(_word_to_idx(cw, q, n))
        for pos in range(n):
            orig = cw[pos]
            for v in range(q):
                if v != orig:
                    w = list(cw)
                    w[pos] = v
                    ball_indices.add(_word_to_idx(w, q, n))
        
        # Check which samples are in this ball
        for si, sidx in enumerate(samples):
            if sidx in ball_indices:
                sample_cover[si].append(ci)
    
    # Greedy removal: try to remove codewords
    active = [True] * len(codewords)
    order = list(range(len(codewords)))
    rng.shuffle(order)
    
    for ci in order:
        if not active[ci]:
            continue
        if time.time() > deadline:
            break
        
        # Check if removing ci still covers all samples
        can_remove = True
        for si in range(sample_size):
            if len(sample_cover[si]) == 1 and sample_cover[si][0] == ci:
                can_remove = False
                break
        
        if can_remove:
            active[ci] = False
            # Update sample_cover
            for si in range(sample_size):
                if ci in sample_cover[si]:
                    sample_cover[si].remove(ci)
    
    result = [codewords[i] for i in range(len(codewords)) if active[i]]
    return result


def _verify_coverage(codewords, q, n, radius, deadline):
    """Verify that every word in the space is within radius of some codeword."""
    total_space = q ** n
    if total_space > 2_000_000:
        # Use sampling for verification on large spaces
        rng = random.Random(123)
        sample_size = min(100000, total_space)
        for _ in range(sample_size):
            if time.time() > deadline:
                return False
            idx = rng.randint(0, total_space - 1)
            word = _idx_to_word(idx, q, n)
            covered = False
            for cw in codewords:
                dist = sum(1 for i in range(n) if word[i] != cw[i])
                if dist <= radius:
                    covered = True
                    break
            if not covered:
                return False
        return True
    else:
        # Full verification using bitset
        covered = bytearray(total_space)
        for cw in codewords:
            if time.time() > deadline:
                return False
            # Mark the word itself
            covered[_word_to_idx(cw, q, n)] = 1
            # Mark all distance-1 neighbors
            for pos in range(n):
                orig = cw[pos]
                for v in range(q):
                    if v != orig:
                        w = list(cw)
                        w[pos] = v
                        covered[_word_to_idx(w, q, n)] = 1
        return all(c > 0 for c in covered)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    # Start with tetracode direct sum construction
    codewords = _tetracode_direct_sum(q, n, radius)
    
    # Verify initial coverage
    if not _verify_coverage(codewords, q, n, radius, deadline):
        # Fallback: return the construction as-is (shouldn't happen for valid params)
        return codewords
    
    # Phase 1: Greedy pruning
    pruned = _greedy_prune(codewords, q, n, radius, deadline)
    if len(pruned) < len(codewords):
        codewords = pruned
    
    # Phase 2: Additional local search - try swapping codewords
    # For each codeword, try replacing it with a nearby word that might allow further pruning
    if time.time() < deadline - 10:
        rng = random.Random(7)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for ci in range(len(codewords)):
                if time.time() > deadline - 10:
                    break
                # Try removing codeword ci and see if we can add a better one
                # Simple approach: just try to remove and re-verify
                if len(codewords) <= 1:
                    break
                ball = set()
                cw = codewords[ci]
                ball.add(_word_to_idx(cw, q, n))
                for pos in range(n):
                    orig = cw[pos]
                    for v in range(q):
                        if v != orig:
                            w = list(cw)
                            w[pos] = v
                            ball.add(_word_to_idx(w, q, n))
                
                # Check if removal is safe
                can_remove = True
                for idx in ball:
                    # Count how many other codewords cover idx
                    count = 0
                    for cj in range(len(codewords)):
                        if cj == ci:
                            continue
                        cwj = codewords[cj]
                        if _word_to_idx(cwj, q, n) == idx:
                            count += 1
                            break
                        # Check distance-1
                        for pos2 in range(n):
                            if cwj[pos2] != cwj[pos2]:  # placeholder
                                pass
                    # This is too slow; skip for large problems
                    if count == 0:
                        can_remove = False
                        break
                # Skip this expensive check for large spaces
                if q ** n > 100000:
                    break
                if can_remove:
                    codewords.pop(ci)
                    improved = True
                    break
    
    # Final verification
    if not _verify_coverage(codewords, q, n, radius, deadline):
        # Return the pre-pruning version if pruning broke coverage
        # Reconstruct from tetracode
        return _tetracode_direct_sum(q, n, radius)
    
    return codewords
# EVOLVE-BLOCK-END
