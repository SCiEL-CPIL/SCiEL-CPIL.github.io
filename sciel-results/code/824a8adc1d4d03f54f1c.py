"""Improved covering-code search: vectorized greedy + simulated annealing with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
import math

# Deadline guard: evaluator kills at 150s wall-clock; keep a safety margin
_DEADLINE_SECONDS = 140.0


def _elapsed(start):
    return time.monotonic() - start


def _idx_to_tuple(idx, n, q):
    """Convert flat index to tuple of digits."""
    t = []
    for i in range(n):
        t.append(idx % q)
        idx //= q
    return t[::-1]


def _tuple_to_idx(t, n, q):
    """Convert tuple of digits to flat index."""
    idx = 0
    for d in t:
        idx = idx * q + d
    return idx


def _all_neighbors(codeword, n, q, radius):
    """Yield all words within Hamming distance <= radius of codeword."""
    yield tuple(codeword)
    for pos in range(n):
        orig = codeword[pos]
        for val in range(q):
            if val == orig:
                continue
            new = list(codeword)
            new[pos] = val
            yield tuple(new)


def _build_coverage_matrix(total, n, q, radius):
    """
    Build a sparse-like representation: for each codeword index, store the list of word indices it covers.
    Also build reverse: for each word, list of codewords that cover it.
    """
    # forward_cover[c] = list of words covered by codeword c
    # reverse_cover[w] = list of codewords covering word w
    forward_cover = [[] for _ in range(total)]
    reverse_cover = [[] for _ in range(total)]
    
    for c in range(total):
        c_tuple = _idx_to_tuple(c, n, q)
        # The codeword itself
        forward_cover[c].append(c)
        reverse_cover[c].append(c)
        # Single position changes (radius 1)
        for pos in range(n):
            orig = c_tuple[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_tuple = list(c_tuple)
                new_tuple[pos] = val
                w = _tuple_to_idx(new_tuple, n, q)
                forward_cover[c].append(w)
                reverse_cover[w].append(c)
    
    return forward_cover, reverse_cover


def _greedy_construct(q, n, radius, rng, max_time_remaining):
    """
    Vectorized greedy set cover.
    Returns list of codewords (tuples) that cover all of {0..q-1}^n.
    """
    total = q ** n
    covered = np.zeros(total, dtype=np.uint8)
    codewords = []

    # Build reverse map
    forward_cover, reverse_cover = _build_coverage_matrix(total, n, q, radius)
    
    # gain[c] = number of currently uncovered words within distance <= radius of codeword c
    gain = np.zeros(total, dtype=np.int32)
    initial_gain = 1 + n * (q - 1)
    gain[:] = initial_gain

    while True:
        elapsed_check = _elapsed(max_time_remaining[0])
        if elapsed_check > max_time_remaining[1]:
            break

        if np.max(gain) <= 0:
            break
        c = int(np.argmax(gain))
        codewords.append(_idx_to_tuple(c, n, q))

        for w in forward_cover[c]:
            if covered[w] == 0:
                covered[w] = 1
                for cc in reverse_cover[w]:
                    gain[cc] -= 1

        if covered.all():
            break

    return codewords


def _is_cover(codewords, q, n, radius):
    """Check if codewords form a valid covering."""
    total = q ** n
    covered = np.zeros(total, dtype=np.uint8)
    for cw in codewords:
        c_idx = _tuple_to_idx(cw, n, q)
        covered[c_idx] = 1
        for pos in range(n):
            orig = cw[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_cw = list(cw)
                new_cw[pos] = val
                w_idx = _tuple_to_idx(new_cw, n, q)
                covered[w_idx] = 1
    return covered.all()


def _simulated_annealing(codewords, q, n, radius, rng, max_time_remaining):
    """
    Simulated annealing on the binary selection vector.
    
    State: boolean array x of length total, where x[c]=1 means codeword c is selected.
    Energy: E = alpha * (uncovered_count) + beta * (num_selected)
    We want to minimize both, with heavy penalty on uncovered words.
    
    Moves: flip one bit (add or remove a codeword).
    """
    total = q ** n
    forward_cover, reverse_cover = _build_coverage_matrix(total, n, q, radius)
    
    # Initialize from greedy result
    x = np.zeros(total, dtype=np.uint8)
    for cw in codewords:
        c_idx = _tuple_to_idx(cw, n, q)
        x[c_idx] = 1
    
    # Current coverage state
    covered_count = np.zeros(total, dtype=np.int32)  # how many selected codewords cover each word
    for c in range(total):
        if x[c]:
            for w in forward_cover[c]:
                covered_count[w] += 1
    
    uncovered = total - int(np.sum(covered_count > 0))
    num_selected = int(np.sum(x))
    
    # Energy parameters
    alpha = 100.0  # heavy penalty for uncovered words
    beta = 1.0     # light penalty for number of selected codewords
    
    def _energy(uncovered, num_sel):
        return alpha * uncovered + beta * num_sel
    
    current_energy = _energy(uncovered, num_selected)
    best_energy = current_energy
    best_x = x.copy()
    
    # Annealing schedule
    T_start = 50.0
    T_end = 0.01
    n_iter = 50000  # target iterations, will stop early if time runs out
    
    # Precompute: for each codeword c, the list of words it covers (for fast delta computation)
    # Already have forward_cover
    
    start_time = time.monotonic()
    for it in range(n_iter):
        # Check deadline
        if time.monotonic() - start_time > (max_time_remaining[1] - max_time_remaining[0]) - 2.0:
            break
        
        # Temperature
        progress = it / n_iter
        T = T_start * math.exp(-10 * progress)
        
        # Random move: flip a random bit
        c = int(rng.integers(0, total))
        
        if x[c] == 1:
            # Remove codeword c
            # Delta: for each word w that c covers, if covered_count[w] becomes 0, uncovered increases by 1
            new_uncovered_delta = 0
            for w in forward_cover[c]:
                if covered_count[w] == 1:  # only c covers this word
                    new_uncovered_delta += 1
            delta_energy = alpha * new_uncovered_delta - beta
        else:
            # Add codeword c
            # Delta: for each word w that c covers, if covered_count[w] was 0, uncovered decreases by 1
            new_uncovered_delta = 0
            for w in forward_cover[c]:
                if covered_count[w] == 0:
                    new_uncovered_delta += 1
            delta_energy = -alpha * new_uncovered_delta + beta
        
        # Acceptance criterion
        if delta_energy < 0 or rng.random() < math.exp(-delta_energy / max(T, 1e-10)):
            # Apply move
            if x[c] == 1:
                x[c] = 0
                for w in forward_cover[c]:
                    covered_count[w] -= 1
                uncovered += new_uncovered_delta
                num_selected -= 1
            else:
                x[c] = 1
                for w in forward_cover[c]:
                    covered_count[w] += 1
                uncovered -= new_uncovered_delta
                num_selected += 1
            
            current_energy = _energy(uncovered, num_selected)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
        
        # Early stop if perfect coverage achieved with minimal size
        if uncovered == 0:
            # Try to continue removing to minimize size
            pass
    
    # Extract codewords from best solution
    best_codewords = []
    for c in range(total):
        if best_x[c]:
            best_codewords.append(_idx_to_tuple(c, n, q))
    
    return best_codewords


def _local_search(codewords, q, n, radius, rng, max_time_remaining):
    """
    Try to remove codewords one at a time if coverage is maintained.
    """
    codewords = list(codewords)
    total = q ** n
    forward_cover, reverse_cover = _build_coverage_matrix(total, n, q, radius)
    
    def _coverage_count(cw_list):
        covered = np.zeros(total, dtype=np.uint8)
        for cw in cw_list:
            c_idx = _tuple_to_idx(cw, n, q)
            for w in forward_cover[c_idx]:
                covered[w] = 1
        return int(np.sum(covered))
    
    # Phase 1: try to remove each codeword
    improved = True
    while improved and _elapsed(max_time_remaining[0]) < max_time_remaining[1]:
        improved = False
        for i in range(len(codewords)):
            if _elapsed(max_time_remaining[0]) > max_time_remaining[1]:
                return codewords
            test_list = codewords[:i] + codewords[i+1:]
            if _coverage_count(test_list) == total:
                codewords = test_list
                improved = True
                break
    
    return codewords


def construct(q, n, radius):
    start = time.monotonic()
    deadline = start + _DEADLINE_SECONDS
    max_time_remaining = [start, deadline]

    rng = np.random.default_rng(42)

    # Start with a greedy construction
    best = _greedy_construct(q, n, radius, rng, max_time_remaining)
    
    # Simulated annealing to improve
    if _elapsed(start) < deadline - 10.0:
        sa_result = _simulated_annealing(best, q, n, radius, rng, max_time_remaining)
        if len(sa_result) < len(best):
            best = sa_result
    
    # Final local search to polish
    if _elapsed(start) < deadline - 3.0:
        best = _local_search(best, q, n, radius, rng, max_time_remaining)

    # Verify coverage
    if not _is_cover(best, q, n, radius):
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(cw) for cw in best]
# EVOLVE-BLOCK-END
