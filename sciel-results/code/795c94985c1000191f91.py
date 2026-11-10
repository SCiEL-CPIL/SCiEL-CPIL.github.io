"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
from collections import deque

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    Returns a list of codewords (each a list of n integers).
    """
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    # Trivial cases
    if n == 0:
        return [[]]
    if radius >= n:
        return [[0] * n]
    if n == 1:
        return [[i] for i in range(q)]

    # For q=3, n=8, radius=1, we aim to beat the direct sum construction (486 words).
    # Strategy:
    # 1. Generate a large set of candidate codewords using a structured greedy approach.
    # 2. Use local search (simulated annealing or greedy removal) to minimize the code size.

    # Step 1: Generate initial code using a greedy set cover approach with randomization
    # But for speed, we'll use a structured heuristic.

    # For ternary codes, we can use the fact that the tetracode is a perfect code of length 4, radius 1.
    # Direct sum gives 9 * 3^(n-4) = 486 for n=8.

    # To beat this, we need a more efficient covering.
    # We'll use a two-phase approach:
    # Phase 1: Generate a superset of codewords using a greedy algorithm with random perturbations.
    # Phase 2: Greedily remove redundant codewords.

    # For efficiency, we'll use numpy for distance calculations.

    # Precompute all possible words? No, 3^8 = 6561, which is manageable.
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    N = len(all_words)

    # Initialize with a direct sum construction as a baseline
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    initial_code = np.array(initial_code, dtype=np.int8)

    # Greedy removal: repeatedly remove a codeword if its coverage is still covered by others.
    # This is O(m^2 * n) where m is the number of codewords, which might be slow for 486.
    # Instead, we'll use a smarter approach.

    # Let's try a different strategy: use a greedy set cover from scratch with randomization.
    # But that might be slow.

    # Alternative: Use the fact that we can improve the direct sum by "mixing" the tetracode with the tail.
    # Specifically, we can use a non-linear code that covers more efficiently.

    # For now, let's implement a greedy removal on the initial code.
    # We'll represent coverage as a boolean array of size N.

    def compute_coverage(codewords):
        """Compute the coverage of a set of codewords."""
        if len(codewords) == 0:
            return np.zeros(N, dtype=bool)
        # For each codeword, find all words within radius.
        # This is O(m * N * n) which is 486 * 6561 * 8 ≈ 25 million operations, which is okay in numpy.
        coverage = np.zeros(N, dtype=bool)
        for cw in codewords:
            # Compute Hamming distance from cw to all words
            dists = np.sum(all_words != cw, axis=1)
            coverage |= (dists <= radius)
        return coverage

    def is_covered(coverage):
        """Check if all words are covered."""
        return np.all(coverage)

    # Start with the initial code
    current_code = initial_code.copy()
    current_coverage = compute_coverage(current_code)

    # Greedy removal
    # We'll try to remove codewords in a random order, checking if coverage is maintained.
    # To speed this up, we'll precompute the "exclusive" coverage of each codeword.

    # For each codeword i, exclusive coverage = coverage by i but not by any other codeword.
    # If exclusive coverage is empty, we can remove i.

    # This is O(m^2 * N * n) which is too slow.
    # Instead, we'll use a simpler approach: try removing each codeword and check coverage.

    # Let's use a priority queue or just iterate.
    # Since m=486, and N=6561, checking coverage for one removal is O(N*n) = 52k operations.
    # 486 * 52k ≈ 25 million, which is okay.

    # But we want to do multiple passes.

    # Let's implement a greedy removal with multiple passes.
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 10:
        improved = False
        # Shuffle the order of codewords to avoid bias
        order = np.random.permutation(len(current_code))
        for i in order:
            if (time.time() - start_time) > DEADLINE - 5:
                break
            # Try removing codeword i
            new_code = np.delete(current_code, i, axis=0)
            new_coverage = compute_coverage(new_code)
            if is_covered(new_coverage):
                current_code = new_code
                current_coverage = new_coverage
                improved = True

    # Convert to list of lists
    return [list(cw) for cw in current_code]
# EVOLVE-BLOCK-END
