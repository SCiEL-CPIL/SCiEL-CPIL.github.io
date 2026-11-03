"""Covering code K_3(6,1) solver: greedy init + simulated annealing removal."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius `radius`.
    Returns a list of codewords (each a list of n ints in 0..q-1).
    Optimized for q=3, n=6, radius=1 (the football pool problem).
    """
    deadline = time.time() + 110.0  # safety margin before 120s kill
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)

    # --- Fast path: exact known optimum for K_3(6,1) is 73.
    #     We will try to find 73 via heuristic search.
    
    total_words = q ** n
    # Map each word to an integer index 0..total_words-1
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    # Precompute neighbors within radius 1 for each word
    # For speed, precompute a list of neighbor indices for each index
    # But 729 * (1 + 6*2) = 729 * 13 = 9477 entries, manageable
    # Actually, for radius 1 in q=3, n=6: each word has 1 + 6*2 = 13 neighbors (including itself)
    # Let's precompute a coverage bitset or just a list of covered indices per codeword.
    
    # For efficiency, represent the set of covered words as a Python set of indices.
    # But for 729 words, we can use a numpy boolean array of size 729.
    
    covered = np.zeros(total_words, dtype=bool)
    
    # Greedy initialization: pick codewords that cover the most uncovered words
    # Start with an empty set, repeatedly add the word that covers the most uncovered.
    
    # Precompute for each index, the list of indices it covers (within radius 1)
    covers = []
    for i in range(total_words):
        word = all_words[i]
        neigh = set()
        # Self
        neigh.add(i)
        # All words at Hamming distance 1
        for pos in range(n):
            for val in range(q):
                if val == word[pos]:
                    continue
                new_word = word[:]
                new_word[pos] = val
                neigh.add(word_to_idx(new_word))
        covers.append(sorted(neigh))
    
    # Greedy set cover
    codewords = []
    uncovered_count = np.ones(total_words, dtype=int)
    
    # For greedy, we want to pick the codeword that covers the most currently uncovered words.
    # This is O(729 * 729) = ~530k operations per step, and we need ~73 steps, so ~38M ops. Fine.
    
    while not covered.all():
        if time.time() > deadline - 5:
            break
        best_idx = -1
        best_gain = -1
        for i in range(total_words):
            if i in codewords:
                continue
            # Count how many of covers[i] are not yet covered
            gain = 0
            for j in covers[i]:
                if not covered[j]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1 or best_gain == 0:
            break
        codewords.append(best_idx)
        for j in covers[best_idx]:
            covered[j] = True
    
    # Convert to list of words
    def idxs_to_words(idxs):
        return [all_words[i] for i in idxs]
    
    current = list(codewords)
    best = list(current)
    best_size = len(current)
    
    # Simulated annealing: try to remove codewords or swap to reduce size while maintaining coverage
    # We'll do removal-based SA: try removing a random codeword, check if coverage is maintained.
    
    def is_covered(cw_idxs):
        """Check if the set of codeword indices covers all words."""
        cov = np.zeros(total_words, dtype=bool)
        for i in cw_idxs:
            for j in covers[i]:
                cov[j] = True
        return cov.all()
    
    def coverage_gaps(cw_idxs):
        """Return set of uncovered indices."""
        cov = np.zeros(total_words, dtype=bool)
        for i in cw_idxs:
            for j in covers[i]:
                cov[j] = True
        return np.where(~cov)[0]
    
    # Verify initial coverage
    if not is_covered(current):
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
    
    temp = 1.0
    cooling = 0.999
    min_temp = 0.01
    
    while temp > min_temp and time.time() < deadline:
        improved = False
        # Try random removals
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            if len(current) <= best_size:
                # Try to remove
                idx_to_remove = rng.randrange(len(current))
            trial = current[:]
            trial.pop(idx_to_remove)
            if is_covered(trial):
                current = trial
                if len(current) < best_size:
                    best = list(current)
                    best_size = len(current)
                improved = True
                break
        # Try random swaps: remove one, add another
        if not improved and len(current) > 10:
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                i_remove = rng.randrange(len(current))
                # Pick a random word not in current to add
                trial = current[:]
                trial.pop(i_remove)
                # Find a word that covers the gaps
                gaps = coverage_gaps(trial)
                if len(gaps) == 0:
                    current = trial
                    if len(current) < best_size:
                        best = list(current)
                        best_size = len(current)
                    break
                # Find a candidate that covers the most gaps
                best_add = -1
                best_gap_covered = -1
                # Sample some candidates
                candidates = rng.sample(range(total_words), min(100, total_words))
                for c in candidates:
                    if c in trial:
                        continue
                    gap_covered = 0
                    for j in covers[c]:
                        if not covered[j]:  # This is stale, need to recompute
                            pass
                    # Recompute properly
                    cov_temp = np.zeros(total_words, dtype=bool)
                    for i in trial:
                        for j in covers[i]:
                            cov_temp[j] = True
                    # Count how many gaps this candidate covers
                    for j in covers[c]:
                        if not cov_temp[j]:
                            gap_covered += 1
                    if gap_covered > best_gap_covered:
                        best_gap_covered = gap_covered
                        best_add = c
                if best_add >= 0 and best_gap_covered == len(gaps):
                    trial.append(best_add)
                    current = trial
                    if len(current) < best_size:
                        best = list(current)
                        best_size = len(current)
                    break
        temp *= cooling
    
    # Return the best found
    return idxs_to_words(best)
# EVOLVE-BLOCK-END
