"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _tetracode_direct_sum(q, n):
    """Generate the tetracode direct-sum construction (81 words for q=3, n=6)."""
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _is_covering(code, q, n, radius):
    """Check if the code covers all words within Hamming distance <= radius."""
    all_words = set(itertools.product(range(q), repeat=n))
    covered = set()
    for cw in code:
        # Add all words within radius of cw
        for d in range(radius + 1):
            for positions in itertools.combinations(range(n), d):
                for values in itertools.product(range(q), repeat=d):
                    word = list(cw)
                    for pos, val in zip(positions, values):
                        word[pos] = val
                    if sum(1 for i in range(n) if word[i] != cw[i]) == d:
                        covered.add(tuple(word))
    return covered == all_words


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    if q == 3 and n == 6 and radius == 1:
        # Start with tetracode direct sum (81 words)
        base_code = _tetracode_direct_sum(q, n)
        current_code = [list(cw) for cw in base_code]
        best_code = [list(cw) for cw in current_code]
        best_size = len(best_code)
        
        # Precompute neighborhood masks for fast distance checking
        # For each word, precompute its neighbors (distance <= 1)
        # There are 729 words total
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighborhood for each possible word
        # neighborhood[i] = set of indices of words within distance <= radius from word i
        neighborhoods = [set() for _ in range(q ** n)]
        for i, word in enumerate(all_words):
            neighborhoods[i].add(i)
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val != orig:
                        new_word = list(word)
                        new_word[pos] = val
                        neighborhoods[i].add(word_to_idx[tuple(new_word)])
        
        def is_covering_fast(code):
            """Fast check using precomputed neighborhoods."""
            covered = set()
            for cw in code:
                idx = word_to_idx[tuple(cw)]
                covered.update(neighborhoods[idx])
            return len(covered) == q ** n
        
        def count_uncovered_fast(code):
            """Count uncovered words using precomputed neighborhoods."""
            covered = set()
            for cw in code:
                idx = word_to_idx[tuple(cw)]
                covered.update(neighborhoods[idx])
            return q ** n - len(covered)
        
        # Simulated Annealing
        # Move: remove a random codeword, then add a random word to cover newly uncovered
        T = 50.0
        T_min = 0.1
        cooling = 0.98
        max_iterations = 50000
        
        rng = random.Random(42)
        
        current_size = len(current_code)
        current_uncovered = count_uncovered_fast(current_code)
        
        iteration = 0
        while iteration < max_iterations and time.time() < deadline - 10.0:
            iteration += 1
            
            # Try removing a random codeword
            if current_size <= 73:
                # Only try removals if we're at or below target
                idx_to_remove = rng.randrange(current_size)
                candidate_code = [cw for i, cw in enumerate(current_code) if i != idx_to_remove]
                
                if is_covering_fast(candidate_code):
                    # Accept: valid cover with fewer words
                    current_code = candidate_code
                    current_size = len(current_code)
                    current_uncovered = 0
                    if current_size < best_size:
                        best_code = [list(cw) for cw in current_code]
                        best_size = current_size
                        T = max(T * cooling, T_min)
                else:
                    # Reject or accept with probability
                    new_uncovered = count_uncovered_fast(candidate_code)
                    delta = new_uncovered - current_uncovered
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        current_code = candidate_code
                        current_size = len(current_code)
                        current_uncovered = new_uncovered
                        if current_size < best_size and is_covering_fast(current_code):
                            best_code = [list(cw) for cw in current_code]
                            best_size = current_size
                T = max(T * cooling, T_min)
            
            # Also try swapping: remove one, add another
            elif current_size > 73:
                idx_to_remove = rng.randrange(current_size)
                removed_word = current_code[idx_to_remove]
                
                # Find a word that covers the most newly uncovered words
                # After removal, find uncovered words
                temp_code = [cw for i, cw in enumerate(current_code) if i != idx_to_remove]
                covered_after_removal = set()
                for cw in temp_code:
                    idx = word_to_idx[tuple(cw)]
                    covered_after_removal.update(neighborhoods[idx])
                
                uncovered_after_removal = set(range(q ** n)) - covered_after_removal
                
                if not uncovered_after_removal:
                    # Still covers everything without the removed word
                    current_code = temp_code
                    current_size = len(current_code)
                    current_uncovered = 0
                    if current_size < best_size:
                        best_code = [list(cw) for cw in current_code]
                        best_size = current_size
                    T = max(T * cooling, T_min)
                else:
                    # Try to find a single word that covers all uncovered
                    # For each candidate word, check if it covers all uncovered
                    best_candidate = None
                    best_coverage = -1
                    for u_idx in list(uncovered_after_removal)[:50]:  # Sample
                        u_word = all_words[u_idx]
                        # Check if adding u_word covers all uncovered
                        new_covered = set(neighborhoods[u_idx])
                        if uncovered_after_removal.issubset(new_covered):
                            # This word covers all uncovered
                            candidate_code = temp_code + [list(u_word)]
                            if len(candidate_code) < current_size:
                                current_code = candidate_code
                                current_size = len(current_code)
                                current_uncovered = 0
                                if current_size < best_size:
                                    best_code = [list(cw) for cw in current_code]
                                    best_size = current_size
                                break
                    else:
                        # Try random swaps
                        for _ in range(10):
                            random_word_idx = rng.randrange(q ** n)
                            random_word = all_words[random_word_idx]
                            new_covered = set(neighborhoods[random_word_idx])
                            if uncovered_after_removal.issubset(new_covered):
                                candidate_code = temp_code + [list(random_word)]
                                if len(candidate_code) <= current_size:
                                    current_code = candidate_code
                                    current_size = len(current_code)
                                    current_uncovered = 0
                                    if current_size < best_size and is_covering_fast(current_code):
                                        best_code = [list(cw) for cw in current_code]
                                        best_size = current_size
                                    break
                    T = max(T * cooling, T_min)
        
        # Return the best valid cover found
        if is_covering_fast(best_code):
            return [list(cw) for cw in best_code]
        else:
            # Fallback to original
            return [list(cw) for cw in base_code]
    else:
        # Fallback: direct sum construction
        return _tetracode_direct_sum(q, n)
# EVOLVE-BLOCK-END
