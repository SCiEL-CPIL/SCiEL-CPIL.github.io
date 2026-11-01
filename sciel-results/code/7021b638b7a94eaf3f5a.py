"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

# Precomputed optimal solution for K_3(6,1) if we can find it
# The best known is 73 words (van Laarhoven, Aarts, van Lint and Wille, 1989)
# Let's try to construct a good solution using a greedy approach with local search


def _all_words(q, n):
    """Generate all words of length n over alphabet {0,...,q-1}."""
    return list(itertools.product(range(q), repeat=n))


def _hamming_distance(w1, w2):
    """Compute Hamming distance between two words."""
    return sum(a != b for a, b in zip(w1, w2))


def _neighbors_within_radius(word, q, n, radius):
    """Generate all words within Hamming distance radius of the given word."""
    if radius == 0:
        return [list(word)]
    
    result = [list(word)]
    for r in range(1, radius + 1):
        # Generate all words at exactly Hamming distance r
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(q), repeat=r):
                new_word = list(word)
                for pos, val in zip(positions, values):
                    if val != word[pos]:
                        new_word[pos] = val
                if list(new_word) not in result:
                    result.append(list(new_word))
    return result


def _greedy_cover(q, n, radius, time_limit=100, rng=None):
    """
    Greedy set cover to find a small covering code.
    Returns a list of codewords.
    """
    if rng is None:
        rng = random.Random(42)
    
    start_time = time.time()
    all_words = _all_words(q, n)
    n_words = len(all_words)
    
    # Create a set of uncovered words
    uncovered = set(all_words)
    
    # Precompute coverage for each possible codeword
    # coverage[codeword] = set of words it covers
    coverage = {}
    for word in all_words:
        covered = set(_neighbors_within_radius(word, q, n, radius))
        coverage[word] = covered
    
    # Greedy selection
    selected = []
    while uncovered and (time.time() - start_time) < time_limit:
        # Find the codeword that covers the most uncovered words
        best_word = None
        best_count = -1
        for word in all_words:
            count = len(set(coverage[word]) & uncovered)
            if count > best_count:
                best_count = count
                best_word = word
                if count == len(uncovered):
                    break
        
        if best_word is None or best_count <= 0:
            break
        
        selected.append(list(best_word))
        uncovered -= coverage[best_word]
    
    return selected


def _local_search_improve(codewords, q, n, radius, time_limit=50, rng=None):
    """
    Try to improve the solution by removing or swapping codewords.
    """
    if rng is None:
        rng = random.Random(42)
    
    start_time = time.time()
    current = [list(c) for c in codewords]
    
    # Precompute coverage
    all_words = _all_words(q, n)
    coverage = {}
    for word in all_words:
        coverage[word] = set(_neighbors_within_radius(word, q, n, radius))
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try removing a codeword if the remaining still cover everything
        for i in range(len(current)):
            if (time.time() - start_time) >= time_limit:
                break
            
            # Check if removing current[i] still covers all words
            remaining = current[:i] + current[i+1:]
            if not remaining:
                continue
            
            # Check coverage
            covered = set()
            for cw in remaining:
                covered |= coverage[cw]
            
            if len(covered) == len(all_words):
                current = remaining
                improved = True
                break
        
        # Try swapping a codeword with another that covers more uncovered
        if not improved:
            # Find uncovered words (should be none if valid, but just in case)
            covered = set()
            for cw in current:
                covered |= coverage[cw]
            
            uncovered = set(all_words) - covered
            if uncovered:
                # Find a better replacement
                for i in range(len(current)):
                    if (time.time() - start_time) >= time_limit:
                        break
                    
                    current_without_i = current[:i] + current[i+1:]
                    covered_without_i = set()
                    for cw in current_without_i:
                        covered_without_i |= coverage[cw]
                    
                    still_uncovered = set(all_words) - covered_without_i
                    if not still_uncovered:
                        continue
                    
                    # Find a word that covers many of still_uncovered
                    best_word = None
                    best_count = -1
                    for word in all_words:
                        if word in current:
                            continue
                        count = len(coverage[word] & still_uncovered)
                        if count > best_count:
                            best_count = count
                            best_word = word
                    
                    if best_word is not None and best_count > 0:
                        new_current = current_without_i + [list(best_word)]
                        # Verify it's valid
                        new_covered = set()
                        for cw in new_current:
                            new_covered |= coverage[cw]
                        if len(new_covered) == len(all_words) and len(new_current) <= len(current):
                            current = new_current
                            improved = True
                            break
    
    return current


def _verify_solution(codewords, q, n, radius):
    """Verify that the codewords form a valid covering code."""
    all_words = _all_words(q, n)
    coverage = {}
    for word in all_words:
        coverage[word] = set(_neighbors_within_radius(word, q, n, radius))
    
    covered = set()
    for cw in codewords:
        covered |= coverage[cw]
    
    return len(covered) == len(all_words)


def _construct_optimal_q3_n6_r1():
    """
    Try to construct a 73-word covering code for K_3(6,1).
    Uses a known construction or heuristic.
    """
    # Start with a greedy solution
    q, n, radius = 3, 6, 1
    initial = _greedy_cover(q, n, radius, time_limit=30)
    
    # Improve with local search
    improved = _local_search_improve(initial, q, n, radius, time_limit=30)
    
    return improved


def construct(q, n, radius):
    """
    Construct a covering code: a list of codewords such that every word
    in {0,...,q-1}^n is within Hamming distance radius of some codeword.
    Minimizes the number of codewords.
    """
    start_time = time.time()
    time_limit = 110  # Leave some margin before the 120s limit
    
    # Special case: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        # Try to get a good solution
        solution = _construct_optimal_q3_n6_r1()
        
        # Verify it's valid
        if not _verify_solution(solution, q, n, radius):
            # Fallback to direct sum construction
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            solution = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        return solution
    
    # General case: use greedy + local search
    if (time.time() - start_time) > time_limit:
        # Fallback
        return [list(range(n))]
    
    initial = _greedy_cover(q, n, radius, time_limit=min(60, time_limit - (time.time() - start_time)))
    
    if (time.time() - start_time) < time_limit - 10:
        improved = _local_search_improve(initial, q, n, radius, time_limit=min(30, time_limit - (time.time() - start_time)))
        if _verify_solution(improved, q, n, radius):
            return improved
    
    return initial
# EVOLVE-BLOCK-END
