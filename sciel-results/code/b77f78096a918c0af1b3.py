"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random


def _precompute_cover_matrix(q, n, radius):
    """Precompute a (total_words, total_words) boolean matrix where
    cover_matrix[i,j] = True if word j is within Hamming distance `radius`
    of word i. Returns (cover_matrix, total_words, word_to_idx, idx_to_word)."""
    total = q ** n
    # Enumerate all words as base-q integers
    words = []
    for idx in range(total):
        w = []
        x = idx
        for _ in range(n):
            w.append(x % q)
            x //= q
        w.reverse()
        words.append(tuple(w))
    
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors for radius 1 (most common case)
    if radius == 1:
        # For each word, compute list of words within distance 1
        # A word covers itself + n*(q-1) neighbors
        cover_sets = []
        for w in words:
            covered = [w]
            for pos in range(n):
                for val in range(q):
                    if val != w[pos]:
                        nw = list(w)
                        nw[pos] = val
                        covered.append(tuple(nw))
            cover_sets.append(covered)
        
        cover_matrix = np.zeros((total, total), dtype=bool)
        for i, covered in enumerate(cover_sets):
            for j_word in covered:
                cover_matrix[i, word_to_idx[j_word]] = True
        
        return cover_matrix, total, word_to_idx, idx_to_word(words)
    else:
        # General case (slower)
        cover_matrix = np.zeros((total, total), dtype=bool)
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                if sum(a != b for a, b in zip(w1, w2)) <= radius:
                    cover_matrix[i, j] = True
        return cover_matrix, total, word_to_idx, idx_to_word(words)


def idx_to_word(words):
    return words


def _greedy_cover_vectorized(cover_matrix, total, deadline, random_seed=0):
    """Greedy set cover using vectorized numpy operations.
    Returns list of codeword indices."""
    rng = random.Random(random_seed)
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    while not covered.all() and time.time() < deadline - 5:
        # Find candidate that covers most uncovered words
        # new_cover_count[i] = number of uncovered words covered by codeword i
        new_cover_count = cover_matrix[:, ~covered].sum(axis=1) if (~covered).any() else np.zeros(total, dtype=int)
        
        # Among those with max count, pick one (with some randomness for diversity)
        max_count = new_cover_count.max()
        if max_count == 0:
            break
        candidates = np.where(new_cover_count == max_count)[0]
        best_idx = int(rng.choice(candidates))
        
        codewords.append(best_idx)
        covered[cover_matrix[best_idx]] = True
    
    return codewords


def _remove_redundant_vectorized(cover_matrix, codewords, deadline):
    """Remove redundant codewords using vectorized coverage check."""
    if not codewords:
        return codewords
    
    n_cw = len(codewords)
    # coverage[i] = set of words covered only by codeword i (not by others)
    # We can compute: for each word, how many codewords cover it
    # A codeword is redundant if no word is covered only by it
    
    # Count coverage: for each word in the space, count how many selected codewords cover it
    # covered_count[w] = number of codewords that cover word w
    covered_count = np.zeros(len(cover_matrix), dtype=np.int32)
    for cw_idx in codewords:
        covered_count[cover_matrix[cw_idx]] += 1
    
    # A codeword i is redundant if for all words w it covers, covered_count[w] >= 2
    # i.e., every word it covers is also covered by at least one other codeword
    
    i = 0
    while i < len(codewords) and time.time() < deadline - 3:
        cw = codewords[i]
        # Check if all words covered by cw are covered by >= 2 codewords
        covered_by_cw = cover_matrix[cw]
        if (covered_count[covered_by_cw] >= 2).all():
            # Safe to remove
            covered_count[covered_by_cw] -= 1
            codewords.pop(i)
        else:
            i += 1
    
    return codewords


def _local_search_1for1(cover_matrix, codewords, total, deadline):
    """Try 1-for-1 swaps: replace one codeword with another to improve (reduce) size.
    Actually, 1-for-1 keeps same size. We use this to find better positions for
    subsequent removals."""
    if not codewords:
        return codewords
    
    # Compute coverage counts
    covered_count = np.zeros(total, dtype=np.int32)
    for cw_idx in codewords:
        covered_count[cover_matrix[cw_idx]] += 1
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(codewords)):
            if time.time() > deadline - 5:
                break
            cw_i = codewords[i]
            # For each potential replacement, check if removing cw_i and adding candidate
            # still covers everything
            # Words uniquely covered by cw_i: those where covered_count == 1
            unique_to_i = cover_matrix[cw_i] & (covered_count == 1)
            if not unique_to_i.any():
                # Already redundant, should have been removed
                continue
            
            # We need a candidate that covers all words in unique_to_i
            # and doesn't introduce new uncovered words
            # Simpler: just try to find if any candidate covers unique_to_i
            # For speed, limit candidates
            # Find candidates that cover all of unique_to_i
            # This is expensive; do it for a subset
            unique_indices = np.where(unique_to_i)[0]
            if len(unique_indices) == 0:
                continue
            
            # Check a few random candidates
            for _ in range(20):
                candidate = np.random.randint(total)
                if candidate in codewords:
                    continue
                # Does candidate cover all unique_to_i?
                if cover_matrix[candidate, unique_indices].all():
                    # Also check: after removing cw_i and adding candidate,
                    # is everything still covered?
                    # Words that were covered only by cw_i must be covered by candidate
                    # Words that candidate covers but cw_i didn't: need to check they're still covered
                    # Actually we need: for all words w, if covered_count[w] was 1 (only by cw_i),
                    # then candidate must cover w. And for words candidate covers but cw_i didn't,
                    # they need to be covered by someone else OR by candidate.
                    # Simpler check: new coverage = (old coverage - coverage of cw_i) + coverage of candidate
                    # This is valid if: every word in (coverage of cw_i) \ (coverage of candidate)
                    # is covered by at least one other codeword.
                    # And every word in (coverage of candidate) \ (coverage of cw_i) 
                    # was already covered by some codeword (or is now covered by candidate).
                    # The second condition is automatically satisfied since candidate covers them.
                    # So we just need: (coverage of cw_i) \ (coverage of candidate) ⊆ (coverage of others)
                    # i.e., for words covered by cw_i but not candidate, covered_count >= 2 (before removal)
                    # Wait, covered_count includes cw_i. So we need covered_count >= 2 for those words.
                    diff = cover_matrix[cw_i] & ~cover_matrix[candidate]
                    if diff.any():
                        if (covered_count[diff] >= 2).all():
                            # Valid swap
                            covered_count[cover_matrix[cw_i]] -= 1
                            covered_count[cover_matrix[candidate]] += 1
                            codewords[i] = candidate
                            improved = True
                            break
    
    return codewords


def _local_search_2for1(cover_matrix, codewords, total, deadline):
    """Try 2-for-1: remove two codewords, add one. Reduces size by 1."""
    if len(codewords) < 3:
        return codewords
    
    covered_count = np.zeros(total, dtype=np.int32)
    for cw_idx in codewords:
        covered_count[cover_matrix[cw_idx]] += 1
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        n_cw = len(codewords)
        # Try random pairs
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            i, j = random.sample(range(n_cw), 2)
            cw_i, cw_j = codewords[i], codewords[j]
            
            # Words uniquely covered by cw_i or cw_j (count == 1)
            # After removing both, words with count 1 become uncovered
            # We need a candidate that covers all such words
            # Count after removing both:
            temp_count = covered_count.copy()
            temp_count[cover_matrix[cw_i]] -= 1
            temp_count[cover_matrix[cw_j]] -= 1
            
            # Words that would become uncovered
            would_uncover = (temp_count == 0)
            if not would_uncover.any():
                # Just remove both, no need to add
                covered_count[cover_matrix[cw_i]] -= 1
                covered_count[cover_matrix[cw_j]] -= 1
                # Remove the larger index first to keep indices valid
                if i > j:
                    i, j = j, i
                codewords.pop(j)
                codewords.pop(i)
                improved = True
                break
            
            # Find candidate that covers all would_uncover words
            uncover_indices = np.where(would_uncover)[0]
            # Check if any single codeword covers all of them
            # This is expensive; try random candidates
            for _ in range(50):
                candidate = np.random.randint(total)
                if candidate in codewords:
                    continue
                if cover_matrix[candidate, uncover_indices].all():
                    # Check: after removing i,j and adding candidate, all covered
                    # Words covered by i or j but not candidate must be covered by others
                    removed_cov = cover_matrix[cw_i] | cover_matrix[cw_j]
                    not_in_candidate = removed_cov & ~cover_matrix[candidate]
                    if not_in_candidate.any():
                        if (temp_count[not_in_candidate] >= 1).all():
                            # Valid
                            covered_count[cover_matrix[cw_i]] -= 1
                            covered_count[cover_matrix[cw_j]] -= 1
                            covered_count[cover_matrix[candidate]] += 1
                            # Remove i, j, add candidate
                            if i > j:
                                i, j = j, i
                            codewords.pop(j)
                            codewords.pop(i)
                            codewords.append(candidate)
                            improved = True
                            break
                if improved:
                    break
    
    return codewords


def _solve_ternary_7_1(deadline):
    """Solve the specific ternary (3,7,1) covering code problem."""
    cover_matrix, total, word_to_idx, idx_to_word = _precompute_cover_matrix(3, 7, 1)
    
    best_codewords = None
    best_size = float('inf')
    
    # Multiple greedy starts with different random seeds
    num_starts = 15
    start_budget = (deadline - 10) / num_starts
    
    for seed in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        t_start = time.time()
        t_end = t_start + start_budget * 0.4  # Time for greedy
        
        codewords = _greedy_cover_vectorized(cover_matrix, total, t_end, random_seed=seed)
        
        if not codewords:
            continue
        
        # Remove redundant
        t_rem = time.time() + start_budget * 0.3
        codewords = _remove_redundant_vectorized(cover_matrix, codewords, t_rem)
        
        # Local search 1-for-1
        t_ls1 = time.time() + start_budget * 0.2
        codewords = _local_search_1for1(cover_matrix, codewords, total, t_ls1)
        
        # Local search 2-for-1
        t_ls2 = time.time() + start_budget * 0.1
        codewords = _local_search_2for1(cover_matrix, codewords, total, t_ls2)
        
        # Final redundancy removal
        codewords = _remove_redundant_vectorized(cover_matrix, codewords, deadline - 5)
        
        # Verify coverage
        covered = np.zeros(total, dtype=bool)
        for cw in codewords:
            covered[cover_matrix[cw]] = True
        if covered.all():
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
    
    # Use remaining time for more local search on best solution
    if best_codewords is not None and time.time() < deadline - 3:
        remaining_deadline = deadline - 3
        # More aggressive local search
        for _ in range(50):
            if time.time() > remaining_deadline:
                break
            old_size = len(best_codewords)
            best_codewords = _local_search_2for1(cover_matrix, best_codewords, total, remaining_deadline)
            best_codewords = _local_search_1for1(cover_matrix, best_codewords, total, remaining_deadline)
            best_codewords = _remove_redundant_vectorized(cover_matrix, best_codewords, remaining_deadline)
            if len(best_codewords) < old_size:
                continue
            else:
                break
    
    if best_codewords is None:
        # Fallback: pure greedy
        codewords = _greedy_cover_vectorized(cover_matrix, total, deadline - 2, random_seed=0)
        return [list(idx_to_word[cw]) for cw in codewords]
    
    return [list(idx_to_word[cw]) for cw in best_codewords]


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius
    'radius'. Uses vectorized greedy set-cover with local search optimization.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(deadline)
    
    # General fallback: greedy set cover (slower, for other parameters)
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    def get_coverage_general(codeword):
        covered = set()
        covered.add(codeword)
        for i in range(n):
            for val in range(q):
                if val != codeword[i]:
                    new_word = list(codeword)
                    new_word[i] = val
                    covered.add(tuple(new_word))
        return covered
    
    codewords = []
    covered = set()
    
    while time.time() < deadline - 10:
        best_cw = None
        best_count = -1
        best_new_covered = set()
        
        for candidate in all_words:
            new_covered = get_coverage_general(candidate) - covered
            count = len(new_covered)
            if count > best_count:
                best_count = count
                best_cw = candidate
                best_new_covered = new_covered
        
        if best_count <= 0:
            break
        
        codewords.append(best_cw)
        covered.update(best_new_covered)
        
        if len(covered) == total:
            break
    
    # Redundancy removal
    i = 0
    while i < len(codewords) and time.time() < deadline - 5:
        cw_to_remove = codewords[i]
        coverage_of_cw = get_coverage_general(cw_to_remove)
        
        can_remove = True
        for w in coverage_of_cw:
            is_covered_by_other = False
            for j, other_cw in enumerate(codewords):
                if i == j:
                    continue
                    if w in get_coverage_general(other_cw):
                        is_covered_by_other = True
                        break
            if not is_covered_by_other:
                can_remove = False
                break
        
        if can_remove:
            codewords.pop(i)
        else:
            i += 1
    
    return [list(cw) for cw in codewords]
# EVOLVE-BLOCK-END
