"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _all_words(q, n):
    """Generate all q^n words as a numpy array of shape (q^n, n)."""
    total = q ** n
    words = np.zeros((total, n), dtype=np.int8)
    for i in range(total):
        v = i
        for j in range(n - 1, -1, -1):
            words[i, j] = v % q
            v //= q
    return words


def _neighbors_of_word(word, q, n):
    """Yield all words within Hamming distance 1 of the given word."""
    # The word itself
    yield tuple(word)
    for pos in range(n):
        orig = word[pos]
        for val in range(q):
            if val == orig:
                continue
            new_word = list(word)
            new_word[pos] = val
            yield tuple(new_word)


def _tetracode_direct_sum(q, n):
    """Generate the tetracode direct sum construction."""
    if n < 4:
        # Fallback: all words
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    result = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            result.append(h + list(tail))
    return result


def _build_coverage_map(codewords, q, n):
    """Build a coverage count array where cov[i] = number of codewords covering word i."""
    total = q ** n
    cov = np.zeros(total, dtype=np.int32)
    all_words = _all_words(q, n)
    
    # Map from tuple to index
    word_to_idx = {}
    for i in range(total):
        word_to_idx[tuple(all_words[i])] = i
    
    for cw in codewords:
        for nb in _neighbors_of_word(cw, q, n):
            idx = word_to_idx.get(nb)
            if idx is not None:
                cov[idx] += 1
    return cov, word_to_idx, all_words


def _is_valid_cover(codewords, q, n, radius=1):
    """Check if codewords form a valid cover."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    all_words = _all_words(q, n)
    
    word_to_idx = {}
    for i in range(total):
        word_to_idx[tuple(all_words[i])] = i
    
    for cw in codewords:
        for nb in _neighbors_of_word(cw, q, n):
            idx = word_to_idx.get(nb)
            if idx is not None:
                covered[idx] = True
    
    return np.all(covered)


def _greedy_add(codewords, q, n, radius=1):
    """Greedy set cover: add codewords that cover the most uncovered words."""
    total = q ** n
    all_words = _all_words(q, n)
    word_to_idx = {tuple(all_words[i]): i for i in range(total)}
    
    covered = np.zeros(total, dtype=bool)
    # Mark words covered by initial codewords
    for cw in codewords:
        for nb in _neighbors_of_word(cw, q, n):
            idx = word_to_idx.get(nb)
            if idx is not None:
                covered[idx] = True
    
    # Precompute coverage sets for all possible codewords
    all_candidates = list(itertools.product(range(q), repeat=n))
    candidate_indices = {tuple(c): i for i, c in enumerate(all_candidates)}
    
    while not np.all(covered):
        # Find candidate that covers most uncovered words
        best_gain = 0
        best_candidate = None
        for cand in all_candidates:
            cand_idx = candidate_indices[cand]
            gain = 0
            for nb in _neighbors_of_word(cand, q, n):
                idx = word_to_idx.get(nb)
                if idx is not None and not covered[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_candidate = cand
        
        if best_candidate is None or best_gain == 0:
            break
        
        codewords.append(list(best_candidate))
        for nb in _neighbors_of_word(best_candidate, q, n):
            idx = word_to_idx.get(nb)
            if idx is not None:
                covered[idx] = True
    
    return codewords


def _local_search(codewords, q, n, radius=1, max_iter=500, time_limit=30):
    """Local search: try swapping codewords to reduce count or improve coverage."""
    start_time = time.time()
    total = q ** n
    all_words = _all_words(q, n)
    word_to_idx = {tuple(all_words[i]): i for i in range(total)}
    
    best_cw = list(codewords)
    best_len = len(best_cw)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Try removing each codeword
        improved = False
        for i in range(len(best_cw)):
            trial = best_cw[:i] + best_cw[i+1:]
            if len(trial) < best_len and _is_valid_cover(trial, q, n, radius):
                best_cw = trial
                best_len = len(best_cw)
                improved = True
                break
        
        if not improved:
            # Try swapping: remove one, add another
            for i in range(len(best_cw)):
                for cand in random.sample(list(itertools.product(range(q), repeat=n)), min(50, total)):
                    trial = best_cw[:i] + best_cw[i+1:] + [list(cand)]
                    if len(trial) <= best_len and _is_valid_cover(trial, q, n, radius):
                        if len(trial) < best_len:
                            best_cw = trial
                            best_len = len(best_cw)
                            improved = True
                            break
                if improved:
                    break
    
    return best_cw


def _simulated_annealing(codewords, q, n, radius=1, time_limit=60):
    """Simulated annealing on binary inclusion vector."""
    start_time = time.time()
    total = q ** n
    all_words = _all_words(q, n)
    word_to_idx = {tuple(all_words[i]): i for i in range(total)}
    
    # Precompute coverage sets for all candidates
    all_candidates = list(itertools.product(range(q), repeat=n))
    n_candidates = len(all_candidates)
    
    # coverage_set[cand_idx] = list of word indices covered by candidate cand_idx
    coverage_sets = []
    for cand in all_candidates:
        covered_indices = []
        for nb in _neighbors_of_word(cand, q, n):
            idx = word_to_idx.get(nb)
            if idx is not None:
                covered_indices.append(idx)
        coverage_sets.append(covered_indices)
    
    # Current state: binary vector of length n_candidates
    current_included = np.zeros(n_candidates, dtype=bool)
    for cw in codewords:
        idx = all_candidates.index(tuple(cw))
        current_included[idx] = True
    
    # Compute coverage
    cov = np.zeros(total, dtype=np.int32)
    for i in range(n_candidates):
        if current_included[i]:
            for w_idx in coverage_sets[i]:
                cov[w_idx] += 1
    
    current_uncovered = np.sum(cov == 0)
    current_weight = np.sum(current_included)
    
    best_cw = list(codewords)
    best_score = current_weight
    
    T = 5.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        T *= cooling
        if T < 0.01:
            break
        
        # Pick a random move: toggle a random candidate
        move_type = random.random()
        if move_type < 0.5:
            # Toggle
            i = random.randint(0, n_candidates - 1)
            delta_weight = 1 if not current_included[i] else -1
            # Compute delta uncovered
            old_uncovered = current_uncovered
            if current_included[i]:
                # Removing
                new_uncovered = 0
                for w_idx in coverage_sets[i]:
                    if cov[w_idx] == 1:
                        new_uncovered += 1
                delta_uncovered = new_uncovered - old_uncovered
            else:
                # Adding
                new_uncovered = 0
                for w_idx in coverage_sets[i]:
                    if cov[w_idx] == 0:
                        new_uncovered -= 1
                delta_uncovered = new_uncovered - old_uncovered
            
            delta_score = delta_weight + 10 * delta_uncovered
            if delta_score <= 0 or random.random() < np.exp(-delta_score / T):
                current_included[i] = not current_included[i]
                for w_idx in coverage_sets[i]:
                    cov[w_idx] += delta_weight
                current_uncovered += delta_uncovered
                current_weight += delta_weight
                
                if current_uncovered == 0 and current_weight < best_score:
                    best_score = current_weight
                    best_cw = [list(all_candidates[i]) for i in range(n_candidates) if current_included[i]]
        
        else:
            # Try to remove a redundant codeword
            redundant = []
            for i in range(n_candidates):
                if current_included[i]:
                    is_redundant = True
                    for w_idx in coverage_sets[i]:
                        if cov[w_idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        redundant.append(i)
            
            if redundant:
                i = random.choice(redundant)
                current_included[i] = False
                for w_idx in coverage_sets[i]:
                    cov[w_idx] -= 1
                current_weight -= 1
                
                if current_uncovered == 0 and current_weight < best_score:
                    best_score = current_weight
                    best_cw = [list(all_candidates[i]) for i in range(n_candidates) if current_included[i]]
    
    return best_cw


def construct(q, n, radius):
    """Construct a covering code using multiple strategies."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with tetracode direct sum
    codewords = _tetracode_direct_sum(q, n)
    best = list(codewords)
    best_len = len(best)
    
    # Phase 1: Greedy deletion
    for _ in range(3):
        if time.time() > deadline - 60:
            break
        changed = True
        while changed:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline - 60:
                    break
                trial = codewords[:i] + codewords[i+1:]
                if _is_valid_cover(trial, q, n, radius):
                    codewords = trial
                    changed = True
                    break
        if len(codewords) < best_len:
            best = list(codewords)
            best_len = len(best)
    
    # Phase 2: Local search with swaps
    if time.time() < deadline - 30:
        result = _local_search(codewords, q, n, radius, max_iter=200, time_limit=20)
        if len(result) < best_len:
            best = result
            best_len = len(best)
            codewords = result
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        result = _simulated_annealing(codewords, q, n, radius, time_limit=min(remaining, 50))
        if len(result) < best_len:
            best = result
            best_len = len(best)
    
    # Final validation
    if not _is_valid_cover(best, q, n, radius):
        # Fallback to original construction
        best = _tetracode_direct_sum(q, n)
    
    return best
# EVOLVE-BLOCK-END
