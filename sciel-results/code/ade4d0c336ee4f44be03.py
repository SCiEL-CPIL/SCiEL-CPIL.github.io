"""Improved program using simulated annealing with vectorized NumPy operations."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _idx_to_word(idx, q, n):
    """Convert an integer index to a word (list of ints)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _word_to_idx(word, q, n):
    """Convert a word (list of ints) to an integer index."""
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _get_neighbor_indices(word_idx, q, n, radius):
    """Get all indices within Hamming distance <= radius from word_idx.
    Returns a numpy array of indices."""
    word = _idx_to_word(word_idx, q, n)
    neighbors = set([word_idx])
    
    if radius >= 1:
        for pos in range(n):
            for new_val in range(q):
                if new_val != word[pos]:
                    new_word = word[:]
                    new_word[pos] = new_val
                    neighbors.add(_word_to_idx(new_word, q, n))
    
    if radius >= 2:
        for pos1 in range(n):
            for pos2 in range(pos1 + 1, n):
                for v1 in range(q):
                    if v1 == word[pos1]:
                        continue
                    for v2 in range(q):
                        if v2 == word[pos2]:
                            continue
                        new_word = word[:]
                        new_word[pos1] = v1
                        new_word[pos2] = v2
                        neighbors.add(_word_to_idx(new_word, q, n))
    
    return np.array(list(neighbors), dtype=np.int64)


def _compute_coverage_matrix(codewords, q, n, radius):
    """Compute coverage count array: coverage[i] = number of codewords covering word i."""
    total_words = q ** n
    coverage = np.zeros(total_words, dtype=np.int32)
    
    for cw in codewords:
        cw_idx = _word_to_idx(cw, q, n)
        neighbors = _get_neighbor_indices(cw_idx, q, n, radius)
        coverage[neighbors] += 1
    
    return coverage


def _check_feasible(codewords, q, n, radius):
    """Check if all words are covered."""
    total_words = q ** n
    coverage = _compute_coverage_matrix(codewords, q, n, radius)
    return np.all(coverage > 0)


def _initial_construction(q, n, radius):
    """Create initial feasible covering code using structured construction."""
    if q == 3 and n == 8 and radius == 1:
        # Tetracode-based construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        codewords = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n-4):
                codewords.append(h + list(tail))
        return codewords
    else:
        # General greedy
        codewords = []
        total_words = q ** n
        covered = np.zeros(total_words, dtype=bool)
        
        while not np.all(covered):
            # Find uncovered word with most uncovered neighbors
            best_idx = -1
            best_count = -1
            # Sample candidates
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            sample = uncovered_indices[:min(50, len(uncovered_indices))]
            random.shuffle(sample)
            
            for idx in sample:
                word = _idx_to_word(idx, q, n)
                neighbors = _get_neighbor_indices(idx, q, n, radius)
                count = np.sum(~covered[neighbors])
                if count > best_count:
                    best_count = count
                    best_idx = idx
            
            if best_idx >= 0:
                word = _idx_to_word(best_idx, q, n)
                codewords.append(word)
                neighbors = _get_neighbor_indices(best_idx, q, n, radius)
                covered[neighbors] = True
        
        return codewords


def _simulated_annealing(codewords, q, n, radius, max_time=100.0):
    """Simulated annealing to minimize code size while maintaining coverage."""
    start_time = time.time()
    deadline = start_time + max_time
    
    current = [list(w) for w in codewords]
    best = [list(w) for w in codewords]
    best_size = len(best)
    
    # Precompute neighbor index sets for all current codewords
    cw_indices = [_word_to_idx(cw, q, n) for cw in current]
    neighbor_sets = [_get_neighbor_indices(idx, q, n, radius) for idx in cw_indices]
    
    coverage = np.zeros(q ** n, dtype=np.int32)
    for ns in neighbor_sets:
        coverage[ns] += 1
    
    T = 1000.0
    cooling_rate = 0.995
    min_T = 1.0
    
    while time.time() < deadline - 5.0 and T > min_T:
        T *= cooling_rate
        
        # Try different move types
        move_type = random.random()
        
        if move_type < 0.4 and len(current) > 1:
            # Try removing a codeword
            i = random.randint(0, len(current) - 1)
            ns = neighbor_sets[i]
            if np.all(coverage[ns] > 1):
                # Safe to remove
                coverage[ns] -= 1
                current.pop(i)
                neighbor_sets.pop(i)
                cw_indices.pop(i)
                if len(current) < best_size:
                    best = [list(w) for w in current]
                    best_size = len(current)
            else:
                # Not safe; accept with probability
                uncovered_after = np.sum(coverage[ns] == 1)
                if random.random() < np.exp(-uncovered_after / T):
                    coverage[ns] -= 1
                    current.pop(i)
                    neighbor_sets.pop(i)
                    cw_indices.pop(i)
                    # Must restore feasibility later
                    # Re-add a codeword that covers the now-uncovered words
                    uncovered_mask = coverage == 0
                    if np.any(uncovered_mask):
                        uncovered_idx = np.random.choice(np.where(uncovered_mask)[0])
                        new_word = _idx_to_word(uncovered_idx, q, n)
                        current.append(new_word)
                        new_idx = _word_to_idx(new_word, q, n)
                        new_ns = _get_neighbor_indices(new_idx, q, n, radius)
                        neighbor_sets.append(new_ns)
                        cw_indices.append(new_idx)
                        coverage[new_ns] += 1
        
        elif move_type < 0.7 and len(current) > 1:
            # Try swapping two codewords
            i = random.randint(0, len(current) - 1)
            j = random.randint(0, len(current) - 1)
            if i == j:
                continue
            
            old_ns_i = neighbor_sets[i]
            old_ns_j = neighbor_sets[j]
            
            # Generate new codewords
            old_word_i = current[i]
            old_word_j = current[j]
            
            new_word_i = old_word_i[:]
            new_word_j = old_word_j[:]
            
            # Perturb
            for _ in range(random.randint(1, 3)):
                pos = random.randint(0, n - 1)
                new_word_i[pos] = random.randint(0, q - 1)
            for _ in range(random.randint(1, 3)):
                pos = random.randint(0, n - 1)
                new_word_j[pos] = random.randint(0, q - 1)
            
            new_idx_i = _word_to_idx(new_word_i, q, n)
            new_idx_j = _word_to_idx(new_word_j, q, n)
            
            new_ns_i = _get_neighbor_indices(new_idx_i, q, n, radius)
            new_ns_j = _get_neighbor_indices(new_idx_j, q, n, radius)
            
            # Compute energy change
            temp_cov = coverage.copy()
            temp_cov[old_ns_i] -= 1
            temp_cov[old_ns_j] -= 1
            temp_cov[new_ns_i] += 1
            temp_cov[new_ns_j] += 1
            
            new_uncovered = np.sum(temp_cov == 0)
            old_uncovered = np.sum(coverage == 0)
            
            delta_E = new_uncovered - old_uncovered
            
            if delta_E <= 0 or random.random() < np.exp(-delta_E / T):
                # Accept
                coverage = temp_cov
                current[i] = new_word_i
                current[j] = new_word_j
                neighbor_sets[i] = new_ns_i
                neighbor_sets[j] = new_ns_j
                cw_indices[i] = new_idx_i
                cw_indices[j] = new_idx_j
                
                if len(current) < best_size and np.all(coverage > 0):
                    best = [list(w) for w in current]
                    best_size = len(current)
        
        elif move_type < 0.9 and len(current) > 2:
            # Try removing one and adding a better one
            i = random.randint(0, len(current) - 1)
            old_ns = neighbor_sets[i]
            
            # Find uniquely covered words
            unique_mask = coverage[old_ns] == 1
            unique_count = np.sum(unique_mask)
            
            if unique_count == 0:
                # Can safely remove
                coverage[old_ns] -= 1
                current.pop(i)
                neighbor_sets.pop(i)
                cw_indices.pop(i)
                if len(current) < best_size:
                    best = [list(w) for w in current]
                    best_size = len(current)
            else:
                # Try to find replacement
                unique_indices = np.where(unique_mask)[0]
                found = False
                for _ in range(20):
                    base_idx = random.choice(unique_indices)
                    new_word = _idx_to_word(base_idx, q, n)
                    for _ in range(random.randint(0, 2)):
                        pos = random.randint(0, n - 1)
                        new_word[pos] = random.randint(0, q - 1)
                    
                    new_idx = _word_to_idx(new_word, q, n)
                    new_ns = _get_neighbor_indices(new_idx, q, n, radius)
                    
                    # Check if covers all unique words
                    if np.all(np.isin(unique_indices, new_ns)):
                        # Accept
                        coverage[old_ns] -= 1
                        coverage[new_ns] += 1
                        current[i] = new_word
                        neighbor_sets[i] = new_ns
                        cw_indices[i] = new_idx
                        found = True
                        break
                
                if not found:
                    # Try accepting with probability
                    if random.random() < np.exp(-unique_count / T):
                        coverage[old_ns] -= 1
                        current.pop(i)
                        neighbor_sets.pop(i)
                        cw_indices.pop(i)
                        # Re-add to maintain feasibility
                        uncovered_mask = coverage == 0
                        if np.any(uncovered_mask):
                            uncovered_idx = np.random.choice(np.where(uncovered_mask)[0])
                            new_word = _idx_to_word(uncovered_idx, q, n)
                            current.append(new_word)
                            new_idx = _word_to_idx(new_word, q, n)
                            new_ns = _get_neighbor_indices(new_idx, q, n, radius)
                            neighbor_sets.append(new_ns)
                            cw_indices.append(new_idx)
                            coverage[new_ns] += 1
        else:
            # Random perturbation: replace a random codeword
            i = random.randint(0, len(current) - 1)
            old_ns = neighbor_sets[i]
            
            new_word = [random.randint(0, q - 1) for _ in range(n)]
            new_idx = _word_to_idx(new_word, q, n)
            new_ns = _get_neighbor_indices(new_idx, q, n, radius)
            
            temp_cov = coverage.copy()
            temp_cov[old_ns] -= 1
            temp_cov[new_ns] += 1
            
            new_uncovered = np.sum(temp_cov == 0)
            old_uncovered = np.sum(coverage == 0)
            delta_E = new_uncovered - old_uncovered
            
            if delta_E <= 0 or random.random() < np.exp(-delta_E / T):
                coverage = temp_cov
                current[i] = new_word
                neighbor_sets[i] = new_ns
                cw_indices[i] = new_idx
                
                if len(current) < best_size and np.all(coverage > 0):
                    best = [list(w) for w in current]
                    best_size = len(current)
    
    # Final cleanup: remove redundant codewords
    if time.time() < deadline - 2.0:
        i = 0
        while i < len(best):
            cw = best[i]
            cw_idx = _word_to_idx(cw, q, n)
            ns = _get_neighbor_indices(cw_idx, q, n, radius)
            
            temp_cov = np.zeros(q ** n, dtype=np.int32)
            for j, cw2 in enumerate(best):
                if j == i:
                    continue
                cw2_idx = _word_to_idx(cw2, q, n)
                ns2 = _get_neighbor_indices(cw2_idx, q, n, radius)
                temp_cov[ns2] += 1
            
            if np.all(temp_cov[ns] > 0):
                best.pop(i)
            else:
                i += 1
    
    return best


def construct(q, n, radius):
    start_time = time.time()
    total_budget = 115.0  # Safety margin
    
    # Initial construction
    codewords = _initial_construction(q, n, radius)
    
    # Verify feasibility
    if not _check_feasible(codewords, q, n, radius):
        # Add more codewords until feasible
        total_words = q ** n
        covered = np.zeros(total_words, dtype=bool)
        for cw in codewords:
            cw_idx = _word_to_idx(cw, q, n)
            ns = _get_neighbor_indices(cw_idx, q, n, radius)
            covered[ns] = True
        
        while not np.all(covered) and (time.time() - start_time) < total_budget - 20.0:
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            target_idx = random.choice(uncovered_indices)
            target_word = _idx_to_word(target_idx, q, n)
            codewords.append(target_word)
            ns = _get_neighbor_indices(target_idx, q, n, radius)
            covered[ns] = True
    
    # Simulated annealing optimization
    remaining_time = total_budget - (time.time() - start_time) - 5.0
    if remaining_time > 10.0:
        codewords = _simulated_annealing(codewords, q, n, radius, max_time=remaining_time)
    
    # Final verification
    if not _check_feasible(codewords, q, n, radius):
        # Emergency fix
        total_words = q ** n
        covered = np.zeros(total_words, dtype=bool)
        for cw in codewords:
            cw_idx = _word_to_idx(cw, q, n)
            ns = _get_neighbor_indices(cw_idx, q, n, radius)
            covered[ns] = True
        
        while not np.all(covered) and (time.time() - start_time) < total_budget - 2.0:
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            target_idx = random.choice(uncovered_indices)
            target_word = _idx_to_word(target_idx, q, n)
            codewords.append(target_word)
            ns = _get_neighbor_indices(target_idx, q, n, radius)
            covered[ns] = True
    
    return codewords
# EVOLVE-BLOCK-END
