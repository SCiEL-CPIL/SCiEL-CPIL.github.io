"""Optimized ternary covering code constructor using Simulated Annealing with NumPy coverage."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import math
import numpy as np


def _hamming_dist(a, b):
    return sum(x != y for x, y in zip(a, b))


def _is_covered(word, codewords, q, n, radius):
    for cw in codewords:
        if _hamming_dist(word, cw) <= radius:
            return True
    return False


def _deltas_within_radius(q, n, radius):
    """Generate all delta vectors with Hamming weight <= radius."""
    deltas = []
    deltas.append([0] * n)
    for weight in range(1, radius + 1):
        for positions in itertools.combinations(range(n), weight):
            for values in itertools.product(range(1, q), repeat=weight):
                delta = [0] * n
                for pos, val in zip(positions, values):
                    delta[pos] = val
                deltas.append(delta)
    return deltas


def _word_to_index(word, q, n):
    """Convert a word (list/tuple of ints) to a flat index."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def _index_to_word(idx, q, n):
    """Convert a flat index to a word (list of ints)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _compute_coverage_array(codewords, q, n, radius, deltas):
    """
    Compute coverage counts as a NumPy array of size q^n.
    """
    total_words = q ** n
    coverage = np.zeros(total_words, dtype=np.int32)
    
    for cw in codewords:
        for delta in deltas:
            word = tuple((cw[i] + delta[i]) % q for i in range(n))
            idx = _word_to_index(word, q, n)
            coverage[idx] += 1
    
    return coverage


def _sa_optimize(codewords, q, n, radius, time_limit, start_time):
    """
    Simulated Annealing optimization using NumPy coverage array.
    """
    deltas = _deltas_within_radius(q, n, radius)
    total_words = q ** n
    
    # Initial coverage as NumPy array
    coverage = _compute_coverage_array(codewords, q, n, radius, deltas)
    
    # Precompute sphere indices for each codeword position
    # For a codeword at position cw, its sphere is all words within radius
    def get_sphere_indices(cw):
        indices = []
        for delta in deltas:
            word = tuple((cw[i] + delta[i]) % q for i in range(n))
            indices.append(_word_to_index(word, q, n))
        return indices
    
    # Track best solution
    best_codewords = [list(cw) for cw in codewords]
    best_coverage = coverage.copy()
    best_size = len(codewords)
    
    # SA parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    current_size = len(codewords)
    
    iteration = 0
    no_improve_count = 0
    last_time = time.time()
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Check time more frequently
        if time.time() - last_time > 0.5:
            last_time = time.time()
        
        move_type = random.random()
        
        if move_type < 0.35 and current_size > 1:
            # Try removing a codeword - pick one with highest redundancy
            # Sample a few candidates and pick the best
            best_idx = -1
            best_redundancy = 0
            
            sample_size = min(current_size, 10)
            candidates = random.sample(range(current_size), sample_size)
            
            for idx in candidates:
                cw = codewords[idx]
                sphere_idx = get_sphere_indices(cw)
                # Find minimum coverage in sphere
                min_cov = min(coverage[s] for s in sphere_idx)
                if min_cov > 1:
                    # This codeword can be removed
                    # Prefer ones with higher minimum coverage (more redundant)
                    if min_cov > best_redundancy:
                        best_redundancy = min_cov
                        best_idx = idx
            
            if best_idx >= 0:
                cw = codewords[best_idx]
                sphere_idx = get_sphere_indices(cw)
                
                codewords.pop(best_idx)
                current_size -= 1
                
                for s in sphere_idx:
                    coverage[s] -= 1
                
                if current_size < best_size:
                    best_size = current_size
                    best_codewords = [list(cw) for cw in codewords]
                    best_coverage = coverage.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        elif move_type < 0.9:
            # Move a random codeword to a nearby position
            idx = random.randint(0, current_size - 1)
            old_cw = codewords[idx]
            
            new_cw = list(old_cw)
            num_changes = random.randint(1, 2)
            positions = random.sample(range(n), num_changes)
            for pos in positions:
                new_cw[pos] = (new_cw[pos] + random.randint(1, q - 1)) % q
            
            # Check if new position is already occupied
            new_tuple = tuple(new_cw)
            if any(tuple(c) == new_tuple for c in codewords):
                no_improve_count += 1
                continue
            
            old_sphere_idx = get_sphere_indices(old_cw)
            new_sphere_idx = get_sphere_indices(new_cw)
            
            # Check if moving would uncover any words
            old_set = set(old_sphere_idx)
            new_set = set(new_sphere_idx)
            
            would_uncover = 0
            for s in old_set - new_set:
                if coverage[s] <= 1:
                    would_uncover += 1
            
            if would_uncover > 0:
                delta_energy = would_uncover * 10
                if random.random() < math.exp(-delta_energy / T):
                    codewords[idx] = new_cw
                    
                    for s in old_sphere_idx:
                        coverage[s] -= 1
                    for s in new_sphere_idx:
                        coverage[s] += 1
                else:
                    no_improve_count += 1
            else:
                codewords[idx] = new_cw
                
                for s in old_sphere_idx:
                    coverage[s] -= 1
                for s in new_sphere_idx:
                    coverage[s] += 1
                
                no_improve_count = 0
        
        else:
            # Try adding a new codeword at a position that covers many uncovered words
            if current_size < 100 and no_improve_count > 50:
                # Find positions with low coverage
                low_cov_threshold = 1
                uncovered_mask = coverage <= low_cov_threshold
                uncovered_count = np.sum(uncovered_mask)
                
                if uncovered_count > 0:
                    # Sample some low-coverage positions
                    low_cov_indices = np.where(uncovered_mask)[0]
                    sample_size = min(len(low_cov_indices), 20)
                    sampled = random.sample(list(low_cov_indices), sample_size)
                    
                    best_new_cw = None
                    best_gain = 0
                    
                    for idx in sampled:
                        word = _index_to_word(idx, q, n)
                        # Try this position and nearby positions
                        candidates = [word]
                        for _ in range(3):
                            c = list(word)
                            pos = random.randint(0, n - 1)
                            c[pos] = (c[pos] + random.randint(1, q - 1)) % q
                            candidates.append(c)
                        
                        for cand in candidates:
                            cand_tuple = tuple(cand)
                            if any(tuple(c) == cand_tuple for c in codewords):
                                continue
                            
                            sphere_idx = get_sphere_indices(cand)
                            sphere_set = set(sphere_idx)
                            
                            # Count how many low-coverage words this would cover
                            gain = 0
                            for s in sphere_set:
                                if coverage[s] <= 1:
                                    gain += 1
                            
                            if gain > best_gain:
                                best_gain = gain
                                best_new_cw = cand
                        break  # Only check first sampled position for speed
                    
                    if best_new_cw is not None:
                        codewords.append(best_new_cw)
                        current_size += 1
                        
                        new_sphere_idx = get_sphere_indices(best_new_cw)
                        for s in new_sphere_idx:
                            coverage[s] += 1
                        
                        # Try to remove some redundant codewords
                        for _ in range(3):
                            removable_idx = -1
                            for i in range(current_size):
                                cw = codewords[i]
                                sphere_idx = get_sphere_indices(cw)
                                if all(coverage[s] > 1 for s in sphere_idx):
                                    removable_idx = i
                                    break
                            
                            if removable_idx != -1:
                                cw = codewords.pop(removable_idx)
                                current_size -= 1
                                sphere_idx = get_sphere_indices(cw)
                                for s in sphere_idx:
                                    coverage[s] -= 1
                                
                                if current_size < best_size:
                                    best_size = current_size
                                    best_codewords = [list(cw) for cw in codewords]
                                    best_coverage = coverage.copy()
                            else:
                                break
                        
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        T = max(T_min, T * cooling_rate)
        
        if no_improve_count > 300:
            codewords = [list(cw) for cw in best_codewords]
            coverage = best_coverage.copy()
            current_size = best_size
            T = 0.5
            no_improve_count = 0
    
    return best_codewords


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q
    with covering radius radius.
    """
    start_time = time.time()
    time_limit = 115.0
    
    if radius >= n:
        return [[0] * n]
    
    if n == 0:
        return [[]]
    
    if q == 1:
        return [[0] * n]
    
    if n <= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 1:
            return [[i] for i in range(q)]
        elif n == 2:
            return [list(p) for p in itertools.product(range(q), repeat=2)]
        elif n == 3:
            return [[a, b, (a + b) % 3] for a in range(3) for b in range(3)]
        else:
            return tetracode[:9] if len(tetracode) >= 9 else tetracode
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    random.shuffle(initial_code)
    
    result = _sa_optimize(initial_code, q, n, radius, time_limit, start_time)
    
    if time.time() - start_time < time_limit - 5:
        deltas = _deltas_within_radius(q, n, radius)
        sample_size = min(500, 3 ** n // 10)
        for _ in range(sample_size):
            word = [random.randint(0, q - 1) for _ in range(n)]
            if not _is_covered(word, result, q, n, radius):
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    return result
# EVOLVE-BLOCK-END
