"""Ternary covering code K_3(7,1) via greedy + simulated annealing with incremental coverage."""
import time
import random
import itertools
import numpy as np

DEADLINE_MARGIN = 3.0  # seconds to keep before the 150s wall-clock kill


def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _precompute_neighbors(q, n, radius):
    """Precompute for each word index, the list of indices within Hamming distance <= radius."""
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    total = q ** n
    neighbors = []
    for w in all_words:
        w_list = list(w)
        idxs = {word_to_idx[w]}
        if radius >= 1:
            for i in range(n):
                orig = w_list[i]
                for v in range(q):
                    if v != orig:
                        w_list[i] = v
                        idxs.add(word_to_idx[tuple(w_list)])
                w_list[i] = orig
        if radius >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    oi, oj = w_list[i], w_list[j]
                    for vi in range(q):
                        if vi != oi:
                            w_list[i] = vi
                            for vj in range(q):
                                if vj != oj:
                                    w_list[j] = vj
                                    idxs.add(word_to_idx[tuple(w_list)])
                    w_list[j] = oj
                w_list[i] = oi
        neighbors.append(sorted(idxs))
    return all_words, word_to_idx, neighbors, total


def _greedy_init(q, n, radius, neighbors, total, uncovered_set, counts):
    """Greedy set cover initialization."""
    code = []
    covered = 0
    while uncovered_set:
        best_idx = -1
        best_cov = 0
        for w_idx in range(total):
            cov = 0
            for nb in neighbors[w_idx]:
                if nb in uncovered_set:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_idx = w_idx
        if best_idx == -1:
            break
        code.append(best_idx)
        for nb in neighbors[best_idx]:
            if counts[nb] == 0:
                covered += 1
                uncovered_set.discard(nb)
            counts[nb] += 1
    return code, covered


def _sa_improve(code, counts, covered, neighbors, total, time_budget, seed):
    """Simulated annealing to minimize code size while maintaining full coverage."""
    rng = random.Random(seed)
    code = list(code)
    counts = list(counts)
    covered = covered
    current_size = len(code)
    in_code = set(code)
    
    P = 50.0  # heavy penalty for uncovered words
    
    def get_cost(size, covered):
        return size + P * (total - covered)
    
    current_cost = get_cost(current_size, covered)
    best_code = list(code)
    best_size = current_size
    best_covered = covered
    best_cost = current_cost
    
    start_time = time.time()
    T = 2.0
    T_min = 0.001
    # Cooling: reach T_min in about time_budget
    cooling = 0.9995
    
    while time.time() - start_time < time_budget:
        if T < T_min:
            T = T_min
        
        move_type = rng.random()
        
        if move_type < 0.5 and current_size > 0:
            # Swap: remove one, add one
            idx_remove = rng.randint(0, current_size - 1)
            w_remove = code[idx_remove]
            
            # Pick a word to add - biased toward uncovered regions
            if total - covered > 0 and rng.random() < 0.7:
                # Find an uncovered word and pick a neighbor
                uncovered_list = [i for i in range(total) if counts[i] == 0]
                if uncovered_list:
                    target = rng.choice(uncovered_list)
                    # Pick a random neighbor of target
                    # Find words that cover target
                    candidates = [i for i in range(total) if target in neighbors[i]]
                    if candidates:
                        idx_add = rng.choice(candidates)
                    else:
                        idx_add = rng.randint(0, total - 1)
                else:
                    idx_add = rng.randint(0, total - 1)
            else:
                idx_add = rng.randint(0, total - 1)
            
            if idx_add == w_remove or idx_add in in_code:
                continue
            
            # Evaluate delta
            nbs_remove = neighbors[w_remove]
            nbs_add = neighbors[idx_add]
            
            # Build affected set
            affected = set()
            for nb in nbs_remove:
                affected.add(nb)
            for nb in nbs_add:
                affected.add(nb)
            
            delta_covered = 0
            for nb in affected:
                old_count = counts[nb]
                new_count = old_count
                if nb in nbs_remove:
                    new_count -= 1
                if nb in nbs_add:
                    new_count += 1
                if old_count == 0 and new_count > 0:
                    delta_covered += 1
                elif old_count > 0 and new_count == 0:
                    delta_covered -= 1
            
            new_size = current_size  # swap keeps size same
            new_covered = covered + delta_covered
            new_cost = get_cost(new_size, new_covered)
            
            delta = new_cost - current_cost
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                # Accept move
                # Apply
                code[idx_remove] = idx_add
                in_code.discard(w_remove)
                in_code.add(idx_add)
                for nb in nbs_remove:
                    counts[nb] -= 1
                for nb in nbs_add:
                    counts[nb] += 1
                covered = new_covered
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_code = list(code)
                    best_size = current_size
                    best_covered = covered
            
        elif move_type < 0.8 and current_size > 1:
            # Remove one codeword
            idx_remove = rng.randint(0, current_size - 1)
            w_remove = code[idx_remove]
            nbs_remove = neighbors[w_remove]
            
            delta_covered = 0
            for nb in nbs_remove:
                if counts[nb] == 1:
                    delta_covered -= 1
            
            new_size = current_size - 1
            new_covered = covered + delta_covered
            new_cost = get_cost(new_size, new_covered)
            
            delta = new_cost - current_cost
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                # Accept
                code.pop(idx_remove)
                in_code.discard(w_remove)
                for nb in nbs_remove:
                    counts[nb] -= 1
                current_size = new_size
                covered = new_covered
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_code = list(code)
                    best_size = current_size
                    best_covered = covered
        
        else:
            # Add one codeword
            # Pick a word that covers many uncovered
            if total - covered > 0:
                best_add = -1
                best_gain = 0
                # Sample some candidates
                sample = rng.sample(range(total), min(200, total))
                for w_idx in sample:
                    gain = 0
                    for nb in neighbors[w_idx]:
                        if counts[nb] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_add = w_idx
                idx_add = best_add
            else:
                idx_add = rng.randint(0, total - 1)
            
            if idx_add in in_code:
                continue
            
            nbs_add = neighbors[idx_add]
            delta_covered = 0
            for nb in nbs_add:
                if counts[nb] == 0:
                    delta_covered += 1
            
            new_size = current_size + 1
            new_covered = covered + delta_covered
            new_cost = get_cost(new_size, new_covered)
            
            delta = new_cost - current_cost
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                # Accept
                code.append(idx_add)
                in_code.add(idx_add)
                for nb in nbs_add:
                    counts[nb] += 1
                current_size = new_size
                covered = new_covered
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_code = list(code)
                    best_size = current_size
                    best_covered = covered
        
        T *= cooling
    
    # Return best found
    return best_code


def _remove_redundant(code, counts, neighbors, total, time_budget):
    """Remove any redundant codewords (those whose removal keeps full coverage)."""
    code = list(code)
    counts = list(counts)
    in_code = set(code)
    start_time = time.time()
    
    improved = True
    while improved and time.time() - start_time < time_budget:
        improved = False
        for i in range(len(code)):
            if time.time() - start_time > time_budget:
                break
            w = code[i]
            nbs = neighbors[w]
            # Check if any word becomes uncovered
            removable = True
            for nb in nbs:
                if counts[nb] == 1:
                    removable = False
                    break
            if removable:
                # Remove
                for nb in nbs:
                    counts[nb] -= 1
                code.pop(i)
                in_code.discard(w)
                improved = True
                break
    return code, counts


def _tetracode_construction(q, n, radius):
    """Direct sum of ternary tetracode with full product on remaining coords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n == 4:
        return [list(t) for t in tetracode]
    result = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            result.append(list(h) + list(tail))
    return result


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Special case: n=4, q=3, radius=1 -> perfect code of size 9
    if q == 3 and n == 4 and radius == 1:
        return _tetracode_construction(q, n, radius)
    
    # For n=7, q=3, radius=1: main approach
    if q == 3 and n == 7 and radius == 1:
        # Precompute
        all_words, word_to_idx, neighbors, total = _precompute_neighbors(q, n, radius)
        
        # Greedy initialization
        counts = [0] * total
        uncovered_set = set(range(total))
        code, covered = _greedy_init(q, n, radius, neighbors, total, uncovered_set, counts)
        
        # Remove redundant first
        time_left = deadline - time.time()
        if time_left > 5:
            code, counts = _remove_redundant(code, counts, neighbors, total, min(20, time_left))
            covered = sum(1 for c in counts if c > 0)
        
        # Simulated annealing
        time_left = deadline - time.time()
        if time_left > 10:
            # Multiple SA runs with different seeds
            best_code = code
            best_size = len(code)
            
            sa_time = time_left * 0.8
            # Run SA with a few different seeds if time permits
            for seed in [42, 123, 7]:
                if time.time() >= deadline:
                    break
                t_left = deadline - time.time()
                if t_left < 15:
                    break
                sa_code = _sa_improve(code, counts, covered, neighbors, total, 
                                      min(sa_time, t_left - 5), seed)
                if len(sa_code) < best_size:
                    # Verify
                    new_counts = [0] * total
                    for cw_idx in sa_code:
                        for nb in neighbors[cw_idx]:
                            new_counts[nb] += 1
                    new_covered = sum(1 for c in new_counts if c > 0)
                    if new_covered == total:
                        best_code = sa_code
                        best_size = len(sa_code)
                        # Update counts for next SA run
                        counts = new_counts
                        covered = new_covered
                        # Remove redundant
                        if time.time() < deadline:
                            best_code, counts = _remove_redundant(best_code, counts, 
                                                                 neighbors, total, 10)
                            covered = sum(1 for c in counts if c > 0)
                            best_size = len(best_code)
        
        # Final redundant removal
        time_left = deadline - time.time()
        if time_left > 3:
            best_code, counts = _remove_redundant(best_code, counts, neighbors, total, time_left - 2)
        
        # Convert to list of words
        result = [list(all_words[idx]) for idx in best_code]
        
        # Verify validity
        if len(result) > 0:
            verify_counts = [0] * total
            for cw in result:
                idx = word_to_idx[tuple(cw)]
                for nb in neighbors[idx]:
                    verify_counts[nb] += 1
            if all(c > 0 for c in verify_counts):
                return result
        
        # Fallback: tetracode construction
        return _tetracode_construction(q, n, radius)
    
    # General case: fallback
    if q == 3 and n >= 4:
        return _tetracode_construction(q, n, radius)
    
    # Full product fallback
    return [list(w) for w in itertools.product(range(q), repeat=n)]
