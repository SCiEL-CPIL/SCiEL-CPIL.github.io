# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import itertools


def _ball_indices(word, q, n, radius):
    """Return a frozenset of indices (integers 0..q^n-1) within Hamming distance radius of word."""
    word = np.array(word, dtype=np.int8)
    indices = [int(np.ravel_multi_index(tuple(word), (q,) * n))]
    if radius >= 1:
        for pos in range(n):
            for val in range(q):
                if val == word[pos]:
                    continue
                w2 = word.copy()
                w2[pos] = val
                indices.append(int(np.ravel_multi_index(tuple(w2), (q,) * n)))
    return frozenset(indices)


def _tetracode(q, n):
    """Product construction: ternary tetracode (9 words of length 4) x all words on remaining coords."""
    if n < 4:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _build_neighbors(total, q, n):
    """Precompute neighbor indices for all words. Returns (neighbors_array, powers)."""
    powers = [q ** p for p in range(n)]
    num_neighbors = 1 + n * (q - 1)
    neighbors = np.empty((total, num_neighbors), dtype=np.int32)
    for i in range(total):
        d = [(i // powers[p]) % q for p in range(n)]
        neighbors[i, 0] = i
        k = 1
        for p in range(n):
            for alt in range(q):
                if alt != d[p]:
                    new_i = i + (alt - d[p]) * powers[p]
                    neighbors[i, k] = new_i
                    k += 1
    return neighbors, powers


def _sa_cover(q, n, radius, deadline):
    """Simulated annealing based covering code optimization with enhanced swap logic."""
    total = q ** n
    neighbors, powers = _build_neighbors(total, q, n)
    num_neighbors = neighbors.shape[1]
    
    def idx_to_list(idx):
        return [(idx // powers[p]) % q for p in range(n)]
    
    # Greedy initialization
    uncovered = np.ones(total, dtype=np.int8)
    codewords = []
    
    while uncovered.sum() > 0 and time.time() < deadline - 60:
        counts = np.sum(uncovered[neighbors], axis=1)
        best_c = int(np.argmax(counts))
        if counts[best_c] == 0:
            break
        codewords.append(best_c)
        uncovered[neighbors[best_c]] = 0
        
    if uncovered.sum() != 0:
        while uncovered.sum() > 0:
            c = int(np.random.randint(total))
            codewords.append(c)
            uncovered[neighbors[c]] = 0
            
    # Initialize coverage counts
    coverage_counts = np.zeros(total, dtype=np.int32)
    for c in codewords:
        coverage_counts[neighbors[c]] += 1
        
    in_set = set(codewords)
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    random.seed(42)
    np.random.seed(42)
    
    T = 1.0
    T_min = 0.005
    cooling = 0.9998
    
    steps = 0
    consecutive_no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 5 and consecutive_no_improve < max_no_improve:
        steps += 1
        
        # Find redundant words
        redundant_list = []
        for idx, c in enumerate(codewords):
            if coverage_counts[neighbors[c]].min() >= 2:
                redundant_list.append(idx)
                
        if redundant_list:
            if len(redundant_list) > 1:
                slacks = [int(coverage_counts[neighbors[codewords[i]]].min()) for i in redundant_list]
                max_slack = max(slacks)
                best_candidates = [i for i, s in zip(redundant_list, slacks) if s == max_slack]
                idx_to_remove = random.choice(best_candidates)
            else:
                idx_to_remove = redundant_list[0]
            
            c_to_remove = codewords[idx_to_remove]
            codewords.pop(idx_to_remove)
            coverage_counts[neighbors[c_to_remove]] -= 1
            in_set.discard(c_to_remove)
            
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1
            
            T = max(T * cooling, T_min)
            continue
            
        # No redundant words. Try enhanced swaps.
        found_swap = False
        num_tries = 8
        for _ in range(num_tries):
            idx_to_remove = random.randint(0, len(codewords) - 1)
            c_to_remove = codewords[idx_to_remove]
            
            # Get critical words (only covered by this codeword)
            crit_mask = (coverage_counts[neighbors[c_to_remove]] == 1)
            critical_words = neighbors[c_to_remove][crit_mask]
            
            if len(critical_words) == 0:
                continue
            
            # For each critical word, find candidate replacements
            valid_candidates = []
            for w0 in critical_words:
                candidates = neighbors[w0]
                for cand in candidates:
                    cand = int(cand)
                    if cand == c_to_remove:
                        continue
                    if cand in in_set:
                        continue
                    # Check if cand covers all critical words
                    if all(cw in neighbors[cand] for cw in critical_words):
                        valid_candidates.append(cand)
                        break  # one candidate per critical word is enough for diversity
            
            if not valid_candidates:
                continue
            
            c_to_add = random.choice(valid_candidates)
            
            # Perform swap
            codewords[idx_to_remove] = c_to_add
            coverage_counts[neighbors[c_to_remove]] -= 1
            coverage_counts[neighbors[c_to_add]] += 1
            in_set.discard(c_to_remove)
            in_set.add(c_to_add)
            
            found_swap = True
            consecutive_no_improve += 1
            break
        
        if not found_swap:
            consecutive_no_improve += 1
        
        T = max(T * cooling, T_min)

    # Final cleanup: remove all redundant words
    improved = True
    while improved:
        improved = False
        for idx in range(len(codewords)):
            c = codewords[idx]
            if coverage_counts[neighbors[c]].min() >= 2:
                codewords.pop(idx)
                coverage_counts[neighbors[c]] -= 1
                improved = True
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                break

    if len(best_codewords) < len(codewords):
        codewords = best_codewords

    return [idx_to_list(c) for c in codewords]


def _local_search_remove(codewords, q, n, radius, deadline):
    """Try to remove redundant codewords using coverage count approach."""
    total = q ** n
    neighbors, powers = _build_neighbors(total, q, n)
    
    def idx_to_list(idx):
        return [(idx // powers[p]) % q for p in range(n)]
    
    # Convert to indices
    cw_indices = []
    for cw in codewords:
        idx = int(np.ravel_multi_index(tuple(cw), (q,) * n))
        cw_indices.append(idx)
    
    coverage_counts = np.zeros(total, dtype=np.int32)
    for c in cw_indices:
        coverage_counts[neighbors[c]] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(cw_indices)):
            if time.time() > deadline:
                break
            c = cw_indices[idx]
            if coverage_counts[neighbors[c]].min() >= 2:
                cw_indices.pop(idx)
                coverage_counts[neighbors[c]] -= 1
                improved = True
                break
    
    return [idx_to_list(c) for c in cw_indices]


def _multi_start_sa(q, n, radius, deadline):
    """Run SA with multiple random seeds to find better solutions."""
    total = q ** n
    neighbors, powers = _build_neighbors(total, q, n)
    
    def idx_to_list(idx):
        return [(idx // powers[p]) % q for p in range(n)]
    
    best_codewords = None
    best_size = float('inf')
    
    num_starts = 3
    time_per_start = (deadline - 5) / num_starts
    
    for start_i in range(num_starts):
        start_time = time.time()
        sub_deadline = start_time + time_per_start
        
        random.seed(42 + start_i * 1000)
        np.random.seed(42 + start_i * 1000)
        
        # Greedy initialization with different random tie-breaking
        uncovered = np.ones(total, dtype=np.int8)
        codewords = []
        
        while uncovered.sum() > 0 and time.time() < sub_deadline - 30:
            counts = np.sum(uncovered[neighbors], axis=1)
            # Add small random perturbation for diversity
            if start_i > 0:
                perturbation = np.random.randn(total) * 0.01
                counts = counts + perturbation
            best_c = int(np.argmax(counts))
            if counts[best_c] == 0:
                break
            codewords.append(best_c)
            uncovered[neighbors[best_c]] = 0
            
        if uncovered.sum() != 0:
            while uncovered.sum() > 0:
                c = int(np.random.randint(total))
                codewords.append(c)
                uncovered[neighbors[c]] = 0
                
        # Initialize coverage counts
        coverage_counts = np.zeros(total, dtype=np.int32)
        for c in codewords:
            coverage_counts[neighbors[c]] += 1
            
        in_set = set(codewords)
        local_best = list(codewords)
        local_best_size = len(codewords)
        
        T = 1.0
        T_min = 0.005
        cooling = 0.9998
        
        while time.time() < sub_deadline - 3:
            # Find redundant words
            redundant_list = []
            for idx, c in enumerate(codewords):
                if coverage_counts[neighbors[c]].min() >= 2:
                    redundant_list.append(idx)
                    
            if redundant_list:
                if len(redundant_list) > 1:
                    slacks = [int(coverage_counts[neighbors[codewords[i]]].min()) for i in redundant_list]
                    max_slack = max(slacks)
                    best_candidates = [i for i, s in zip(redundant_list, slacks) if s == max_slack]
                    idx_to_remove = random.choice(best_candidates)
                else:
                    idx_to_remove = redundant_list[0]
                
                c_to_remove = codewords[idx_to_remove]
                codewords.pop(idx_to_remove)
                coverage_counts[neighbors[c_to_remove]] -= 1
                in_set.discard(c_to_remove)
                
                if len(codewords) < local_best_size:
                    local_best_size = len(codewords)
                    local_best = list(codewords)
                
                T = max(T * cooling, T_min)
                continue
                
            # Try swaps
            found_swap = False
            num_tries = 5
            for _ in range(num_tries):
                idx_to_remove = random.randint(0, len(codewords) - 1)
                c_to_remove = codewords[idx_to_remove]
                
                crit_mask = (coverage_counts[neighbors[c_to_remove]] == 1)
                critical_words = neighbors[c_to_remove][crit_mask]
                
                if len(critical_words) == 0:
                    continue
                
                valid_candidates = []
                for w0 in critical_words:
                    candidates = neighbors[w0]
                    for cand in candidates:
                        cand = int(cand)
                        if cand == c_to_remove or cand in in_set:
                            continue
                        if all(cw in neighbors[cand] for cw in critical_words):
                            valid_candidates.append(cand)
                            break
                
                if not valid_candidates:
                    continue
                
                c_to_add = random.choice(valid_candidates)
                codewords[idx_to_remove] = c_to_add
                coverage_counts[neighbors[c_to_remove]] -= 1
                coverage_counts[neighbors[c_to_add]] += 1
                in_set.discard(c_to_remove)
                in_set.add(c_to_add)
                found_swap = True
                break
            
            T = max(T * cooling, T_min)

        # Final cleanup
        improved = True
        while improved:
            improved = False
            for idx in range(len(codewords)):
                c = codewords[idx]
                if coverage_counts[neighbors[c]].min() >= 2:
                    codewords.pop(idx)
                    coverage_counts[neighbors[c]] -= 1
                    improved = True
                    if len(codewords) < local_best_size:
                        local_best_size = len(codewords)
                        local_best = list(codewords)
                    break

        if len(local_best) < best_size:
            best_size = len(local_best)
            best_codewords = local_best

    if best_codewords is None:
        return []
    return [idx_to_list(c) for c in best_codewords]


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115  # 115s safety margin out of 150s
    
    # Start with tetracode product construction
    initial = _tetracode(q, n)
    best = [tuple(cw) for cw in initial]
    best_size = len(best)
    
    # Try SA-based cover with multiple starts
    if time.time() < deadline - 20:
        try:
            sa_result = _multi_start_sa(q, n, radius, deadline - 10)
            if len(sa_result) < best_size:
                best = [tuple(cw) for cw in sa_result]
                best_size = len(best)
        except Exception:
            pass
    
    # Local search to remove redundant codewords
    if time.time() < deadline - 5:
        try:
            removed = _local_search_remove([list(cw) for cw in best], q, n, radius, deadline - 3)
            if len(removed) < best_size:
                best = [tuple(cw) for cw in removed]
                best_size = len(best)
        except Exception:
            pass
    
    # Final removal pass
    if time.time() < deadline - 1:
        try:
            removed = _local_search_remove([list(cw) for cw in best], q, n, radius, deadline - 0.5)
            if len(removed) < best_size:
                best = [tuple(cw) for cw in removed]
                best_size = len(best)
        except Exception:
            pass
    
    return [list(cw) for cw in best]
# EVOLVE-BLOCK-END
