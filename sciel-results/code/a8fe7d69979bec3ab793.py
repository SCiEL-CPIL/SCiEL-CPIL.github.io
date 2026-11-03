"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    if n == 0:
        return [[]]

    total_words = q ** n
    if radius < 0:
        return []

    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word

    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbor lists for all words
    neighbors = []
    for w in range(total_words):
        word = int_to_word(w, n, q)
        neigh = [w]
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    new_word = word[:]
                    new_word[pos] = val
                    neigh.append(word_to_int(new_word, q))
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for val1 in range(q):
                        if val1 == word[pos1]:
                            continue
                        for val2 in range(q):
                            if val2 == word[pos2]:
                                continue
                            new_word = word[:]
                            new_word[pos1] = val1
                            new_word[pos2] = val2
                            new_int = word_to_int(new_word, q)
                            if new_int not in neigh:
                                neigh.append(new_int)
        neighbors.append(neigh)

    # Build reverse neighbors
    reverse_neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        for u in neighbors[w]:
            reverse_neighbors[u].append(w)

    # Precompute neighbor arrays for vectorized operations
    neighbor_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighbors]

    def greedy_init(seed_offset=0):
        """Fast greedy construction with diverse strategies."""
        rng = random.Random(seed_offset)
        covered = np.zeros(total_words, dtype=np.uint8)
        codewords = []
        uncovered_count = total_words
        
        # Strategy: prioritize words that cover many uncovered words
        # Use a priority queue approach with sampling
        while uncovered_count > 0:
            best_word = -1
            best_new_coverage = -1
            
            # Sample uncovered words and their potential codewords
            # Focus on words that are currently uncovered
            uncovered_indices = np.where(covered == 0)[0]
            if len(uncovered_indices) == 0:
                break
                
            # Sample a subset
            sample_size = min(len(uncovered_indices), max(200, len(uncovered_indices) // 3))
            sample = rng.sample(uncovered_indices.tolist(), sample_size)
            
            # For each sampled uncovered word, consider codewords that cover it
            candidate_set = set()
            for u in sample:
                for cw in reverse_neighbors[u]:
                    candidate_set.add(cw)
            
            candidates = list(candidate_set)
            if not candidates:
                # Fallback: random codeword
                best_word = rng.randint(0, total_words - 1)
            else:
                # Score candidates by new coverage
                rng.shuffle(candidates)
                for w in candidates[:100]:
                    new_cov = 0
                    for u in neighbor_arrays[w]:
                        if not covered[u]:
                            new_cov += 1
                    if new_cov > best_new_coverage:
                        best_new_coverage = new_cov
                        best_word = w
                    if new_cov >= best_new_coverage and new_cov > total_words * 0.1:
                        break
            
            if best_word == -1:
                best_word = rng.randint(0, total_words - 1)
            
            codewords.append(best_word)
            for u in neighbor_arrays[best_word]:
                if not covered[u]:
                    covered[u] = 1
                    uncovered_count -= 1
        
        return codewords

    def optimize(codewords, time_limit, seed_offset=0):
        """Optimize the covering with multiple local search moves."""
        rng = random.Random(seed_offset + 1000)
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            for u in neighbor_arrays[cw]:
                coverage_count[u] += 1
        
        cw_set = set(codewords)
        
        def try_remove_redundant():
            """Remove codewords that don't uniquely cover any word."""
            removed_any = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                rng.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    can_remove = True
                    for u in neighbor_arrays[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbor_arrays[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any
        
        try_remove_redundant()
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if try_remove_redundant():
                continue
            
            if len(codewords) == 0:
                break
            
            # Move type 1: Remove a redundant codeword
            idx = rng.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            would_uncover = []
            for u in neighbor_arrays[old_cw]:
                if coverage_count[u] == 1:
                    would_uncover.append(u)
            
            if len(would_uncover) == 0:
                for u in neighbor_arrays[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Move type 2: Swap with a better codeword
            candidate_set = set()
            for u in would_uncover:
                candidate_set.update(reverse_neighbors[u])
            
            candidates = list(candidate_set)
            candidates = [c for c in candidates if c not in cw_set and c != old_cw]
            rng.shuffle(candidates)
            
            best_new_cw = -1
            best_score = -1
            
            for new_cw in candidates[:80]:
                valid = True
                for u in would_uncover:
                    if u not in neighbor_arrays[new_cw]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                score = 0
                for u in neighbor_arrays[new_cw]:
                    if coverage_count[u] >= 1:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_new_cw = new_cw
            
            if best_new_cw != -1:
                for u in neighbor_arrays[old_cw]:
                    coverage_count[u] -= 1
                for u in neighbor_arrays[best_new_cw]:
                    coverage_count[u] += 1
                codewords[idx] = best_new_cw
                cw_set.discard(old_cw)
                cw_set.add(best_new_cw)
                continue
            
            # Move type 3: 2-for-1 - replace two codewords with one
            if len(codewords) >= 2:
                i, j = rng.sample(range(len(codewords)), 2)
                cw1, cw2 = codewords[i], codewords[j]
                
                # Words uniquely covered by cw1 or cw2
                unique_to_1 = [u for u in neighbor_arrays[cw1] if coverage_count[u] == 1]
                unique_to_2 = [u for u in neighbor_arrays[cw2] if coverage_count[u] == 1]
                
                if unique_to_1 or unique_to_2:
                    # Need a replacement that covers all unique words
                    all_unique = set(unique_to_1 + unique_to_2)
                    candidate_set = set()
                    for u in all_unique:
                        candidate_set.update(reverse_neighbors[u])
                    
                    candidates = list(candidate_set)
                    candidates = [c for c in candidates if c not in cw_set and c != cw1 and c != cw2]
                    rng.shuffle(candidates)
                    
                    best_new_cw = -1
                    best_score = -1
                    
                    for new_cw in candidates[:50]:
                        valid = True
                        for u in all_unique:
                            if u not in neighbor_arrays[new_cw]:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        score = 0
                        for u in neighbor_arrays[new_cw]:
                            if coverage_count[u] >= 1:
                                score += 1
                        
                        if score > best_score:
                            best_score = score
                            best_new_cw = new_cw
                    
                    if best_new_cw != -1:
                        # Remove both old codewords
                        for u in neighbor_arrays[cw1]:
                            coverage_count[u] -= 1
                        for u in neighbor_arrays[cw2]:
                            coverage_count[u] -= 1
                        codewords.remove(cw1)
                        codewords.remove(cw2)
                        cw_set.discard(cw1)
                        cw_set.discard(cw2)
                        
                        # Add new codeword
                        for u in neighbor_arrays[best_new_cw]:
                            coverage_count[u] += 1
                        codewords.append(best_new_cw)
                        cw_set.add(best_new_cw)
                        continue
        
        try_remove_redundant()
        return codewords

    # Multiple restarts with diverse initializations
    start_time = time.time()
    time_limit = 115
    best_codewords = []
    best_len = float('inf')
    
    seed_counter = 0
    while time.time() - start_time < time_limit:
        iter_time = min(12, time_limit - (time.time() - start_time))
        if iter_time <= 0:
            break
        
        cw = greedy_init(seed_counter)
        cw = optimize(cw, iter_time - 2, seed_counter)
        
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw
        
        seed_counter += 1
        
        if time.time() - start_time > time_limit - 5:
            break
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
