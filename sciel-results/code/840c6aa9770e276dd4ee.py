"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search with multiple restarts.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
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
    
    all_words = list(range(total_words))
    
    # Precompute neighbors
    # Using lists for faster iteration, sets for membership if needed.
    # For small K, lists are fine.
    neighbors = [None] * total_words
    for w in all_words:
        word = int_to_word(w, n, q)
        neigh = [w]
        if radius >= 1:
            for pos in range(n):
                orig_val = word[pos]
                for val in range(q):
                    if val == orig_val:
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
        neighbors[w] = neigh
    
    def greedy_init():
        covered = [False] * total_words
        codewords = []
        uncovered_count = total_words
        remaining = list(all_words)
        
        while uncovered_count > 0:
            best_word = -1
            best_new_coverage = -1
            
            # Shuffle to break ties and diversify
            random.shuffle(remaining)
            
            for w in remaining:
                new_cov = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        new_cov += 1
                if new_cov > best_new_coverage:
                    best_new_coverage = new_cov
                    best_word = w
                if new_cov == uncovered_count:
                    break
            
            if best_word == -1:
                break
            
            codewords.append(best_word)
            # O(N) removal, acceptable for N=6561
            remaining.remove(best_word)
            for u in neighbors[best_word]:
                if not covered[u]:
                    covered[u] = True
                    uncovered_count -= 1
        
        return codewords
    
    def optimize(codewords, time_limit):
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1
                
        cw_set = set(codewords)
        
        def try_remove_redundant():
            removed_any = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    can_remove = True
                    for u in neighbors[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any
        
        def find_redundant_candidates():
            """Find codewords that can be removed (all their neighbors have coverage >= 2)"""
            redundant = []
            for i, cw in enumerate(codewords):
                can_remove = True
                for u in neighbors[cw]:
                    if coverage_count[u] == 1:
                        can_remove = False
                        break
                if can_remove:
                    redundant.append(i)
            return redundant
        
        try_remove_redundant()
        
        start_time = time.time()
        stuck_count = 0
        last_size = len(codewords)
        
        while time.time() - start_time < time_limit:
            # Try to remove redundant codewords
            if try_remove_redundant():
                stuck_count = 0
                last_size = len(codewords)
                continue
            
            if len(codewords) == 0:
                break
            
            # Identify codewords with unique coverage
            unique_cw_indices = []
            for i, cw in enumerate(codewords):
                has_unique = False
                for u in neighbors[cw]:
                    if coverage_count[u] == 1:
                        has_unique = True
                        break
                if has_unique:
                    unique_cw_indices.append(i)
            
            if unique_cw_indices:
                # Pick one randomly
                idx = random.choice(unique_cw_indices)
                old_cw = codewords[idx]
                
                would_uncover = []
                for u in neighbors[old_cw]:
                    if coverage_count[u] == 1:
                        would_uncover.append(u)
                
                # Candidates: Words that cover all would_uncover words
                # Use intersection of neighbor sets for efficiency
                if not would_uncover:
                    candidates = set()
                else:
                    candidates = set(neighbors[would_uncover[0]])
                    for u in would_uncover[1:]:
                        candidates &= set(neighbors[u])
                        if not candidates:
                            break
                
                valid_candidates = []
                for new_cw in candidates:
                    if new_cw in cw_set or new_cw == old_cw:
                        continue
                    # Score: Maximize overlap with existing coverage (redundancy)
                    score = 0
                    for u in neighbors[new_cw]:
                        if coverage_count[u] > 0:
                            score += 1
                    valid_candidates.append((score, new_cw))
                
                if valid_candidates:
                    # Sort by score to prefer better swaps
                    valid_candidates.sort(key=lambda x: x[0], reverse=True)
                    
                    # Stochastic choice to escape local minima
                    if random.random() < 0.2:
                        chosen = random.choice(valid_candidates)[1]
                    else:
                        chosen = valid_candidates[0][1]
                    
                    # Perform swap
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    for u in neighbors[chosen]:
                        coverage_count[u] += 1
                    codewords[idx] = chosen
                    cw_set.discard(old_cw)
                    cw_set.add(chosen)
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                # No unique coverage. Try to find a swap that reduces size.
                # Look for a codeword that can be replaced by one that covers more redundantly
                best_swap = None
                best_score = 0
                
                # Sample some codewords to find potential improvements
                sample_size = min(len(codewords), 20)
                sampled_indices = random.sample(range(len(codewords)), sample_size)
                
                for idx in sampled_indices:
                    old_cw = codewords[idx]
                    
                    # Find words that overlap significantly with old_cw's coverage
                    # but are not already codewords
                    overlap_candidates = []
                    for u in neighbors[old_cw]:
                        if coverage_count[u] >= 2:
                            for v in neighbors[u]:
                                if v not in cw_set and v != old_cw:
                                    overlap_candidates.append(v)
                    
                    if not overlap_candidates:
                        continue
                    
                    # Try a few candidates
                    for new_cw in random.sample(overlap_candidates, min(5, len(overlap_candidates))):
                        # Check if this swap maintains coverage
                        valid = True
                        for u in neighbors[old_cw]:
                            if coverage_count[u] == 1 and u not in neighbors[new_cw]:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Score: How much redundancy does this add?
                        score = 0
                        for u in neighbors[new_cw]:
                            if coverage_count[u] > 0:
                                score += 1
                        
                        if score > best_score:
                            best_score = score
                            best_swap = (idx, old_cw, new_cw)
                
                if best_swap:
                    idx, old_cw, new_cw = best_swap
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    for u in neighbors[new_cw]:
                        coverage_count[u] += 1
                    codewords[idx] = new_cw
                    cw_set.discard(old_cw)
                    cw_set.add(new_cw)
                    stuck_count = 0
                else:
                    # Random walk to escape local optimum
                    idx = random.randint(0, len(codewords) - 1)
                    old_cw = codewords[idx]
                    new_cw = random.choice(all_words)
                    if new_cw == old_cw:
                        continue
                    
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    for u in neighbors[new_cw]:
                        coverage_count[u] += 1
                    codewords[idx] = new_cw
                    cw_set.discard(old_cw)
                    cw_set.add(new_cw)
                    stuck_count += 1
            
            # If stuck too long, do a more aggressive random perturbation
            if stuck_count > 50:
                # Remove a random codeword and re-add to break symmetry
                if len(codewords) > 10:
                    idx = random.randint(0, len(codewords) - 1)
                    old_cw = codewords[idx]
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    codewords.pop(idx)
                    cw_set.discard(old_cw)
                    stuck_count = 0
                    last_size = len(codewords)
        
        # Final polish: try harder to remove redundant codewords
        for _ in range(100):
            if not try_remove_redundant():
                break
        
        return codewords
    
    # Multiple restarts
    start_time = time.time()
    time_limit = 115
    best_codewords = []
    
    while time.time() - start_time < time_limit:
        # Allocate time for this iteration
        time_elapsed = time.time() - start_time
        if time_elapsed > time_limit - 10:
            break
        
        cw = greedy_init()
        
        # Calculate remaining time for optimization
        opt_time = max(1, time_limit - (time.time() - start_time) - 2)
        cw = optimize(cw, opt_time)
        
        if len(cw) < len(best_codewords) or not best_codewords:
            best_codewords = cw
        
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
    
    # Final verification and cleanup
    if best_codewords:
        # Convert to words
        result = [int_to_word(cw, n, q) for cw in best_codewords]
        
        # Verify coverage (safety check)
        covered = [False] * total_words
        for cw in best_codewords:
            for u in neighbors[cw]:
                covered[u] = True
        
        if all(covered):
            return result
    
    # Fallback: return greedy result if something went wrong
    cw = greedy_init()
    result = [int_to_word(cw, n, q) for cw in cw]
    return result
# EVOLVE-BLOCK-END
