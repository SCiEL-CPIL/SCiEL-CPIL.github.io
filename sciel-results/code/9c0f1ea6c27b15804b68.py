import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n
    if total_words == 1:
        return [[0] * n]
    
    n_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, n_neighbors), dtype=np.int32)
    
    q_powers = [q ** i for i in range(n)]
    
    for w in range(total_words):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            digit = (w // q_powers[n - 1 - pos]) % q
            for v in range(q):
                if v == digit:
                    continue
                diff = (v - digit) * q_powers[n - 1 - pos]
                neighbors[w, idx] = w + diff
                idx += 1

    # Precompute neighbor sets as Python sets for fast membership testing
    neighbor_sets = [set(neighbors[w].tolist()) for w in range(total_words)]
    
    def greedy_init(seed_offset=0):
        rng = random.Random(seed_offset)
        covered = np.zeros(total_words, dtype=bool)
        gain = np.full(total_words, n_neighbors, dtype=np.int32)
        codewords = []
        remaining = total_words
        
        while remaining > 0:
            if time.time() > deadline - 30.0:
                break
            best_word = int(np.argmax(gain))
            best_gain = gain[best_word]
            
            if best_gain == 0:
                uncovered = np.where(~covered)[0]
                if len(uncovered) == 0:
                    break
                best_word = int(uncovered[0])
            
            codewords.append(best_word)
            
            for v in neighbors[best_word]:
                if not covered[v]:
                    covered[v] = True
                    remaining -= 1
                    for w in neighbors[v]:
                        gain[w] -= 1
        
        return codewords
    
    def local_search(cw, time_limit):
        cw = cw[:]
        best_cw = cw[:]
        best_size = len(cw)
        
        count = np.zeros(total_words, dtype=np.int32)
        in_code = np.zeros(total_words, dtype=bool)
        
        for ci in cw:
            in_code[ci] = True
            for u in neighbors[ci]:
                count[u] += 1
        
        # Precompute for each word, which codewords cover it (for faster swap evaluation)
        # Actually, let's just do iterative local search with vectorized checks
        
        while time.time() < time_limit:
            improved = False
            
            # Pass 1: Remove redundant codewords (vectorized min check)
            indices = list(range(len(cw)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(cw):
                    continue
                ci = cw[i]
                
                # Check if all neighbors have count > 1
                if np.min(count[neighbors[ci]]) > 1:
                    for u in neighbors[ci]:
                        count[u] -= 1
                    in_code[ci] = False
                    cw.pop(i)
                    improved = True
                    
                    if len(cw) < best_size:
                        best_size = len(cw)
                        best_cw = cw[:]
            
            # Pass 2: Swaps
            if not improved:
                indices = list(range(len(cw)))
                random.shuffle(indices)
                
                for i in indices:
                    if i >= len(cw):
                        continue
                    ci = cw[i]
                    
                    critical_mask = count[neighbors[ci]] == 1
                    critical = neighbors[ci][critical_mask]
                    
                    if len(critical) == 0:
                        for u in neighbors[ci]:
                            count[u] -= 1
                        in_code[ci] = False
                        cw.pop(i)
                        improved = True
                        break
                    
                    if len(critical) > 0:
                        first_crit = critical[0]
                        
                        if len(critical) == 1:
                            valid_cands = []
                            for cand in neighbor_sets[first_crit]:
                                if cand == ci or in_code[cand]:
                                    continue
                                valid_cands.append(cand)
                            
                            if valid_cands:
                                cand = random.choice(valid_cands)
                                for u in neighbors[ci]:
                                    count[u] -= 1
                                for u in neighbors[cand]:
                                    count[u] += 1
                                
                                in_code[ci] = False
                                in_code[cand] = True
                                cw[i] = cand
                                improved = True
                                break
                        else:
                            # Multiple critical: need candidate in intersection of all critical neighbors' neighbor sets
                            # Start with candidates from first critical
                            candidate_set = neighbor_sets[first_crit].copy()
                            for u in critical[1:]:
                                candidate_set &= neighbor_sets[u]
                                if not candidate_set:
                                    break
                            
                            if candidate_set:
                                valid_cands = [c for c in candidate_set if c != ci and not in_code[c]]
                                if valid_cands:
                                    cand = random.choice(valid_cands)
                                    for u in neighbors[ci]:
                                        count[u] -= 1
                                    for u in neighbors[cand]:
                                        count[u] += 1
                                    
                                    in_code[ci] = False
                                    in_code[cand] = True
                                    cw[i] = cand
                                    improved = True
                                    break
            
            if not improved:
                # Try a perturbation: randomly swap a few codewords to escape local minimum
                if len(cw) > 2 and random.random() < 0.3:
                    idx1, idx2 = random.sample(range(len(cw)), 2)
                    c1, c2 = cw[idx1], cw[idx2]
                    
                    # Check if swapping maintains coverage without copying entire array
                    # Only check words that are affected: neighbors of c1 and c2
                    affected = set(neighbors[c1].tolist()) | set(neighbors[c2].tolist())
                    min_val = float('inf')
                    for u in affected:
                        val = count[u]
                        if u in set(neighbors[c1].tolist()):
                            val -= 1
                        if u in set(neighbors[c2].tolist()):
                            val += 1
                        min_val = min(min_val, val)
                    
                    if min_val >= 1:
                        # Perform the swap
                        for u in neighbors[c1]:
                            count[u] -= 1
                        for u in neighbors[c2]:
                            count[u] += 1
                        in_code[c1] = False
                        in_code[c2] = True
                        cw[idx1] = c2
                        improved = True
                        
                        if len(cw) < best_size:
                            best_size = len(cw)
                            best_cw = cw[:]
        
        return best_cw

    # Try multiple greedy initializations with different seeds
    best_overall = None
    best_overall_size = float('inf')
    
    # Allocate time for multiple attempts
    num_attempts = 3
    time_per_attempt = (deadline - 10.0 - start_time) / num_attempts
    
    for attempt in range(num_attempts):
        attempt_start = time.time()
        remaining_time = deadline - 10.0 - attempt_start
        if remaining_time < 15.0:
            break
        
        # Greedy init with different randomization
        if attempt == 0:
            cw = greedy_init(0)
        else:
            # Add some random perturbation to greedy
            cw = greedy_init(attempt * 1000)
        
        # Local search with remaining time
        ls_time = max(5.0, remaining_time * 0.7)
        result = local_search(cw, start_time + ls_time)
        
        if len(result) < best_overall_size:
            best_overall_size = len(result)
            best_overall = result
        
        if time.time() > deadline - 15.0:
            break
    
    if best_overall is None:
        # Fallback: single greedy
        best_overall = greedy_init(0)
    
    # Final polish: one more local search pass on the best result with remaining time
    final_time_limit = deadline - 3.0
    if time.time() < final_time_limit - 5.0:
        best_overall = local_search(best_overall, final_time_limit)
    
    # Extra cleanup: remove any remaining redundant codewords
    if time.time() < deadline - 2.0:
        count = np.zeros(total_words, dtype=np.int32)
        in_code = np.zeros(total_words, dtype=bool)
        for ci in best_overall:
            in_code[ci] = True
            for u in neighbors[ci]:
                count[u] += 1
        
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            for i in range(len(best_overall)):
                ci = best_overall[i]
                if np.min(count[neighbors[ci]]) > 1:
                    for u in neighbors[ci]:
                        count[u] -= 1
                    in_code[ci] = False
                    best_overall.pop(i)
                    changed = True
                    break
    
    # Convert back to list of lists
    result = []
    for ci in best_overall:
        word = []
        val = ci
        for pos in range(n - 1, -1, -1):
            word.append(val % q)
            val //= q
        word.reverse()
        result.append(word)
    
    return result
