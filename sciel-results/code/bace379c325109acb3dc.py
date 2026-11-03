import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 113.0  # Safety margin
    
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
    
    # Precompute neighbor sets for fast membership
    neighbor_sets = [set(neighbors[w].tolist()) for w in range(total_words)]
    
    # Convert index to vector over GF(3)
    def idx_to_vec(idx):
        vec = np.zeros(n, dtype=np.int32)
        for pos in range(n - 1, -1, -1):
            vec[pos] = idx % q
            idx //= q
        return vec
    
    # Convert vector to index
    def vec_to_idx(vec):
        idx = 0
        for pos in range(n):
            idx = idx * q + vec[pos]
        return idx
    
    # Generate a structured initial code using linear code cosets
    def structured_init(method=0):
        """Generate initial code using algebraic structure."""
        # Method 0: Use a [8,4] linear code over GF(3) and its cosets
        # The Hamming code over GF(3) has parameters [8,4,3] or similar
        # We'll use a systematic construction
        
        if method == 0:
            # Use all cosets of a [8,4] code - gives 3^4 = 81 codewords per coset leader
            # Actually, for covering radius 1, we need a different approach
            # Let's use greedy but with algebraic bias
            
            # Start with a linear code: all vectors with even sum (or similar)
            # Better: use the fact that we need to cover 6561 words
            # Each codeword covers 17 words, so we need at least 6561/17 ≈ 386 codewords
            
            # Use a structured greedy: process words in algebraic order
            covered = np.zeros(total_words, dtype=bool)
            gain = np.full(total_words, n_neighbors, dtype=np.int32)
            codewords = []
            remaining = total_words
            
            # Process in blocks of cosets for better structure
            while remaining > 0:
                if time.time() > deadline - 40.0:
                    break
                
                # Find word with max gain
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
        
        elif method == 1:
            # Randomized greedy with algebraic perturbation
            rng = random.Random(42 + method * 1000)
            covered = np.zeros(total_words, dtype=bool)
            gain = np.full(total_words, n_neighbors, dtype=np.int32)
            codewords = []
            remaining = total_words
            
            # Add small random noise to gain to break ties differently
            gain_noisy = gain.astype(np.float64) + rng.random_sample(total_words) * 0.5
            gain = gain_noisy.astype(np.int32)
            
            while remaining > 0:
                if time.time() > deadline - 40.0:
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
        
        else:
            # Method 2+: Use different random seeds
            rng = random.Random(12345 + method * 7777)
            covered = np.zeros(total_words, dtype=bool)
            gain = np.full(total_words, n_neighbors, dtype=np.int32)
            codewords = []
            remaining = total_words
            
            while remaining > 0:
                if time.time() > deadline - 40.0:
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
        """Efficient local search with removal and swap operations."""
        cw = cw[:]
        best_cw = cw[:]
        best_size = len(cw)
        
        count = np.zeros(total_words, dtype=np.int32)
        in_code = np.zeros(total_words, dtype=bool)
        
        for ci in cw:
            in_code[ci] = True
            for u in neighbors[ci]:
                count[u] += 1
        
        no_improve_count = 0
        max_no_improve = 10
        
        while time.time() < time_limit:
            improved = False
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Perturbation: randomly remove and re-add
                if len(cw) > 5:
                    for _ in range(min(3, len(cw) // 10)):
                        idx = random.randint(0, len(cw) - 1)
                        ci = cw[idx]
                        for u in neighbors[ci]:
                            count[u] -= 1
                        in_code[ci] = False
                        cw.pop(idx)
                    # Re-cover uncovered
                    uncovered = np.where(count == 0)[0]
                    for u in uncovered[:len(uncovered)]:
                        # Find a candidate to cover u
                        for cand in neighbor_sets[u]:
                            if not in_code[cand]:
                                cw.append(cand)
                                in_code[cand] = True
                                for v in neighbors[cand]:
                                    count[v] += 1
                                break
                    no_improve_count = 0
                    continue
            
            # Pass 1: Remove redundant codewords
            indices = list(range(len(cw)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(cw):
                    continue
                ci = cw[i]
                
                if np.min(count[neighbors[ci]]) > 1:
                    for u in neighbors[ci]:
                        count[u] -= 1
                    in_code[ci] = False
                    cw.pop(i)
                    improved = True
                    
                    if len(cw) < best_size:
                        best_size = len(cw)
                        best_cw = cw[:]
            
            # Pass 2: Swaps - try to replace codewords with better ones
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
                        # Can remove
                        for u in neighbors[ci]:
                            count[u] -= 1
                        in_code[ci] = False
                        cw.pop(i)
                        improved = True
                        no_improve_count = 0
                        break
                    
                    if len(critical) > 0:
                        # Need to find a replacement that covers all critical words
                        if len(critical) == 1:
                            first_crit = critical[0]
                            # Candidates are neighbors of first_crit that aren't in code
                            for cand in neighbor_sets[first_crit]:
                                if cand == ci or in_code[cand]:
                                    continue
                                
                                # Check if cand covers all critical words
                                valid = True
                                for u in critical:
                                    if u not in neighbor_sets[cand]:
                                        valid = False
                                        break
                                
                                if valid:
                                    for u in neighbors[ci]:
                                        count[u] -= 1
                                    for u in neighbors[cand]:
                                        count[u] += 1
                                    in_code[ci] = False
                                    in_code[cand] = True
                                    cw[i] = cand
                                    improved = True
                                    no_improve_count = 0
                                    break
                        
                        if not improved and len(critical) > 1:
                            # Multiple critical: need intersection of neighbor sets
                            candidate_set = neighbor_sets[critical[0]].copy()
                            for u in critical[1:]:
                                candidate_set &= neighbor_sets[u]
                                if not candidate_set:
                                    break
                            
                            if candidate_set:
                                for cand in candidate_set:
                                    if cand == ci or in_code[cand]:
                                        continue
                                    
                                    for u in neighbors[ci]:
                                        count[u] -= 1
                                    for u in neighbors[cand]:
                                        count[u] += 1
                                    in_code[ci] = False
                                    in_code[cand] = True
                                    cw[i] = cand
                                    improved = True
                                    no_improve_count = 0
                                    break
            
            if improved:
                no_improve_count = 0
        
        return best_cw
    
    # Try multiple diverse initializations
    best_overall = None
    best_overall_size = float('inf')
    
    num_attempts = 4
    elapsed = time.time() - start_time
    time_per_attempt = (deadline - 15.0 - start_time) / num_attempts
    
    for attempt in range(num_attempts):
        attempt_start = time.time()
        remaining_time = deadline - 12.0 - attempt_start
        if remaining_time < 20.0:
            break
        
        # Different initialization methods
        if attempt == 0:
            cw = structured_init(0)
        elif attempt == 1:
            cw = structured_init(1)
        elif attempt == 2:
            cw = structured_init(2)
        else:
            cw = structured_init(3)
        
        # Local search with remaining time
        ls_time = max(8.0, remaining_time * 0.75)
        result = local_search(cw, start_time + ls_time)
        
        if len(result) < best_overall_size:
            best_overall_size = len(result)
            best_overall = result
        
        if time.time() > deadline - 20.0:
            break
    
    if best_overall is None:
        best_overall = structured_init(0)
    
    # Final polish with remaining time
    final_time_limit = deadline - 5.0
    if time.time() < final_time_limit - 10.0:
        best_overall = local_search(best_overall, final_time_limit)
    
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
