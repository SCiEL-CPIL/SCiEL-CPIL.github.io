import time
import itertools
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        if q**n < 10000:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        else:
            return []

    # Specific optimization for q=3, n=7, r=1
    space_size = 3**n
    
    # Precompute all words as integers 0..2186
    words = np.array(list(itertools.product(range(3), repeat=n)), dtype=np.int32)
    
    # Build spheres
    spheres = np.zeros((space_size, 15), dtype=np.int32)
    
    def tup_to_idx(t):
        idx = 0
        for v in t:
            idx = idx * 3 + v
        return idx

    for i in range(space_size):
        w = words[i].copy()
        neighbors = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val != orig:
                    w[pos] = new_val
                    neighbors.append(tup_to_idx(w))
            w[pos] = orig
        spheres[i] = neighbors

    # Precompute coverers: for each word w, list of codewords c such that w in spheres[c]
    coverers = [[] for _ in range(space_size)]
    for c in range(space_size):
        for w in spheres[c]:
            coverers[w].append(c)

    def run_greedy_randomized(seed):
        """
        Runs greedy set cover with random tie-breaking using incremental gain updates.
        Returns list of codeword indices.
        """
        rng = random.Random(seed)
        coverage_counts = np.zeros(space_size, dtype=np.int32)
        code_indices = []
        
        # Precompute gain for each candidate: number of uncovered words in its sphere
        # gain[c] = count of w in spheres[c] where coverage_counts[w] == 0
        # We'll maintain this incrementally.
        gain = np.zeros(space_size, dtype=np.int32)
        for c in range(space_size):
            gain[c] = 15  # all 15 words in sphere are initially uncovered
        
        while time.time() < deadline - 5.0:
            if len(code_indices) > 250:
                break
            
            # Find candidates with max gain
            max_gain = np.max(gain)
            if max_gain == 0:
                break
                
            # Find all candidates with max gain
            candidates = np.where(gain == max_gain)[0]
            
            # Randomly pick one
            best_c = candidates[rng.randint(0, len(candidates) - 1)]
            
            code_indices.append(best_c)
            
            # Update coverage_counts and gain incrementally
            sphere = spheres[best_c]
            for w in sphere:
                if coverage_counts[w] == 0:
                    # This word is now covered for the first time
                    # Decrement gain for all codewords that cover w
                    for c2 in coverers[w]:
                        gain[c2] -= 1
                coverage_counts[w] += 1
            
            if np.all(coverage_counts > 0):
                break

        return code_indices

    def try_remove_redundants(code_list, cov_counts):
        removed = False
        i = 0
        while i < len(code_list):
            c = code_list[i]
            # Check if c is essential
            # Essential if any word in sphere(c) has count 1
            if np.any(cov_counts[spheres[c]] == 1):
                i += 1
            else:
                code_list.pop(i)
                cov_counts[spheres[c]] -= 1
                removed = True
                # Don't increment i since we removed element at position i
        return removed

    def local_search(code_list, cov_counts):
        """
        Randomized local search for swaps.
        """
        code_set = set(code_list)
        improved = True
        local_deadline = deadline - 2.0
        no_improve_count = 0
        max_no_improve = 50
        
        while improved and time.time() < local_deadline:
            improved = False
            no_improve_count = 0
            
            # Phase A: Remove redundants
            if try_remove_redundants(code_list, cov_counts):
                improved = True
                code_set = set(code_list)
                continue
            
            # Phase B: Try swaps
            # Pick a random codeword to try replacing
            if not code_list:
                break
                
            # Try multiple random swaps in one pass
            for _ in range(min(10, len(code_list))):
                if time.time() > local_deadline:
                    break
                    
                idx = random.randint(0, len(code_list) - 1)
                c_in = code_list[idx]
                
                s_in = spheres[c_in]
                critical_mask = (cov_counts[s_in] == 1)
                
                if not np.any(critical_mask):
                    # Should have been removed
                    code_list.pop(idx)
                    cov_counts[s_in] -= 1
                    code_set.remove(c_in)
                    improved = True
                    break
                
                critical_words = s_in[critical_mask]
                
                # Find candidates that cover all critical words
                # Intersection of coverers
                common = set(coverers[critical_words[0]])
                for w in critical_words[1:]:
                    common.intersection_update(coverers[w])
                
                candidates = list(common - code_set)
                
                if not candidates:
                    continue
                
                # Pick a random candidate
                c_out = candidates[random.randint(0, len(candidates) - 1)]
                
                # Perform swap
                cov_counts[s_in] -= 1
                cov_counts[spheres[c_out]] += 1
                code_list[idx] = c_out
                code_set.remove(c_in)
                code_set.add(c_out)
                
                # Try to remove redundants
                len_before = len(code_list)
                try_remove_redundants(code_list, cov_counts)
                len_after = len(code_list)
                
                if len_after < len_before:
                    improved = True
                    code_set = set(code_list)
                    break
                else:
                    # Revert
                    code_list[idx] = c_in
                    code_set.remove(c_out)
                    code_set.add(c_in)
                    cov_counts[spheres[c_out]] -= 1
                    cov_counts[s_in] += 1
                    no_improve_count += 1
                    if no_improve_count >= max_no_improve:
                        break

        return code_list

    best_code = None
    best_size = float('inf')
    
    seed = 42
    while time.time() < deadline - 10.0: # Leave 10s buffer
        seed += 1
        # Run Greedy
        code_indices = run_greedy_randomized(seed)
        
        if not code_indices:
            continue
            
        # Calculate initial coverage
        cov_counts = np.zeros(space_size, dtype=np.int32)
        for c in code_indices:
            cov_counts[spheres[c]] += 1
            
        # Verify coverage (safety check)
        if not np.all(cov_counts > 0):
            # If greedy failed to cover (shouldn't happen if max_gain > 0 until covered),
            # fix it by adding necessary words
            uncovered = np.where(cov_counts == 0)[0]
            for w in uncovered:
                # Pick any coverer
                if coverers[w]:
                    c_pick = coverers[w][0]
                    code_indices.append(c_pick)
                    cov_counts[spheres[c_pick]] += 1
                else:
                    # Should not happen
                    pass
        
        # Run Local Search
        code_indices = local_search(code_indices, cov_counts)
        
        # Update best
        if len(code_indices) < best_size:
            best_size = len(code_indices)
            best_code = code_indices.copy()
            # Reset the random state to continue exploring
            random.seed(seed)
            
        # If we reached a very good upper bound, stop early to save time
        if best_size <= 182:
            # Do a few more quick local search passes on the best solution
            if best_code is not None and time.time() < deadline - 3.0:
                cov_counts_best = np.zeros(space_size, dtype=np.int32)
                for c in best_code:
                    cov_counts_best[spheres[c]] += 1
                best_code = local_search(best_code, cov_counts_best)
                if len(best_code) < best_size:
                    best_size = len(best_code)
            break

    if best_code is None:
        # Fallback to a deterministic run if multi-start failed
        code_indices = run_greedy_randomized(0)
        cov_counts = np.zeros(space_size, dtype=np.int32)
        for c in code_indices:
            cov_counts[spheres[c]] += 1
        if not np.all(cov_counts > 0):
             uncovered = np.where(cov_counts == 0)[0]
             for w in uncovered:
                 if coverers[w]:
                     c_pick = coverers[w][0]
                     code_indices.append(c_pick)
                     cov_counts[spheres[c_pick]] += 1
        best_code = code_indices

    result = [words[i].tolist() for i in best_code]
    return result
