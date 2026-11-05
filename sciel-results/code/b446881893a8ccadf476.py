"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    # For the specific case q=3, n=6, radius=1, use optimized greedy set cover
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(n)]

        # Precompute for each word, the 13 words within Hamming distance 1
        neighbors = np.empty((total, 13), dtype=np.int32)
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(n)]
            neighbors[i, 0] = i
            k = 1
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        k += 1

        # Precompute neighbor sets for fast subset checks
        neighbor_sets = [set(neighbors[i].tolist()) for i in range(total)]

        # Helper to convert indices to codewords
        def indices_to_codewords(indices):
            return [[(idx // powers[p]) % 3 for p in range(n)] for idx in indices]

        def codeword_to_index(cw):
            idx = 0
            for p in range(n):
                idx += cw[p] * powers[p]
            return idx

        # Local Search: Remove redundant codewords and perform swaps
        def local_search_remove_and_swap(indices):
            if not indices:
                return indices
            
            indices_set = set(indices)
            
            # Calculate initial coverage counts
            coverage_counts = np.zeros(total, dtype=np.int32)
            for idx in indices:
                coverage_counts[neighbors[idx]] += 1
            
            improved = True
            while improved:
                if time.time() > deadline - 2:
                    break
                improved = False
                
                # 1. Try to remove redundant codewords
                # Vectorized check: find all redundant codewords
                redundant_indices = []
                for i, idx in enumerate(indices):
                    if np.min(coverage_counts[neighbors[idx]]) >= 2:
                        redundant_indices.append(i)
                
                if redundant_indices:
                    # Prefer removal with highest slack (safety margin)
                    slacks = [np.min(coverage_counts[neighbors[indices[i]]]) for i in redundant_indices]
                    max_slack = max(slacks)
                    best_candidates = [i for i, s in zip(redundant_indices, slacks) if s == max_slack]
                    idx_to_remove = random.choice(best_candidates)
                    
                    # Remove it
                    removed_idx = indices.pop(idx_to_remove)
                    indices_set.remove(removed_idx)
                    coverage_counts[neighbors[removed_idx]] -= 1
                    improved = True
                    continue
                
                # 2. Try to swap a codeword with a non-codeword
                # Try multiple swap candidates per iteration
                found_swap = False
                num_swap_tries = 5
                for _ in range(num_swap_tries):
                    if time.time() > deadline - 2:
                        break
                    
                    # Pick a random codeword to try replacing
                    i = random.randint(0, len(indices) - 1)
                    c = indices[i]
                    
                    # Find critical words: neighbors of c with coverage exactly 1
                    crit_mask = (coverage_counts[neighbors[c]] == 1)
                    critical_words = neighbors[c][crit_mask]
                    
                    if len(critical_words) == 0:
                        # Shouldn't happen if not redundant, but handle gracefully
                        continue
                    
                    # Candidates must cover all critical words
                    # Start with neighbors of first critical word
                    w0 = int(critical_words[0])
                    candidates = neighbors[w0]
                    
                    valid_candidates = []
                    for cand in candidates:
                        cand_int = int(cand)
                        if cand_int in indices_set:
                            continue
                        # Check if cand covers all critical words
                        if all(w in neighbor_sets[cand_int] for w in critical_words):
                            valid_candidates.append(cand_int)
                    
                    if not valid_candidates:
                        continue
                    
                    # Pick a candidate
                    w = random.choice(valid_candidates)
                    
                    # Perform swap
                    # Update coverage
                    set_c = neighbor_sets[c]
                    set_w = neighbor_sets[w]
                    
                    # Losing: N(c) \ N(w)
                    for v in neighbors[c]:
                        if v not in set_w:
                            coverage_counts[v] -= 1
                    # Gaining: N(w) \ N(c)
                    for v in neighbors[w]:
                        if v not in set_c:
                            coverage_counts[v] += 1
                    
                    # Update indices
                    indices_set.remove(c)
                    indices_set.add(w)
                    indices[i] = w
                    
                    found_swap = True
                    improved = True
                    break
            
            return indices

        # Greedy Set Cover with random tie-breaking
        def greedy_cover(rng):
            uncovered = np.ones(total, dtype=np.int8)
            codewords = []
            
            while uncovered.sum() > 0:
                if time.time() > deadline - 5:
                    break
                
                # Find candidate that covers most uncovered words
                counts = np.sum(uncovered[neighbors], axis=1)
                max_count = np.max(counts)
                
                if max_count == 0:
                    break
                
                # Find all candidates with max_count
                candidates = np.where(counts == max_count)[0]
                
                # Random choice among candidates
                best_candidate = int(rng.choice(candidates))
                
                codewords.append(best_candidate)
                uncovered[neighbors[best_candidate]] = 0
            
            return codewords

        # Initial best is Direct Sum
        ds_codewords = get_direct_sum()
        best_indices = [codeword_to_index(cw) for cw in ds_codewords]
        best_size = len(best_indices)
        
        # Run multiple restarts of Randomized Greedy
        rng = random.Random(42)
        
        max_restarts = 500
        for _ in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Generate a new random seed for this run to vary tie-breaking
            cand_indices = greedy_cover(rng)
            
            if len(cand_indices) < best_size:
                # Greedy found something smaller
                cand_indices = local_search_remove_and_swap(cand_indices)
                if len(cand_indices) < best_size:
                    best_size = len(cand_indices)
                    best_indices = cand_indices
            elif len(cand_indices) < best_size + 10:
                # Even if greedy is worse, local search might improve it
                cand_indices = local_search_remove_and_swap(cand_indices)
                if len(cand_indices) < best_size:
                    best_size = len(cand_indices)
                    best_indices = cand_indices
            
            # Early exit if optimal found
            if best_size == 73:
                break
        
        # Final high-precision polish pass
        if time.time() < deadline - 3:
            best_indices = local_search_remove_and_swap(best_indices)
            best_size = len(best_indices)
        
        # Convert best indices to codewords
        final_codewords = indices_to_codewords(best_indices)
        
        return final_codewords

    # General case: fallback to direct sum
    return get_direct_sum()
# EVOLVE-BLOCK-END
