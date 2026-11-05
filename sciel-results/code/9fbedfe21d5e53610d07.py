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
            # Ternary tetracode: 9 words of length 4
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # Extend to length n by appending all possible tails
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    # For the specific case q=3, n=6, radius=1, use optimized local search on direct sum
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

        # Convert list of codewords (lists of ints) to list of indices
        def words_to_indices(words):
            indices = []
            for w in words:
                idx = 0
                for p in range(n):
                    idx += w[p] * powers[p]
                indices.append(idx)
            return indices

        def indices_to_words(indices):
            return [[(idx // powers[p]) % 3 for p in range(n)] for idx in indices]

        # Initial solution: Direct sum construction
        initial_words = get_direct_sum()
        initial_indices = words_to_indices(initial_words)
        
        best_codewords = initial_indices.copy()
        best_len = len(best_codewords)

        def local_search(codewords, max_iter=100):
            """Aggressive local search: removal + 1-swap + 2-opt."""
            if time.time() > deadline - 3:
                return codewords
            
            improved = True
            iter_count = 0
            while improved and iter_count < max_iter:
                if time.time() > deadline - 3:
                    break
                improved = False
                iter_count += 1
                
                # Build coverage counts
                coverage_counts = np.zeros(total, dtype=np.int32)
                for idx_c in codewords:
                    coverage_counts[neighbors[idx_c]] += 1
                
                # Phase 1: Remove redundant codewords
                for i in range(len(codewords) - 1, -1, -1):
                    idx_c = codewords[i]
                    if np.min(coverage_counts[neighbors[idx_c]]) >= 2:
                        codewords.pop(i)
                        coverage_counts[neighbors[idx_c]] -= 1
                        improved = True
                
                if improved:
                    continue
                
                # Phase 2: 1-swap — try replacing codeword i with non-codeword j
                # such that coverage is maintained
                if time.time() > deadline - 3:
                    break
                
                codeword_set = set(codewords)
                
                # Randomize order for diversity
                order = list(range(len(codewords)))
                random.shuffle(order)
                
                for i in order:
                    idx_c = codewords[i]
                    # Words only covered by this codeword
                    only_mask = (coverage_counts[neighbors[idx_c]] == 1)
                    only_words = neighbors[idx_c][only_mask]
                    
                    if len(only_words) == 0:
                        # Can remove
                        codewords.pop(i)
                        coverage_counts[neighbors[idx_c]] -= 1
                        improved = True
                        break
                    
                    # Find a non-codeword that covers all only_words
                    # A candidate j must cover all words in only_words
                    # and j must not be in codeword_set
                    # Check: for each word w in only_words, j must be in neighbors^{-1}(w)
                    # i.e., w must be in neighbors[j]
                    
                    # For efficiency, intersect the sets of candidates covering each only_word
                    # Start with all candidates covering the first only_word
                    first_w = only_words[0]
                    # Candidates are words whose neighbor set includes first_w
                    # This is the reverse: which codeword positions cover first_w?
                    # neighbors[j] contains first_w iff first_w is within Hamming distance 1 of j
                    # i.e., j is within Hamming distance 1 of first_w
                    # So candidates are neighbors[first_w]
                    candidates = neighbors[first_w].tolist()
                    
                    # Filter: must cover all only_words
                    valid_candidates = []
                    for cand in candidates:
                        if cand in codeword_set:
                            continue
                        # Check if cand covers all only_words
                        cand_neighbors_set = set(neighbors[cand].tolist())
                        if all(w in cand_neighbors_set for w in only_words):
                            valid_candidates.append(cand)
                    
                    if valid_candidates:
                        # Try each valid candidate
                        random.shuffle(valid_candidates)
                        for new_cand in valid_candidates:
                            # Check if adding new_cand and removing idx_c maintains coverage
                            # After removal, coverage_counts decreases by 1 for neighbors[idx_c]
                            # After addition, coverage_counts increases by 1 for neighbors[new_cand]
                            # Need: for all words w, new_coverage[w] >= 1
                            # i.e., coverage_counts[w] - (1 if w in neighbors[idx_c] else 0) + (1 if w in neighbors[new_cand] else 0) >= 1
                            # i.e., if w in neighbors[idx_c] and not in neighbors[new_cand], need coverage_counts[w] >= 2
                            
                            idx_neighbors = set(neighbors[idx_c].tolist())
                            new_neighbors = set(neighbors[new_cand].tolist())
                            
                            ok = True
                            for w in idx_neighbors:
                                if w not in new_neighbors and coverage_counts[w] < 2:
                                    ok = False
                                    break
                            
                            if ok:
                                # Perform the swap
                                codewords[i] = new_cand
                                codeword_set.discard(idx_c)
                                codeword_set.add(new_cand)
                                coverage_counts[neighbors[idx_c]] -= 1
                                coverage_counts[neighbors[new_cand]] += 1
                                improved = True
                                break
                        
                        if improved:
                            break
                
                if not improved:
                    # Phase 3: Try 2-swap (remove two, add one) — limited
                    if time.time() > deadline - 3:
                        break
                    
                    # Try to find two codewords that can be replaced by one
                    # This is expensive, so limit to a few attempts
                    # Randomize pairs
                    pairs = [(i, j) for i in range(len(codewords)) for j in range(i + 1, len(codewords))]
                    random.shuffle(pairs)
                    
                    for i, j in pairs:
                        if time.time() > deadline - 3:
                            break
                        # Check if removing both i and j leaves some words uncovered
                        # and if there's a single codeword that covers those words
                        idx_i = codewords[i]
                        idx_j = codewords[j]
                        
                        # Compute coverage after removing both
                        temp_coverage = coverage_counts.copy()
                        temp_coverage[neighbors[idx_i]] -= 1
                        temp_coverage[neighbors[idx_j]] -= 1
                        
                        # Words that would become uncovered
                        uncovered_after = (temp_coverage < 1)
                        if not uncovered_after.any():
                            # Both can be removed!
                            codewords.pop(j)
                            codewords.pop(i)
                            improved = True
                            break
                        
                        # Words that are now only covered by 0 (uncovered)
                        # Actually we need words with coverage < 1 after removal
                        # But we also need to check if a single new codeword covers them
                        # The uncovered words are those with temp_coverage == 0
                        # But some words might have coverage 1 before, so after removal they have 0
                        # We need a codeword that covers ALL words with temp_coverage == 0
                        
                        zero_words = np.where(temp_coverage == 0)[0]
                        if len(zero_words) == 0:
                            codewords.pop(j)
                            codewords.pop(i)
                            improved = True
                            break
                        
                        if len(zero_words) > 4:  # Too many to cover with one codeword (max 13 neighbors)
                            continue
                        
                        # Find a codeword that covers all zero_words
                        # It must be within Hamming distance 1 of all zero_words
                        # Intersect neighbors of each zero_word
                        common_candidates = set(neighbors[zero_words[0]].tolist())
                        for zw in zero_words[1:]:
                            common_candidates &= set(neighbors[zw].tolist())
                            if not common_candidates:
                                break
                        
                        if common_candidates:
                            # Try each candidate
                            for new_cand in common_candidates:
                                if new_cand in codeword_set:
                                    continue
                                # Check: after adding new_cand, all words have coverage >= 1
                                # temp_coverage + 1 for neighbors[new_cand]
                                new_temp = temp_coverage.copy()
                                new_temp[neighbors[new_cand]] += 1
                                if np.min(new_temp) >= 1:
                                    # Perform the 2-swap
                                    codewords.pop(j)
                                    codewords.pop(i)
                                    codewords.append(new_cand)
                                    codeword_set.discard(idx_i)
                                    codeword_set.discard(idx_j)
                                    codeword_set.add(new_cand)
                                    coverage_counts = new_temp
                                    improved = True
                                    break
                            
                            if improved:
                                break
                        
                    # End of 2-swap loop
                # End of while loop
            return codewords

        # Run local search on the initial direct sum solution
        current = initial_indices.copy()
        current = local_search(current)
        
        if len(current) < best_len:
            best_codewords = current
            best_len = len(current)

        # If we have time, try a few more restarts with slightly perturbed initial solutions
        # or just run local search again on the best found so far with different random seeds
        while time.time() < deadline - 10:
            # Try to improve the best solution found
            candidate = best_codewords.copy()
            # Perturb: randomly swap one codeword with a random neighbor? 
            # Or just run local search again, it might find a different local minimum due to randomization
            candidate = local_search(candidate)
            
            if len(candidate) < best_len:
                best_codewords = candidate
                best_len = len(candidate)
            else:
                # If no improvement, break to save time for safety
                # Or continue if we have plenty of time
                if best_len <= 73:
                    break
                if time.time() > deadline - 20:
                    break

        # Convert indices back to codeword lists
        final_codewords = indices_to_words(best_codewords)
        
        # Verify coverage just in case
        covered = np.zeros(total, dtype=np.bool_)
        for idx in best_codewords:
            covered[neighbors[idx]] = True
        if not covered.all():
            # Fallback to direct sum if something went wrong
            return get_direct_sum()

        return final_codewords

    # General case: fallback to direct sum
    return get_direct_sum()
# EVOLVE-BLOCK-END
