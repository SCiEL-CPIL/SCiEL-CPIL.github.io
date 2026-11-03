import itertools
import random
import numpy as np
import time

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n  # 729
        all_words = np.array(list(itertools.product(range(q), repeat=n)))
        
        # Precompute coverage: coverage[i] is a numpy bool array of length total_words
        # coverage[i][j] is True if dist(all_words[i], all_words[j]) <= radius
        # Use broadcasting for speed
        diff = np.abs(all_words[:, None, :] - all_words[None, :, :])
        dist = (diff != 0).sum(axis=2)
        coverage = (dist <= radius)  # shape (729, 729)
        
        # Also keep as list of sets for easier manipulation in some parts? 
        # Actually, numpy is faster for bulk operations. Let's stick to numpy.
        
        def greedy_set_cover(uncovered_mask, code_indices, random_seed=None):
            # uncovered_mask: numpy bool array of length total_words
            # code_indices: list of indices
            if random_seed is not None:
                rng = random.Random(random_seed)
            else:
                rng = random.Random()
                
            uncovered = uncovered_mask.copy()
            code_indices = list(code_indices)
            
            while uncovered.any():
                # Calculate coverage of each word on the uncovered set
                # coverage[i] is (729,), uncovered is (729,)
                # new_cover[i] = sum(coverage[i] & uncovered)
                new_cover = (coverage & uncovered[None, :]).sum(axis=1)
                
                best_cover_count = new_cover.max()
                if best_cover_count == 0:
                    break # Should not happen if valid problem, but safety
                    
                # Find all candidates with max coverage
                candidates = np.where(new_cover == best_cover_count)[0]
                chosen = rng.choice(candidates.tolist())
                
                code_indices.append(chosen)
                uncovered &= ~coverage[chosen]
                
            return code_indices

        def local_search(code_set):
            code_set = set(code_set)
            # counts[w] = number of codewords covering word w
            counts = np.zeros(total_words, dtype=np.int32)
            for c in code_set:
                counts[coverage[c]] += 1
            
            improved = True
            iterations = 0
            max_iterations = 300
            
            while improved and iterations < max_iterations:
                improved = False
                iterations += 1
                
                # 1. Pruning: remove codeword if all words it covers are covered by >= 2 others
                # This is O(|code_set| * |coverage|) which is 73 * 19 ~ 1400 ops. Fast.
                removable = None
                for c in code_set:
                    # Check if min count in coverage[c] is > 1
                    if counts[coverage[c]].min() > 1:
                        removable = c
                        break
                        
                if removable is not None:
                    code_set.remove(removable)
                    counts[coverage[removable]] -= 1
                    improved = True
                    continue
                
                # 2. Swaps: try to replace a codeword with another to reduce size or maintain
                # Actually, for minimum size, we only swap if it allows pruning later?
                # Or just try to find a swap that doesn't increase size?
                # A better heuristic: try to remove a codeword, and if it breaks coverage,
                # see if we can fix it by adding one back (which is just undo) or adding a different one.
                # Standard local search for set cover:
                # - Remove c.
                # - If coverage is broken, find w that covers the most broken words.
                # - If w covers all broken words, swap c -> w. (Size stays same, but structure changes)
                # - If size can be reduced, we would have pruned in step 1.
                
                current_list = list(code_set)
                random.shuffle(current_list)
                
                swapped = False
                for c in current_list:
                    # Remove c temporarily
                    code_set.remove(c)
                    counts[coverage[c]] -= 1
                    
                    # Find uncovered words
                    uncovered = counts == 0
                    if not uncovered.any():
                        # We removed c and coverage is still valid!
                        # This means c was redundant? But step 1 checks min count > 1.
                        # If min count was 1, removing it makes count 0.
                        # So if uncovered.any() is False, it means counts were all >= 1 after removal?
                        # That implies counts were >= 2 before removal for all covered words.
                        # Which means step 1 should have caught it.
                        # Unless there's a bug. Let's assume this is a reduction.
                        improved = True
                        swapped = True
                        break
                    
                    # Find best replacement
                    # We need a word w that covers all currently uncovered words.
                    # Ideally, we want to minimize the "cost" or just find any valid one.
                    # To allow further pruning, we might prefer a word that covers many words already covered by others.
                    
                    # Efficient check:
                    # For each candidate w not in code_set:
                    #   if (uncovered & ~coverage[w]).sum() == 0:
                    #       w covers all uncovered.
                    
                    # Vectorized check:
                    # uncovered is (729,) bool
                    # coverage is (729, 729) bool
                    # Does w cover uncovered? -> (uncovered & ~coverage[w]).sum() == 0
                    # Equivalent to: (uncovered & coverage[w]).sum() == uncovered.sum()
                    
                    # Let's compute coverage of uncovered for all w
                    # cov_uncovered[w] = sum(coverage[w] & uncovered)
                    cov_uncovered = (coverage & uncovered[None, :]).sum(axis=1)
                    num_uncovered = uncovered.sum()
                    
                    # Candidates are those where cov_uncovered[w] == num_uncovered
                    # And w is not in code_set
                    valid_candidates = np.where(cov_uncovered == num_uncovered)[0]
                    valid_candidates = [w for w in valid_candidates if w not in code_set]
                    
                    if valid_candidates:
                        # Pick one. Maybe pick the one that covers the most total words (to help future pruning)?
                        # Or random.
                        # Let's pick the one with max total coverage to be "strong"
                        # total_cov = coverage.sum(axis=1)
                        # best_w = max(valid_candidates, key=lambda w: total_cov[w])
                        # Or random to escape local minima
                        chosen_w = random.choice(valid_candidates)
                        
                        code_set.add(chosen_w)
                        counts[coverage[chosen_w]] += 1
                        swapped = True
                        # We don't mark improved=True here because size didn't change.
                        # But the structure changed, so we continue the loop to check for pruning again.
                        # Actually, if we swapped, we should restart the pruning phase or continue.
                        # Breaking out of the for-loop and continuing the while-loop is good.
                        break
                    else:
                        # No valid replacement, put c back
                        code_set.add(c)
                        counts[coverage[c]] += 1
                        
                if not swapped:
                    # No swap found and no pruning possible
                    break
                    
            return code_set

        best_code = None
        best_size = float('inf')
        
        start_time = time.time()
        time_limit = 115
        
        # Initial Greedy + Local Search
        initial_uncovered = np.ones(total_words, dtype=bool)
        initial_code = greedy_set_cover(initial_uncovered, [])
        ls_code = local_search(initial_code)
        if len(ls_code) < best_size:
            best_size = len(ls_code)
            best_code = ls_code
            
        # Iterated Local Search
        iterations = 0
        while time.time() - start_time < time_limit:
            iterations += 1
            
            # Perturb the best code
            perturbed_code = list(best_code)
            k_remove = random.randint(1, 3)
            if len(perturbed_code) > k_remove:
                to_remove = random.sample(perturbed_code, k_remove)
                for r in to_remove:
                    perturbed_code.remove(r)
            
            # Calculate uncovered
            covered = np.zeros(total_words, dtype=bool)
            for c in perturbed_code:
                covered |= coverage[c]
            uncovered = ~covered
            
            # Re-cover greedily
            new_code = greedy_set_cover(uncovered, perturbed_code, random_seed=iterations)
            
            # Local Search
            ls_code = local_search(new_code)
            
            if len(ls_code) < best_size:
                best_size = len(ls_code)
                best_code = ls_code
                # If we reach 73, we can try to hold onto it and keep searching for 72
                # but 73 is the known optimum.
                
        return [list(all_words[i]) for i in best_code]
    
    else:
        # Fallback
        all_words = list(itertools.product(range(q), repeat=n))
        total_words = len(all_words)
        
        def hamming_distance(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
            
        coverage = [set() for _ in range(total_words)]
        for i, c in enumerate(all_words):
            for j, w in enumerate(all_words):
                if hamming_distance(c, w) <= radius:
                    coverage[i].add(j)
        
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                count = len(coverage[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1: break
            code_indices.append(best_idx)
            uncovered -= coverage[best_idx]
            
        return [list(all_words[i]) for i in code_indices]
