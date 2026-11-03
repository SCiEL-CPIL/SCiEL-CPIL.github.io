import itertools
import random
import numpy as np
import time

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words as a 2D numpy array (total_words, n)
        all_words = np.array(list(itertools.product(range(q), repeat=n)))
        
        # Precompute distance matrix
        diff = np.abs(all_words[:, None, :] - all_words[None, :, :])
        dist = (diff != 0).sum(axis=2)
        
        # coverage_bool[i, j] is True if dist[i, j] <= radius
        coverage_bool = dist <= radius
        # Convert to list of sets for greedy, but keep bool array for LS
        # Actually, for greedy, sets are good. For LS, bool array is better.
        
        def greedy_set_cover(uncovered, code_indices, random_seed=None):
            if random_seed is not None:
                rng = random.Random(random_seed)
            else:
                rng = random.Random()
                
            uncovered = set(uncovered)
            code_indices = list(code_indices)
            
            while uncovered:
                best_idx = -1
                best_cover_count = -1
                candidates = []
                # Vectorized coverage count for uncovered?
                # uncovered is a set. 
                # We can use a boolean mask for uncovered.
                # But updating the mask is O(N).
                # Current loop is O(N) per step.
                # Let's stick to the loop for greedy, it's fast enough for the initial step.
                for i in range(total_words):
                    if i in code_indices:
                        continue
                    # Count how many uncovered words i covers
                    # We need to intersect coverage[i] with uncovered
                    # Using sets is O(size of intersection)
                    # Precomputed sets are better for this.
                    pass
                
                # Re-implement greedy with sets for clarity, it's only run once or few times
                # We need sets for greedy.
                coverage_sets = [set(np.where(coverage_bool[i])[0]) for i in range(total_words)]
                
                for i in range(total_words):
                    if i in code_indices:
                        continue
                    new_cover = len(coverage_sets[i] & uncovered)
                    if new_cover > best_cover_count:
                        best_cover_count = new_cover
                        best_idx = i
                        candidates = [i]
                    elif new_cover == best_cover_count and new_cover > 0:
                        candidates.append(i)
                    
                if best_idx == -1:
                    break
                
                chosen = rng.choice(candidates)
                code_indices.append(chosen)
                uncovered -= coverage_sets[chosen]
                
            return code_indices

        def fast_local_search(code_set):
            code_set = set(code_set)
            # counts[w] is number of codewords covering w
            counts = np.zeros(total_words, dtype=np.int8)
            for c in code_set:
                counts[coverage_bool[c]] += 1
            
            improved = True
            iterations = 0
            max_iterations = 1000 # Can go higher now
            
            while improved and iterations < max_iterations:
                improved = False
                iterations += 1
                
                # 1. Pruning
                removable = None
                for c in list(code_set):
                    # If min count in coverage[c] > 1, c is redundant
                    if np.min(counts[coverage_bool[c]]) > 1:
                        removable = c
                        break
                        
                if removable is not None:
                    code_set.remove(removable)
                    counts[coverage_bool[removable]] -= 1
                    improved = True
                    continue
                
                # 2. Swaps (Vectorized)
                current_list = list(code_set)
                random.shuffle(current_list)
                
                swapped = False
                for c in current_list:
                    # Find private words of c
                    # private_words_mask is True for w where counts[w]==1 and coverage_bool[c,w]==True
                    private_mask = (counts == 1) & coverage_bool[c]
                    private_words = np.where(private_mask)[0]
                    
                    if len(private_words) == 0:
                        code_set.remove(c)
                        counts[coverage_bool[c]] -= 1
                        improved = True
                        swapped = True
                        break
                    
                    # Find any w not in code_set that covers all private_words
                    # Check all w not in code_set
                    # We can check all w, and filter out code_set later?
                    # Or create a mask for not_in_code
                    not_in_code = np.ones(total_words, dtype=bool)
                    for cc in code_set:
                        not_in_code[cc] = False
                        
                    # For each candidate w, check if it covers all private_words
                    # cov_private[w, p] is True if w covers p
                    # We want sum(cov_private[w, private_words]) == len(private_words)
                    
                    # Extract submatrix: coverage_bool[:, private_words]
                    # Shape: (total_words, len(private_words))
                    sub_cov = coverage_bool[:, private_words]
                    
                    # Sum along axis 1
                    covers_count = sub_cov.sum(axis=1)
                    
                    # Candidates are those where covers_count == len(private_words)
                    # And not_in_code is True
                    valid_candidates = np.where((covers_count == len(private_words)) & not_in_code)[0]
                    
                    if len(valid_candidates) > 0:
                        # Pick one randomly
                        chosen = valid_candidates[random.randint(0, len(valid_candidates)-1)]
                        code_set.remove(c)
                        counts[coverage_bool[c]] -= 1
                        code_set.add(chosen)
                        counts[coverage_bool[chosen]] += 1
                        improved = True
                        swapped = True
                        break
                        
                if not swapped:
                    break
                    
            return code_set

        best_code = None
        best_size = float('inf')
        
        start_time = time.time()
        time_limit = 110
        
        # Initial Greedy
        initial_code = greedy_set_cover(set(range(total_words)), [])
        ls_code = fast_local_search(initial_code)
        if len(ls_code) < best_size:
            best_size = len(ls_code)
            best_code = ls_code
            
        # Iterated Local Search
        iterations = 0
        while time.time() - start_time < time_limit:
            iterations += 1
            perturbed_code = list(best_code)
            k_remove = random.randint(1, 3)
            if len(perturbed_code) > k_remove:
                to_remove = random.sample(perturbed_code, k_remove)
                for r in to_remove:
                    perturbed_code.remove(r)
            
            covered = set()
            for c in perturbed_code:
                covered.update(np.where(coverage_bool[c])[0])
            uncovered = set(range(total_words)) - covered
            
            new_code = greedy_set_cover(uncovered, perturbed_code, random_seed=iterations)
            ls_code = fast_local_search(new_code)
            
            if len(ls_code) < best_size:
                best_size = len(ls_code)
                best_code = ls_code
        
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
