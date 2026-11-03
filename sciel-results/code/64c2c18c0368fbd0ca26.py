import itertools
import random
import numpy as np
import time
import sys

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words as a 2D numpy array (total_words, n)
        all_words = np.array(list(itertools.product(range(q), repeat=n)))
        
        # Precompute distance matrix or coverage sets
        # For efficiency, we compute coverage sets directly
        # coverage[i] is a set of indices j such that dist(i, j) <= radius
        coverage = [set() for _ in range(total_words)]
        
        # Efficient coverage calculation
        # For each word i, find all words within distance 1
        for i in range(total_words):
            word_i = all_words[i]
            # Distance 0
            coverage[i].add(i)
            # Distance 1
            for pos in range(n):
                for val in range(q):
                    if val == word_i[pos]:
                        continue
                    # Create new word
                    new_word = word_i.copy()
                    new_word[pos] = val
                    # Find index of new_word
                    # Since it's a small space, we can precompute a dictionary
                    # But for 729 words, linear search or dict is fine.
                    # Let's build a dict for faster lookup in the main loop if needed, 
                    # but here we just need to populate coverage.
                    # We can convert to tuple and use a map.
                    pass
                    
        # Better: Build a map from tuple to index
        word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
        
        for i in range(total_words):
            word_i = all_words[i]
            coverage[i].add(i)
            for pos in range(n):
                for val in range(q):
                    if val == word_i[pos]:
                        continue
                    new_word = word_i.copy()
                    new_word[pos] = val
                    idx = word_to_idx[tuple(new_word)]
                    coverage[i].add(idx)
        
        # Convert to numpy arrays for faster intersection if needed, 
        # but sets are okay for 729 elements.
        
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
                
                # Optimization: Only consider candidates that cover at least one uncovered word
                # We can iterate through uncovered words and look at their neighbors
                # Or just iterate through all words. 729 is small.
                for i in range(total_words):
                    new_cover = len(coverage[i] & uncovered)
                    if new_cover > best_cover_count:
                        best_cover_count = new_cover
                        best_idx = i
                        candidates = [i]
                    elif new_cover == best_cover_count and new_cover > 0:
                        candidates.append(i)
                    
                if best_idx == -1:
                    break
                
                # Pick randomly from best candidates
                chosen = rng.choice(candidates)
                code_indices.append(chosen)
                uncovered -= coverage[chosen]
                
            return code_indices

        def local_search(code_set):
            code_set = set(code_set)
            
            def get_counts(codes):
                counts = np.zeros(total_words, dtype=int)
                for c in codes:
                    counts[list(coverage[c])] += 1
                return counts
                
            counts = get_counts(code_set)
            
            improved = True
            iterations = 0
            max_iterations = 200
            
            while improved and iterations < max_iterations:
                improved = False
                iterations += 1
                
                # 1. Pruning
                removable = None
                for c in list(code_set):
                    # If min count in coverage[c] is > 1, it's safe to remove
                    if np.min(counts[list(coverage[c])]) > 1:
                        removable = c
                        break
                        
                if removable is not None:
                    code_set.remove(removable)
                    counts[list(coverage[removable])] -= 1
                    improved = True
                    continue
                
                # 2. Swaps
                current_list = list(code_set)
                random.shuffle(current_list)
                
                swapped = False
                for c in current_list:
                    private_words = [w for w in coverage[c] if counts[w] == 1]
                    
                    if not private_words:
                        code_set.remove(c)
                        counts[list(coverage[c])] -= 1
                        improved = True
                        swapped = True
                        break
                    
                    found_w = None
                    # Try to find a replacement w that covers all private words
                    # Optimization: Iterate through words that cover the first private word
                    first_pw = private_words[0]
                    candidates = list(coverage[first_pw])
                    
                    for w in candidates:
                        if w in code_set:
                            continue
                        # Check if w covers all other private words
                        covers_all = True
                        for pw in private_words:
                            if pw not in coverage[w]:
                                covers_all = False
                                break
                        if covers_all:
                            found_w = w
                            break
                            
                    if found_w is not None:
                        code_set.remove(c)
                        counts[list(coverage[c])] -= 1
                        code_set.add(found_w)
                        counts[list(coverage[found_w])] += 1
                        improved = True
                        swapped = True
                        break
                        
                if not swapped:
                    break
                    
            return code_set

        def lagrangian_search(start_code, time_limit):
            # Lagrangian Relaxation for Set Covering
            # Min sum(x_i) s.t. A x >= 1
            # L(lambda) = sum(lambda_j) + sum(x_i * (1 - sum(lambda_j for j in N(i))))
            
            # Initialize lambda to 0
            lambda_vec = np.zeros(total_words, dtype=float)
            
            # Start with a valid solution
            current_code = set(start_code)
            
            # If start_code is not valid, fix it
            covered = set()
            for c in current_code:
                covered.update(coverage[c])
            uncovered = set(range(total_words)) - covered
            if uncovered:
                current_code = set(greedy_set_cover(uncovered, list(current_code)))
            
            best_code = set(current_code)
            best_size = len(best_code)
            
            start_time = time.time()
            max_iter = 1000
            step_size = 0.1
            
            for _ in range(max_iter):
                if time.time() - start_time > time_limit:
                    break
                
                # 1. Solve Primal Subproblem
                # x_i = 1 if (1 - sum(lambda_j for j in N(i))) < 0
                # i.e., sum(lambda_j for j in N(i)) > 1
                
                # Calculate scores for each i
                # score[i] = 1 - sum(lambda[j] for j in coverage[i])
                # If score[i] < 0, select i
                
                # Efficient calculation:
                # sum_lambda = np.zeros(total_words)
                # for j in range(total_words):
                #     if lambda_vec[j] > 0:
                #         for i in coverage[j]:
                #             sum_lambda[i] += lambda_vec[j]
                
                # This is O(N * Deg). N=729, Deg ~ 1+6*2=13. Fast.
                sum_lambda = np.zeros(total_words, dtype=float)
                for j in range(total_words):
                    if lambda_vec[j] > 0:
                        for i in coverage[j]:
                            sum_lambda[i] += lambda_vec[j]
                
                scores = 1.0 - sum_lambda
                # Select i where score < 0
                candidate_code = set(np.where(scores < 0)[0].tolist())
                
                # 2. Check Validity and Update
                # If candidate_code is a valid cover, it's a feasible solution
                # But it might be large. We use it to update best if smaller.
                
                covered_cand = set()
                for c in candidate_code:
                    covered_cand.update(coverage[c])
                
                is_valid = (len(covered_cand) == total_words)
                
                if is_valid and len(candidate_code) < best_size:
                    best_size = len(candidate_code)
                    best_code = set(candidate_code)
                
                # 3. Update Lambda
                # If a constraint j is violated (not covered by candidate_code), increase lambda_j
                # If satisfied, decrease lambda_j (towards 0)
                
                # We use the "candidate_code" coverage status for the update
                # This drives the search towards covering all points.
                
                for j in range(total_words):
                    if j not in covered_cand:
                        # Uncovered, increase penalty
                        lambda_vec[j] += step_size
                    else:
                        # Covered, decrease penalty
                        lambda_vec[j] = max(0, lambda_vec[j] - step_size * 0.5)
                
                # Periodically refine with Local Search on the best found code
                if _ % 10 == 0:
                    ls_code = local_search(best_code)
                    if len(ls_code) < best_size:
                        best_size = len(ls_code)
                        best_code = ls_code
            
            return best_code

        best_code = None
        best_size = float('inf')
        
        start_time = time.time()
        time_limit = 110
        
        # 1. Initial Greedy
        initial_code = greedy_set_cover(set(range(total_words)), [])
        ls_code = local_search(initial_code)
        best_code = ls_code
        best_size = len(ls_code)
        
        # 2. Lagrangian Search
        # Run Lagrangian for a portion of the time
        lag_code = lagrangian_search(initial_code, time_limit / 2)
        if len(lag_code) < best_size:
            best_size = len(lag_code)
            best_code = lag_code
        
        # 3. Iterated Local Search on the best code
        iterations = 0
        while time.time() - start_time < time_limit:
            iterations += 1
            perturbed_code = list(best_code)
            k_remove = random.randint(1, 2)
            if len(perturbed_code) > k_remove:
                to_remove = random.sample(perturbed_code, k_remove)
                for r in to_remove:
                    perturbed_code.remove(r)
            
            covered = set()
            for c in perturbed_code:
                covered.update(coverage[c])
            uncovered = set(range(total_words)) - covered
            
            new_code = greedy_set_cover(uncovered, perturbed_code, random_seed=iterations)
            ls_code = local_search(new_code)
            
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
