import itertools
import random
import numpy as np
import time
from collections import defaultdict

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words as a 2D numpy array (total_words, n)
        all_words = np.array(list(itertools.product(range(q), repeat=n)))
        
        # Precompute distance matrix and coverage
        # Coverage: for each word i, which words j are within Hamming distance <= radius
        # We need coverage sets for greedy and local search
        diff = np.abs(all_words[:, None, :] - all_words[None, :, :])
        dist = (diff != 0).sum(axis=2)
        
        # For radius 1, each word covers itself + words at distance 1
        # Number of words at distance 1 from a word of length n over alphabet q is n*(q-1)
        # So coverage size is 1 + n*(q-1) = 1 + 6*2 = 13
        coverage = [set() for _ in range(total_words)]
        for i in range(total_words):
            idxs = np.where(dist[i] <= radius)[0]
            coverage[i] = set(idxs.tolist())
        
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
            max_iterations = 500
            
            while improved and iterations < max_iterations:
                improved = False
                iterations += 1
                
                # 1. Pruning: remove codewords whose all covered words are covered by others
                removable = None
                for c in list(code_set):
                    if np.min(counts[list(coverage[c])]) > 1:
                        removable = c
                        break
                        
                if removable is not None:
                    code_set.remove(removable)
                    counts[list(coverage[removable])] -= 1
                    improved = True
                    continue
                
                # 2. Swap: try to replace a codeword with one that covers its private words
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
                    # Try to find a replacement word that covers all private words
                    for w in range(total_words):
                        if w in code_set:
                            continue
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

        best_code = None
        best_size = float('inf')
        
        start_time = time.time()
        time_limit = 110
        
        # Initial Greedy + Local Search
        initial_code = greedy_set_cover(set(range(total_words)), [])
        ls_code = local_search(initial_code)
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
                covered.update(coverage[c])
            uncovered = set(range(total_words)) - covered
            
            new_code = greedy_set_cover(uncovered, perturbed_code, random_seed=iterations)
            ls_code = local_search(new_code)
            
            if len(ls_code) < best_size:
                best_size = len(ls_code)
                best_code = ls_code
        
        return [list(all_words[i]) for i in best_code]
    
    else:
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
