import itertools
import random
import numpy as np

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words as a 2D numpy array (total_words, n)
        all_words = np.array(list(itertools.product(range(q), repeat=n)))
        
        # Precompute coverage using broadcasting
        # diff = abs(all_words[:, None, :] - all_words[None, :, :])
        # This is 729x729x6, which is ~3.2M elements. Manageable.
        # dist = (diff != 0).sum(axis=2)
        
        # To save memory/time, we can compute coverage sets directly or use bitsets.
        # Let's use sets for flexibility in local search, but build them fast.
        
        # Fast distance calculation
        # We can loop over positions and build masks? 
        # Or just use the numpy approach.
        
        diff = np.abs(all_words[:, None, :] - all_words[None, :, :])
        dist = (diff != 0).sum(axis=2)
        
        # coverage[i] is a set of indices j such that dist[i][j] <= radius
        coverage = [set() for _ in range(total_words)]
        for i in range(total_words):
            # Get indices where dist[i][j] <= radius
            idxs = np.where(dist[i] <= radius)[0]
            coverage[i] = set(idxs.tolist())
            
        # Greedy Set Cover
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_cover_count = -1
            for i in range(total_words):
                new_cover = len(coverage[i] & uncovered)
                if new_cover > best_cover_count:
                    best_cover_count = new_cover
                    best_idx = i
                    if best_cover_count == len(uncovered):
                        break
            
            if best_idx == -1:
                break
                
            code_indices.append(best_idx)
            uncovered -= coverage[best_idx]
            
        # Local Search
        code_set = set(code_indices)
        
        # Precompute reverse coverage? 
        # rev_coverage[w] = set of codewords that cover w? 
        # This changes as code_set changes. 
        # Better to maintain counts.
        
        def get_counts(codes):
            counts = np.zeros(total_words, dtype=int)
            for c in codes:
                counts[list(coverage[c])] += 1
            return counts
            
        counts = get_counts(code_set)
        
        improved = True
        iterations = 0
        max_iterations = 1000
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # 1. Pruning
            # Find any codeword c such that no word in coverage[c] has count 1
            # i.e., min(counts[w] for w in coverage[c]) > 1
            removable = None
            for c in list(code_set):
                # Check if all covered words have count > 1
                # If any has count 1, c is critical
                if np.min(counts[list(coverage[c])]) > 1:
                    removable = c
                    break
                    
            if removable is not None:
                code_set.remove(removable)
                # Update counts
                counts[list(coverage[removable])] -= 1
                improved = True
                continue
                
            # 2. Swaps
            # For each c in code_set:
            #   Find private words P_c = {w in coverage[c] | counts[w] == 1}
            #   If P_c is empty, c is removable (handled above)
            #   Find w not in code_set such that P_c is subset of coverage[w]
            #   If found, swap c with w if valid.
            
            # To make this faster, we can iterate through c's in random order
            current_list = list(code_set)
            random.shuffle(current_list)
            
            swapped = False
            for c in current_list:
                # Find private words
                private_words = [w for w in coverage[c] if counts[w] == 1]
                
                if not private_words:
                    # Should have been pruned, but maybe counts changed?
                    # If not private, it's removable.
                    code_set.remove(c)
                    counts[list(coverage[c])] -= 1
                    improved = True
                    swapped = True
                    break
                
                # Find candidate w
                # w must cover all private_words
                # And w should ideally cover other words to maintain validity
                # Actually, if w covers all private words, and we remove c,
                # the only words that lose coverage are the private words of c.
                # If w covers them, they are still covered.
                # Other words covered by c are covered by others (count >= 2).
                # So the swap is valid IF w covers all private words.
                
                # Optimization: 
                # We can precompute for each word w, the set of private words it covers? No.
                # Just iterate through all w not in code_set.
                
                found_w = None
                for w in range(total_words):
                    if w in code_set:
                        continue
                    # Check if w covers all private_words
                    # Since private_words is small (usually), this is fast
                    covers_all = True
                    for pw in private_words:
                        if pw not in coverage[w]:
                            covers_all = False
                            break
                    if covers_all:
                        found_w = w
                        break
                        
                if found_w is not None:
                    # Perform swap
                    code_set.remove(c)
                    counts[list(coverage[c])] -= 1
                    code_set.add(found_w)
                    counts[list(coverage[found_w])] += 1
                    improved = True
                    swapped = True
                    break
                    
            if not swapped:
                # Try random restart or more aggressive search?
                # For now, break if no improvement in this pass
                # To avoid infinite loops, we break.
                # But we might be stuck in local minimum.
                # Let's add a random perturbation if stuck for many iterations?
                # Given time limits, let's just break.
                break
                
        return [list(all_words[i]) for i in code_set]
    
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
