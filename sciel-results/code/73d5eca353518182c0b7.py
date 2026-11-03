import random
import time
import sys

def construct(q, n, radius):
    total_words = q ** n
    
    # Pre-compute neighbors for radius 1
    # neighbors[idx] = list of indices within distance <= 1
    neighbors = [[] for _ in range(total_words)]
    # Pre-compute powers of q for index calculation
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
    # powers[i] is q^(n-1-i) ? No, let's align with standard base-q expansion.
    # idx = w[0]*q^(n-1) + ... + w[n-1]*q^0
    # So digit i has weight q^(n-1-i)
    
    for idx in range(total_words):
        w = [0] * n
        temp = idx
        for d in range(n-1, -1, -1):
            w[d] = temp % q
            temp //= q
            
        neigh = [idx]
        for i in range(n):
            for delta in range(1, q):
                new_val = (w[i] + delta) % q
                # Calculate difference in index
                # weight of position i is q^(n-1-i)
                weight = q ** (n - 1 - i)
                new_idx = idx + (new_val - w[i]) * weight
                neigh.append(new_idx)
        neighbors[idx] = neigh

    def word_to_idx(w):
        idx = 0
        for d in range(n):
            idx = idx * q + w[d]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for d in range(n-1, -1, -1):
            w[d] = idx % q
            idx //= q
        return w

    def greedy_construct(random_seed, top_k=5):
        random.seed(random_seed)
        uncovered = [1] * total_words
        code = []
        remaining_count = total_words
        
        # Pre-compute initial gains? No, dynamic.
        # To speed up, we can maintain a list of uncovered indices?
        # But checking gain for all 2187 words is fast enough.
        
        while remaining_count > 0:
            best_gain = -1
            candidates = []
            
            # Calculate gain for all words
            # Optimization: If we only pick from uncovered words, we might miss a covered word that covers many others?
            # No, a covered word's ball is a subset of the union of balls of words that cover it?
            # Actually, if a word is covered, it doesn't need to be covered again.
            # But adding a covered word *could* cover other uncovered words.
            # However, standard greedy for set cover usually considers all sets.
            # But for covering codes, it is often beneficial to pick from uncovered words?
            # Let's stick to picking from all words to be safe, but we can optimize.
            
            for idx in range(total_words):
                gain = 0
                for nb in neighbors[idx]:
                    if uncovered[nb]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    candidates = [idx]
                elif gain == best_gain:
                    candidates.append(idx)
                    
            if best_gain == 0:
                break
                
            # Randomized selection from top candidates
            # Pick randomly from the candidates with max gain
            chosen_idx = random.choice(candidates)
            
            code.append(chosen_idx)
            
            for nb in neighbors[chosen_idx]:
                if uncovered[nb]:
                    uncovered[nb] = 0
                    remaining_count -= 1
                    
        return code

    def local_search(code):
        # Convert to set for fast membership? No, we need to iterate.
        # Maintain coverage counts
        coverage = [0] * total_words
        for idx in code:
            for nb in neighbors[idx]:
                coverage[nb] += 1
        
        improved = True
        while improved:
            improved = False
            
            # 1. Greedy Removal
            # Try to remove any codeword
            # We iterate through a copy or handle indices carefully
            removable_indices = []
            for i in range(len(code)):
                idx = code[i]
                # Check if removing idx leaves any word with coverage 0
                # We can check: for all nb in neighbors[idx], if coverage[nb] == 1, then it becomes 0.
                # So if any nb has coverage 1, we cannot remove.
                can_remove = True
                for nb in neighbors[idx]:
                    if coverage[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    removable_indices.append(i)
            
            if removable_indices:
                # Remove one (or all? Removing one might make others removable. Let's remove one and restart loop)
                # Or remove all found? Removing all is faster but might be suboptimal if they interact?
                # If A and B are both removable, removing A doesn't change B's removability status regarding *other* words, 
                # but it changes the coverage counts.
                # If coverage[nb] was 2 for all nb in A, and 2 for all nb in B.
                # Removing A makes coverage 1. B is still removable if no nb in B has coverage 1 *after* A removed?
                # Actually, if we remove A, coverage drops by 1 for A's neighbors.
                # If B's neighbors had coverage 2, they become 1. B is still removable?
                # Yes, if all B's neighbors have coverage >= 2, removing A (which might overlap) reduces them.
                # If they become 1, B is still removable (coverage 1 is fine, we just can't remove B *if* it becomes 0).
                # Wait, if coverage[nb] becomes 1, removing B would make it 0. So B is NOT removable anymore.
                # So we should remove one at a time and re-evaluate.
                
                # Pick a random removable index to introduce diversity
                rem_idx = random.choice(removable_indices)
                idx_to_remove = code[rem_idx]
                
                for nb in neighbors[idx_to_remove]:
                    coverage[nb] -= 1
                del code[rem_idx]
                improved = True
                continue # Restart while loop to re-check removals
                
            # 2. Swaps
            # If no removal possible, try swaps.
            # Try to swap code[i] with some w such that the new code is valid.
            # Heuristic: Try to swap a codeword with a word that covers "critical" words (coverage 1) better?
            # Or just random swaps.
            
            # To speed up, we can limit the number of swap attempts per pass.
            swaps_made = 0
            # Shuffle code to avoid bias
            random.shuffle(code)
            
            for i in range(len(code)):
                c = code[i]
                # Try a few random candidates
                # We want to find a w such that:
                # 1. Removing c doesn't break coverage (i.e., all neighbors of c have coverage >= 2, OR
                #    the neighbors of c that have coverage 1 are covered by w).
                # 2. Adding w doesn't cause issues (it never does, it only adds coverage).
                # So the condition is: For all nb in neighbors[c] with coverage[nb] == 1, nb must be in neighbors[w].
                
                critical_neighbors = [nb for nb in neighbors[c] if coverage[nb] == 1]
                
                if not critical_neighbors:
                    # c is redundant? But we already checked removal.
                    # If c is redundant, it would have been removed in Phase 1.
                    # So this case should not happen if Phase 1 is thorough.
                    # However, Phase 1 only removes if *all* neighbors have coverage > 1.
                    # If c has neighbors with coverage 1, it's not removable.
                    pass
                
                # Try random swaps
                for _ in range(20):
                    w = random.randint(0, total_words - 1)
                    if w == c:
                        continue
                    
                    # Check if w covers all critical neighbors of c
                    valid_swap = True
                    for cnb in critical_neighbors:
                        if cnb not in neighbors[w]:
                            valid_swap = False
                            break
                    
                    if valid_swap:
                        # Perform swap
                        # Remove c
                        for nb in neighbors[c]:
                            coverage[nb] -= 1
                        # Add w
                        for nb in neighbors[w]:
                            coverage[nb] += 1
                        
                        code[i] = w
                        improved = True
                        swaps_made += 1
                        break # Restart while loop after a swap?
                        # Or continue with next i?
                        # Restarting is safer to re-evaluate removals.
            
            if swaps_made > 0:
                continue # Restart while loop

        return code

    # Main Iterated Local Search
    start_time = time.time()
    best_code = None
    best_size = float('inf')
    seed = 42
    
    # Run multiple iterations
    while time.time() - start_time < 110:
        # Generate random greedy cover
        code = greedy_construct(seed, top_k=10)
        seed += 1
        
        # Local search
        code = local_search(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
        # Optional: Perturb the best code and local search again?
        # For now, just restart with new seed.
        
    if best_code is None:
        # Fallback: Standard greedy
        code = greedy_construct(0, top_k=1)
        best_code = code
        best_size = len(code)

    # Convert to list of lists
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
