import random
import time
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    # Pre-compute neighbors
    neighbors = []
    weights = [q ** (n - 1 - i) for i in range(n)]
    
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
                new_idx = idx + (new_val - w[i]) * weights[i]
                neigh.append(new_idx)
        neighbors.append(np.array(neigh, dtype=np.int32))

    # Neighbor mask for fast checks
    neighbor_mask = np.zeros((total_words, total_words), dtype=bool)
    for idx in range(total_words):
        neighbor_mask[idx, neighbors[idx]] = True

    def idx_to_word(idx):
        w = [0] * n
        for d in range(n-1, -1, -1):
            w[d] = idx % q
            idx //= q
        return w

    def randomized_greedy(random_seed, k_candidates=10):
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.int8)
        code = []
        remaining_count = total_words
        
        # M matrix for gains
        M = np.zeros((total_words, total_words), dtype=np.int8)
        for idx in range(total_words):
            M[idx, neighbors[idx]] = 1

        while remaining_count > 0:
            gains = (M @ uncovered).astype(np.int32)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            
            # Pick randomly from top candidates
            if len(candidates) > k_candidates:
                # Get top k candidates by gain (all have max_gain here, so just random subset)
                # To be more diverse, we can pick from top 10% or similar, but for simplicity, random from all max gain
                chosen_idx = candidates[np.random.randint(len(candidates))]
            else:
                chosen_idx = candidates[np.random.randint(len(candidates))]
                
            code.append(chosen_idx)
            nb_idx = neighbors[chosen_idx]
            num_covered = np.sum(uncovered[nb_idx])
            uncovered[nb_idx] = 0
            remaining_count -= num_covered
            
        return code

    def local_search(code, random_seed):
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        code = list(code)
        coverage = np.zeros(total_words, dtype=np.int32)
        for idx in code:
            coverage[neighbors[idx]] += 1
        
        improved = True
        max_iters = 500
        iter_count = 0
        
        while improved and iter_count < max_iters:
            iter_count += 1
            improved = False
            
            # 1. Removal
            # Find indices where all neighbors have coverage > 1
            removable_indices = []
            for i, idx in enumerate(code):
                if np.min(coverage[neighbors[idx]]) > 1:
                    removable_indices.append(i)
            
            if removable_indices:
                rem_idx = random.choice(removable_indices)
                idx_to_remove = code[rem_idx]
                coverage[neighbors[idx_to_remove]] -= 1
                del code[rem_idx]
                improved = True
                continue
                
            # 2. Single Swap
            # Try to swap a codeword c with w such that the new set is still valid.
            # Ideally, we want a swap that enables a subsequent removal, but 
            # a simple validity-preserving swap helps escape local minima in the removal landscape.
            
            # We prioritize codewords with many "critical" neighbors (coverage == 1)
            # as they are the bottleneck.
            
            # Shuffle to avoid bias
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            swap_made = False
            for i in indices:
                c = code[i]
                crits = neighbors[c][coverage[neighbors[c]] == 1]
                
                if len(crits) == 0:
                    continue
                
                # Find w != c such that w covers all crits.
                # w must be in intersection of neighbors of all crits.
                # Since crits are small, we can iterate neighbors of the first crit.
                first_crit = crits[0]
                candidates_w = neighbors[first_crit]
                
                for w in candidates_w:
                    if w == c:
                        continue
                    # Check if w covers all other crits
                    if np.all(neighbor_mask[w, crits]):
                        # Check if swap is valid:
                        # After removing c, coverage of neighbors of c decreases by 1.
                        # After adding w, coverage of neighbors of w increases by 1.
                        # We need min(coverage[new_state]) >= 1 for all words.
                        
                        # Approximation: 
                        # Words covered ONLY by c (and not w) will become uncovered.
                        # Words covered ONLY by w (and not c) will be covered.
                        
                        # Let's simulate:
                        # new_coverage = coverage.copy()
                        # new_coverage[neighbors[c]] -= 1
                        # new_coverage[neighbors[w]] += 1
                        # if np.min(new_coverage) >= 1: valid
                        
                        # Optimization:
                        # The only words at risk are those in neighbors[c] \ neighbors[w]
                        # and those in neighbors[w] \ neighbors[c] are safe (gain coverage).
                        # Words in intersection are safe (coverage -1 +1 = same).
                        # Words outside both are safe.
                        
                        # So we only need to check if any word in (neighbors[c] \ neighbors[w]) has coverage == 1.
                        # If coverage[x] == 1 and x is in neighbors[c] and x NOT in neighbors[w], then x becomes 0.
                        
                        # Mask for neighbors of c
                        mask_c = neighbor_mask[c]
                        # Mask for neighbors of w
                        mask_w = neighbor_mask[w]
                        
                        # Words covered by c but not w
                        at_risk = mask_c & ~mask_w
                        
                        if np.any(at_risk):
                            # Check coverage of these words
                            if np.min(coverage[at_risk]) > 1:
                                # Safe to swap
                                code[i] = w
                                coverage[neighbors[c]] -= 1
                                coverage[neighbors[w]] += 1
                                improved = True
                                swap_made = True
                                break
                if swap_made:
                    break
            
            if swap_made:
                continue

        return code

    start_time = time.time()
    best_code = None
    best_size = float('inf')
    
    # Run multiple iterations with different seeds
    seed = 0
    while time.time() - start_time < 115:
        # Generate initial cover
        code = randomized_greedy(seed, k_candidates=5)
        seed += 1
        
        # Local search
        code = local_search(code, seed)
        seed += 1
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
    if best_code is None:
        code = randomized_greedy(0, k_candidates=1)
        best_code = code
        best_size = len(code)

    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
