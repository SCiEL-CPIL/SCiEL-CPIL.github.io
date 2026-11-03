import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115
    
    total_words = q ** n
    cover_count = 1 + n * (q - 1)
    
    # Precompute neighbors
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    for i in range(total_words):
        idxs = [i]
        for pos in range(n):
            original_val = (i // (q ** (n - 1 - pos))) % q
            for new_val in range(q):
                if new_val == original_val:
                    continue
                j = i + (new_val - original_val) * (q ** (n - 1 - pos))
                idxs.append(j)
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx
    
    # Build inverse neighbors
    inv_neighbor_lists = [[] for _ in range(total_words)]
    for i in range(total_words):
        for j in range(cover_count):
            neighbor_idx = neighbors_arr[i, j]
            inv_neighbor_lists[neighbor_idx].append(i)
    
    def idx_to_list(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return list(reversed(t))
    
    def greedy_construct(random_seed, top_k=5):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any() and time.time() < deadline - 10:
            # Calculate gain for all words
            gains = np.sum(uncovered[neighbors_arr], axis=1)
            
            # Find max gain
            max_gain = np.max(gains)
            
            # Get all candidates with max gain
            best_candidates = np.where(gains == max_gain)[0]
            
            # If too many, pick from top K to diversify
            if len(best_candidates) > top_k:
                chosen = random.choice(best_candidates.tolist())
            else:
                chosen = random.choice(best_candidates.tolist())
                
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
            
        return code
    
    def reduce_code(code):
        if not code:
            return code
        
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors_arr[c, :]] -= 1
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code
    
    def local_search(code, dl):
        code = list(code)
        if not code:
            return code
            
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        # Use a queue or just iterate multiple times until no improvement
        while improved and time.time() < dl:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= dl:
                    break
                
                c_remove = code[i]
                neighbors_c = neighbors_arr[c_remove, :]
                counts[neighbors_c] -= 1
                
                if np.any(counts == 0):
                    holes = np.where(counts == 0)[0]
                    
                    # Find candidates that cover all holes
                    # Intersection of inv_neighbor_lists[h] for h in holes
                    # Start with the hole with the fewest candidates
                    min_hole = min(holes, key=lambda h: len(inv_neighbor_lists[h]))
                    candidates = set(inv_neighbor_lists[min_hole])
                    
                    for h in holes:
                        if h == min_hole:
                            continue
                        candidates &= set(inv_neighbor_lists[h])
                        if not candidates:
                            break
                    
                    candidates.discard(c_remove)
                    
                    if not candidates:
                        counts[neighbors_c] += 1
                        continue
                    
                    best_cand = None
                    best_score = -1
                    cand_list = list(candidates)
                    
                    # Limit candidates to avoid slow scoring
                    if len(cand_list) > 20:
                        cand_list = random.sample(cand_list, 20)
                        
                    for cand in cand_list:
                        cand_neigh = neighbors_arr[cand, :]
                        # Score: how many words does this candidate cover that are already covered?
                        # We want to minimize redundancy, so we want a candidate that covers words with low counts?
                        # Or simply pick one that covers the most "extra" coverage?
                        # Actually, any valid candidate restores coverage.
                        # To reduce future removal difficulty, we might prefer one that is "central" or has high overlap.
                        score = np.sum(counts[cand_neigh])
                        if score > best_score:
                            best_score = score
                            best_cand = cand
                            
                    if best_cand is not None:
                        code[i] = best_cand
                        counts[neighbors_arr[best_cand, :]] += 1
                        improved = True
                        break # Restart the loop to handle the new state
                    else:
                        counts[neighbors_c] += 1
                else:
                    # Safe to remove
                    code.pop(i)
                    improved = True
                    break # Restart the loop
        return code
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline - 5:
        # Generate initial code
        code = greedy_construct(seed)
        
        # Reduce
        code = reduce_code(code)
        
        # Local Search
        if time.time() < deadline - 10:
            code = local_search(code, deadline - 5)
            code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            
        # If we have a good code, try perturbing it
        if time.time() < deadline - 20 and best_code is not None:
            # Perturb: remove a few random codewords
            k = random.randint(2, 5)
            if len(best_code) > k:
                perturbed = list(best_code)
                remove_indices = random.sample(range(len(perturbed)), k)
                for idx in sorted(remove_indices, reverse=True):
                    perturbed.pop(idx)
                
                # Repair
                counts = np.zeros(total_words, dtype=np.int16)
                for c in perturbed:
                    counts[neighbors_arr[c, :]] += 1
                uncovered = counts == 0
                
                while uncovered.any() and time.time() < deadline - 10:
                    gains = np.sum(uncovered[neighbors_arr], axis=1)
                    max_gain = np.max(gains)
                    if max_gain == 0:
                        break
                    best_candidates = np.where(gains == max_gain)[0]
                    chosen = random.choice(best_candidates.tolist())
                    perturbed.append(chosen)
                    counts[neighbors_arr[chosen, :]] += 1
                    uncovered = counts == 0
                
                # Polish
                perturbed = reduce_code(perturbed)
                if time.time() < deadline - 10:
                    perturbed = local_search(perturbed, deadline - 5)
                    perturbed = reduce_code(perturbed)
                
                if len(perturbed) < best_size:
                    best_size = len(perturbed)
                    best_code = perturbed
        
        seed += 1
    
    result = []
    if best_code:
        for idx in best_code:
            result.append(idx_to_list(idx))
    else:
        # Fallback: should not happen
        for idx in range(total_words):
            result.append(idx_to_list(idx))
            
    return result
