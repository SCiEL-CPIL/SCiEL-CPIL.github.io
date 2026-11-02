import time
import random
import numpy as np

def construct(q, n, radius):
    # This task is specifically for q=3, n=7, radius=1.
    # We handle the general case by falling back to a simple construction if parameters differ,
    # but optimize for the specific case.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Direct sum of tetracode if n >= 4 and q=3
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            import itertools
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: all-zero codeword covers only 1 word, so we need many.
            # Just return a large set to be safe, though score will be bad.
            return [[0] * n]

    # Specific optimization for K_3(7, 1)
    start_time = time.time()
    time_limit = 145.0 # Safety margin before 150s kill
    
    total_words = 3 ** n
    # Map word (tuple of ints) to index 0..total_words-1
    # We use base-3 representation
    
    def to_index(word):
        idx = 0
        for i in range(n):
            idx = idx * 3 + word[i]
        return idx

    def from_index(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word

    # Precompute spheres of radius 1 for all indices
    # Sphere of idx: idx itself and all indices differing in exactly 1 position
    indices = list(range(total_words))
    words_list = [from_index(i) for i in indices]
    
    # Precompute the list of neighbors for each index
    # For n=7, radius=1, sphere size is 1 + 7*2 = 15
    spheres = []
    for idx in indices:
        word = words_list[idx]
        neigh = [idx]
        for pos in range(n):
            original_val = word[pos]
            for delta in [1, 2]:
                new_val = (original_val + delta) % 3
                new_word = word[:]
                new_word[pos] = new_val
                neigh.append(to_index(new_word))
        spheres.append(neigh)

    sphere_size = len(spheres[0])
    spheres_np = np.array(spheres, dtype=np.int32)
    
    # Precompute the transpose of the incidence matrix for fast gain/loss calculation
    # M[i, j] = 1 if word j is in sphere of word i
    # We want to calculate counts of covered words by a set of centers.
    # Let x be a binary vector of centers.
    # Covered = M @ x > 0
    # For swap evaluation, we need to know how many words are ONLY covered by a specific center (loss)
    # and how many words are NOT covered by any other center but WOULD be covered by a new center (gain).
    
    def greedy_construct(seed, time_budget):
        random.seed(seed)
        uncovered = np.ones(total_words, dtype=bool)
        code_indices = []
        
        # Randomize initial order to get diverse starts
        candidates = list(range(total_words))
        random.shuffle(candidates)
        
        while np.any(uncovered):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Find the index that covers the most uncovered words
            # We iterate through all candidates to find the best one.
            # To speed this up, we can just iterate all 2187 words.
            best_idx = -1
            max_count = -1
            
            # Optimization: Use numpy to compute coverage counts for all potential centers at once?
            # counts[i] = sum(uncovered[spheres_np[i]])
            # This is a matrix-vector product if uncovered is a vector?
            # spheres_np is (N, S). uncovered is (N,).
            # We want row sums of (spheres_np[i, :] * uncovered) ? No, that's not quite right for boolean sum.
            # Actually, M is the incidence matrix (N, N) where M[i, j] = 1 if j in sphere(i).
            # Then counts = M @ uncovered.
            # Let's construct M once.
            
            best_idx = -1
            max_count = -1
            for i in candidates:
                # Only consider if it covers at least one uncovered word
                # This check is cheap if we maintain a list of active centers?
                # For now, just compute sum.
                c = np.sum(uncovered[spheres_np[i]])
                if c > max_count:
                    max_count = c
                    best_idx = i
                    if c == sphere_size:
                        # Can't get better than full sphere coverage
                        pass
            
            if best_idx == -1 or max_count == 0:
                break
                
            code_indices.append(best_idx)
            uncovered[spheres_np[best_idx]] = False
            
        return code_indices

    # Precompute Incidence Matrix M (N x N) where M[i, j] = 1 if j is in sphere(i)
    # This allows fast matrix-vector products for coverage counts.
    M = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        for j in spheres_np[i]:
            M[i, j] = 1
            
    def local_search(code_list, time_budget):
        """
        Performs local search using swap operations.
        Returns the best code found.
        """
        if not code_list:
            return code_list
            
        # Use a set for fast membership checks
        code_set = set(code_list)
        # Binary vector of centers
        x = np.zeros(total_words, dtype=np.int8)
        for idx in code_list:
            x[idx] = 1
            
        # Coverage counts: how many centers cover each word
        cover_count = M @ x
        
        # Track best
        best_code = list(code_list)
        best_size = len(code_list)
        
        # We will perform swaps: remove i, add j.
        # Loss of removing i: number of words where cover_count becomes 0.
        #   This is sum(1 for w in sphere(i) if cover_count[w] == 1)
        # Gain of adding j: number of words where cover_count becomes >0 from 0.
        #   This is sum(1 for w in sphere(j) if cover_count[w] == 0)
        
        # Precompute "critical" counts for current state?
        # Let's just evaluate swaps iteratively.
        
        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            # Try all pairs (i in code, j not in code)
            # This is O(|Code| * (N - |Code|)) which is large (186 * 2000 ~ 372k).
            # Each eval is O(1) if we have precomputed arrays, but we need to update cover_count.
            # Actually, we can vectorize the evaluation of all j for a fixed i.
            
            # For each i in code:
            #   loss_i = sum(cover_count[sphere(i)] == 1)
            #   For all j not in code:
            #     gain_j = sum(cover_count[sphere(j)] == 0)
            #   If gain_j > loss_i, swap.
            
            # We can compute gain_j for all j at once?
            # gain_j = sum(M[j, :] * (cover_count == 0))
            # Let zero_mask = (cover_count == 0)
            # gains = M @ zero_mask
            
            zero_mask = (cover_count == 0).astype(np.int8)
            gains = M @ zero_mask
            
            # For each i in code, calculate loss
            # loss_i = sum(cover_count[sphere(i)] == 1)
            # one_mask = (cover_count == 1).astype(np.int8)
            # losses = M @ one_mask  -> This gives count of words in sphere(i) that are covered exactly once.
            
            one_mask = (cover_count == 1).astype(np.int8)
            losses = M @ one_mask
            
            # Find best swap
            best_i = -1
            best_j = -1
            best_net_gain = 0
            
            # Iterate i in code
            for i in code_list:
                if i not in code_set: continue # Should be in set
                li = losses[i]
                if li == 0:
                    # Removing i causes no loss? Then we can just delete it? 
                    # Wait, if li==0, it means no word is covered ONLY by i. So i is redundant.
                    # This is a deletion, which is a swap with j = None (gain 0, loss 0? No, size decreases).
                    # A deletion is better than a swap (size -1 vs size 0 change).
                    # If we can delete, do it.
                    # Check if cover_count[sphere(i)] > 1 for all w in sphere(i)?
                    # If li == 0, then min(cover_count[sphere(i)]) >= 2. So yes, redundant.
                    # Perform deletion
                    code_set.remove(i)
                    x[i] = 0
                    cover_count -= M[i, :]
                    best_code = list(code_set)
                    best_size -= 1
                    improved = True
                    # Restart loop to recompute masks
                    break 
                
                # We want to find j not in code_set such that gains[j] > li
                # And we want to maximize gains[j] - li (which is just maximizing gains[j] since li is fixed)
                # But we also need to ensure j is not in code.
                
                # Create a mask of valid j's (not in code)
                # We can just iterate j? Or use numpy.
                # Let's find the max gain among j not in code.
                
                # To avoid O(N) python loop for each i, we can do this:
                # valid_gains = gains.copy()
                # for k in code_set: valid_gains[k] = -1
                # max_gain, best_j = np.max(valid_gains), np.argmax(valid_gains)
                
                # But modifying array for each i is O(N).
                # Total O(|Code| * N) ~ 186 * 2187 ~ 400k ops. This is fast in numpy if vectorized properly.
                
                # Let's try a different approach:
                # Iterate j not in code, check if gains[j] > min_loss?
                # No, let's stick to iterating i, and finding best j.
                
                # Optimization: Precompute list of non-code indices?
                # It changes on every swap.
                
                # Let's just loop j in non_code_indices.
                # To speed up, we can check if max(gains[non_code_indices]) > li.
                
                # For now, simple loop. 400k iterations of simple numpy scalar access might be slow in Python.
                # Let's try to vectorize the "find best j" part.
                
                # Create a boolean mask for "not in code"
                # This is expensive to create every time.
                
                # Alternative:
                # We know losses[i]. We want max(gains[j]) for j not in code.
                # If we sort gains, we can pick the top ones and check if they are in code.
                # Sort indices by gain descending.
                # sorted_indices = np.argsort(gains)[::-1]
                # For a given i, iterate sorted_indices until we find j not in code_set.
                # Since gains are small integers (0..15), we can bucket them.
                
                # Let's use the bucket approach for gains.
                # Max gain is 15.
                
                # This local search logic is getting complex. Let's simplify.
                # Standard Local Search:
                # 1. Compute gains and losses.
                # 2. Find best swap (i, j) maximizing gains[j] - losses[i] subject to j not in code.
                #    Also consider deletions (j=None) if losses[i] == 0.
                
                # Let's just iterate all i in code and all j not in code.
                # To make it fast, we will use a precomputed list of non-code indices.
                
                pass # Placeholder for logic refinement
            if improved:
                continue # Restart while loop to recompute masks after deletion
                
            # If no deletion found, look for swaps
            # We need to find (i, j) such that gains[j] > losses[i]
            
            # Let's find the best swap globally.
            # For each i, best_j_i is the j (not in code) with max gains[j].
            # Then check if gains[best_j_i] > losses[i].
            
            # To get best_j_i for all i efficiently:
            # We can't just take global max gain because that j might be the same for all i, 
            # but we only care if it's better than specific loss[i].
            
            # Let's iterate i in code.
            for i in code_list:
                li = losses[i]
                if li == 0:
                    # Should have been caught in deletion phase, but just in case
                    code_set.remove(i)
                    x[i] = 0
                    cover_count -= M[i, :]
                    best_code = list(code_set)
                    best_size -= 1
                    improved = True
                    break
                    
                # Find j not in code with max gains[j]
                # We can iterate through all j? 
                # Let's just pick the j with global max gain that is not in code.
                # If that global max gain is > li, we do the swap.
                # Is it possible that a j with lower gain is the only one not in code?
                # Yes.
                
                # So we need the max gain among j not in code.
                # Let's compute this once per iteration of the outer while loop?
                # No, gains change if we swap.
                
                # Let's just find the best swap for this i.
                # Iterate j in range(total_words):
                #   if j in code_set: continue
                #   if gains[j] > li:
                #       candidate_net = gains[j] - li
                #       if candidate_net > best_net_gain:
                #           best_net_gain = candidate_net
                #           best_i = i
                #           best_j = j
                #           if best_net_gain > 1: # Can't get better than 15-0? No, max gain 15, min loss 1. Max net 14.
                #               break # Found a very good swap? No, we want max.
                
                # This inner loop is O(N). Total O(|Code| * N).
                # 186 * 2187 ~ 400k.
                # Python loop of 400k is ~0.1s. Acceptable.
                
                for j in range(total_words):
                    if j in code_set:
                        continue
                    gj = gains[j]
                    if gj > li:
                        net = gj - li
                        if net > best_net_gain:
                            best_net_gain = net
                            best_i = i
                            best_j = j
                            # Can't get better than 15 (max gain) - 1 (min loss) = 14?
                            # Actually max gain is 15, min loss is 1.
                            # If we find net == 14, it's the best possible.
                            if net >= 14:
                                break
                if best_net_gain >= 14:
                    break # Found best possible swap for this i? No, for the whole set?
                    # If we break the i loop, we might miss a better swap for another i?
                    # No, net gain is bounded by 14. If we found 14, it's optimal.
            
            if best_net_gain > 0:
                # Perform swap
                code_set.remove(best_i)
                code_set.add(best_j)
                x[best_i] = 0
                x[best_j] = 1
                cover_count -= M[best_i, :]
                cover_count += M[best_j, :]
                best_code = list(code_set)
                # Size remains same
                improved = True
            else:
                break # No improvement found
                
        return best_code

    best_code_found = None
    best_size = float('inf')
    
    # Multi-restart strategy
    # Run several greedy constructions with different seeds
    # Then apply local search to the best ones
    
    num_restarts = 5
    time_per_restart = (time_limit - 10) / num_restarts
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Greedy phase
        code = greedy_construct(42 + r, time_per_restart / 2)
        
        if not code:
            continue
            
        # Local Search Phase
        # Limit time for local search
        ls_start = time.time()
        ls_time_limit = time_per_restart / 2
        # Modify local_search to respect global time limit
        code = local_search(code, ls_time_limit)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code_found = code
            
        # If we reached 186, we can stop? 
        # 186 is the known best. If we hit it, it's very good.
        if best_size <= 186:
            break

    if best_code_found is None:
        # Fallback to a simple greedy result if everything failed
        best_code_found = greedy_construct(0, 10)
        
    return [words_list[i] for i in best_code_found]
