# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if n == 0:
        return [[]]

    # Specific optimization for the known hard case: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Fallback for other cases
    # Use a simple greedy or direct sum if applicable
    if q == 3 and radius == 1 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return code
    
    # Generic greedy for small n
    return _greedy_covering(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    N = q ** n # 729
    
    # Precompute neighbors for each word index
    # neighbors[i] is a list of indices covered by word i (distance <= radius)
    neighbors = [None] * N
    for i, w in enumerate(all_words):
        neigh = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbors[i] = neigh
        
    # Initial code: Direct Sum (Tetracode x 3^2)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    # Convert to indices
    code_indices = [word_to_idx[tuple(w)] for w in init_code_words]
    
    # Coverage counts
    counts = np.zeros(N, dtype=np.int32)
    for idx in code_indices:
        for n_idx in neighbors[idx]:
            counts[n_idx] += 1
            
    # Local Search
    # We maintain a list of active indices
    active_indices = list(code_indices)
    
    # Helper to check if a word can be removed
    # A word i can be removed if min(counts[neighbors[i]]) > 1
    # Actually, we just need to ensure that for all j in neighbors[i], counts[j] > 1
    # Since counts[i] is always >= 1 if i is in code, and we are checking removal of i,
    # we check if any neighbor has count 1. If so, i is critical.
    
    def can_remove(idx):
        for n_idx in neighbors[idx]:
            if counts[n_idx] == 1:
                return False
        return True

    # Removal Loop
    # We use a strategy of repeated removals, and if stuck, we try perturbation (add random, remove 2)
    # We also try different random orders to escape local minima.
    
    best_size = len(active_indices)
    best_code = list(active_indices)
    
    # Main loop
    while time.time() < deadline:
        removed_any = False
        
        # Shuffle to break symmetry/bias
        random.shuffle(active_indices)
        
        for idx in active_indices:
            if time.time() > deadline:
                break
            
            if can_remove(idx):
                # Remove idx
                for n_idx in neighbors[idx]:
                    counts[n_idx] -= 1
                active_indices.remove(idx)
                removed_any = True
                
        if len(active_indices) < best_size:
            best_size = len(active_indices)
            best_code = list(active_indices)
            
        if not removed_any:
            # Stuck. Try a perturbation: Add a random word, then try to remove.
            # This allows escaping local minima.
            
            # Pick a random word from the universe
            random_idx = random.randint(0, N - 1)
            
            # If it's already in the code, skip (or pick another)
            if random_idx in active_indices:
                continue
                
            # Add it
            active_indices.append(random_idx)
            for n_idx in neighbors[random_idx]:
                counts[n_idx] += 1
                
            # Now try to remove redundant words
            # We want to see if we can remove MORE than 1 word (the one we added? No, we added 1, so we need to remove 2 to improve size, or 1 to stay same but maybe get to a better state? 
            # Usually we just try to minimize size.
            # If we add 1, size increases by 1.
            # If we can remove 2, net change -1.
            # If we can remove 1, net change 0.
            
            removed_count = 0
            # We need to be careful: we should prioritize removing the newly added one if it becomes redundant? 
            # Or just remove any redundant ones.
            
            # Let's try to remove up to 2 words.
            # We iterate through a copy of the list to avoid modification during iteration issues, 
            # but we are modifying active_indices. 
            # Better: create a list of candidates to remove.
            
            candidates = []
            for idx in active_indices:
                if can_remove(idx):
                    candidates.append(idx)
            
            # We want to remove the most "redundant" ones? 
            # Or just remove 2 if possible.
            # To improve the score (minimize size), we need net removal.
            # If we can remove >= 2 words, we accept the perturbation.
            # If we can remove 1, we are at same size. We might accept it with some probability to explore, but for now, let's just accept if size decreases.
            
            # Sort candidates? 
            # Maybe remove the one that is "most covered" (highest min count in neighborhood)?
            # For simplicity, just try to remove 2.
            
            if len(candidates) >= 2:
                # Remove 2
                # Pick 2 distinct indices
                # To be safe, pick the first two
                to_remove = candidates[:2]
                
                # Verify they are still removable after removing the first?
                # Removing one might make the other non-removable?
                # If we remove A, counts decrease. If B was removable before, it might become critical?
                # No, counts only decrease. If B was removable (all neighbors count > 1), 
                # and we remove A, counts in N(A) decrease.
                # If N(B) intersects N(A), counts in intersection decrease.
                # If a count drops to 1, B might no longer be removable.
                # So we must check sequentially.
                
                # Let's try removing the first candidate.
                idx1 = to_remove[0]
                # Check if idx1 is still removable (it was)
                # Remove idx1
                for n_idx in neighbors[idx1]:
                    counts[n_idx] -= 1
                active_indices.remove(idx1)
                
                # Now check if we can remove another one.
                # We can check if idx2 (the second candidate) is still removable.
                # Or just scan for any new removable word.
                # Scanning is safer.
                
                for idx in active_indices:
                    if can_remove(idx):
                        for n_idx in neighbors[idx]:
                            counts[n_idx] -= 1
                        active_indices.remove(idx)
                        removed_count = 2
                        break
                        
            elif len(candidates) == 1:
                # Remove 1. Net change 0.
                # We might do this to explore state space, but it doesn't improve size immediately.
                # However, it might lead to a state where 2 can be removed later.
                idx1 = candidates[0]
                for n_idx in neighbors[idx1]:
                    counts[n_idx] -= 1
                active_indices.remove(idx1)
                removed_count = 1

            # If we removed 2, we improved.
            # If we removed 1, we are same size.
            # If we removed 0, we added 1 and couldn't remove. Size increased by 1.
            # We should revert if size increased?
            # Or just keep going?
            # If size increased, we are worse. We should probably revert to the previous best state?
            # Or just let the next removal loop try to fix it.
            
            # To be safe, if size increased, we revert.
            # But we don't have a snapshot.
            # Let's just track the best code found so far.
            
            if len(active_indices) < best_size:
                best_size = len(active_indices)
                best_code = list(active_indices)
            
            # If we got stuck and didn't improve, we might want to restart from best_code?
            # Or just continue.
            
            # To avoid getting stuck in a large code, if we added 1 and removed 0, 
            # we are at size + 1.
            # Let's check if we should revert.
            # If we removed 0, we added 1.
            # If we removed 1, we added 1 and removed 1.
            # If we removed 2, we added 1 and removed 2.
            
            # If removed_count < 1, we increased size.
            # Let's revert to best_code if we increased size?
            # Reverting is expensive (rebuild counts).
            # Let's just rely on the removal loop to shrink it back.
            
            # But if we keep adding and not removing, we might time out with a large code.
            # So, if we increased size, we should probably revert.
            
            # Let's implement a simple revert:
            # If len(active_indices) > best_size, we revert to best_code.
            
            if len(active_indices) > best_size:
                # Revert
                # Rebuild counts from best_code
                counts = np.zeros(N, dtype=np.int32)
                for idx in best_code:
                    for n_idx in neighbors[idx]:
                        counts[n_idx] += 1
                active_indices = list(best_code)
                
            # If we are still stuck (no removals in this perturbation cycle), we might want to restart?
            # The loop continues, so we will try another perturbation.

    return [list(idx_to_word[i]) for i in best_code]

def _greedy_covering(q, n, radius, deadline):
    """
    Greedy construction of a covering code.
    """
    if n > 8:
        # Fallback: Direct sum if possible, else random
        if q == 3:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             if n >= 4:
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
             else:
                 pass
        
        if n <= 8:
            all_words = list(itertools.product(range(q), repeat=n))
            uncovered = set(all_words)
            code = []
            
            def get_neighbors(c):
                neighbors = {c}
                for i in range(n):
                    orig = c[i]
                    for val in range(q):
                        if val == orig: continue
                        lst = list(c)
                        lst[i] = val
                        neighbors.add(tuple(lst))
                return neighbors

            while uncovered and time.time() < deadline:
                best_word = None
                best_count = -1
                
                if n <= 6:
                    # Cache neighbors for speed
                    neighbors_map = {w: get_neighbors(w) for w in all_words}
                    for w in all_words:
                        cnt = sum(1 for x in neighbors_map[w] if x in uncovered)
                        if cnt > best_count:
                            best_count = cnt
                            best_word = w
                else:
                    best_word = random.choice(all_words)
                    
                if best_word is None or best_count == 0:
                    break
                    
                code.append(list(best_word))
                uncovered.difference_update(get_neighbors(best_word))
                
            return code
        else:
            if q == 3 and n >= 4:
                 tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            else:
                 return [[0]*n]
# EVOLVE-BLOCK-END
