import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit
    
    # Specific parameters for this task: q=3, n=8, radius=1
    if not (q == 3 and n == 8 and radius == 1):
        # Fallback for other parameters
        # For q=3, n >= 4, use tetracode construction
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Generic small case fallback
        centers = []
        covered = set()
        all_words = set(itertools.product(range(q), repeat=n))
        candidates = list(all_words)
        random.shuffle(candidates)
        
        while covered != all_words and time.time() < deadline:
            best_center = None
            max_new = 0
            for c in candidates:
                neigh = [c]
                for i in range(n):
                    for val in range(q):
                        if val == c[i]: continue
                        new_c = list(c)
                        new_c[i] = val
                        neigh.append(tuple(new_c))
                new_covered = [w for w in neigh if w not in covered]
                if len(new_covered) > max_new:
                    max_new = len(new_covered)
                    best_center = c
                    best_neigh = neigh
            if best_center is None or max_new == 0:
                break
            centers.append(list(best_center))
            for w in best_neigh:
                covered.add(w)
            candidates.remove(best_center)
        return centers

    # --- Optimized Path for q=3, n=8, radius=1 ---
    
    total_words = 3 ** 8  # 6561
    
    # 1. Pre-compute mapping and neighbors
    word_to_idx = {}
    idx_to_word = [None] * total_words
    
    # Generate all words and map to indices
    for word in itertools.product(range(3), repeat=8):
        idx = 0
        for digit in word:
            idx = idx * 3 + digit
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    # Pre-compute neighbors for each word (including itself)
    # neighbors[i] is a list of indices of words within distance 1 of idx_to_word[i]
    neighbors = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        center = idx_to_word[i]
        neighbors[i].append(i)
        for pos in range(8):
            old_val = center[pos]
            for new_val in range(3):
                if new_val == old_val:
                    continue
                neighbor = list(center)
                neighbor[pos] = new_val
                n_idx = word_to_idx[tuple(neighbor)]
                neighbors[i].append(n_idx)
    
    # 2. Initial Code: Direct Sum Construction (Tetracode x {0,1,2}^4)
    # Tetracode: 9 codewords of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_code_indices = []
    for t in tetracode:
        for tail in itertools.product(range(3), repeat=4):
            full_word = t + list(tail)
            idx = word_to_idx[tuple(full_word)]
            initial_code_indices.append(idx)
            
    code_set = set(initial_code_indices)
    
    # 3. Setup for Simulated Annealing / Local Search
    # counts[w] = number of centers in code_set covering w
    counts = np.zeros(total_words, dtype=np.int32)
    
    def update_counts(idx, delta):
        for n_idx in neighbors[idx]:
            counts[n_idx] += delta

    for idx in code_set:
        update_counts(idx, 1)
        
    # Check initial validity
    min_count = np.min(counts)
    if min_count < 1:
        # Fallback if initial code is invalid (should not happen for this construction)
        return [list(idx_to_word[idx]) for idx in code_set]

    best_len = len(code_set)
    best_code = set(code_set)
    best_counts = counts.copy()
    
    # Candidate pool: We restrict swaps to a subset of "plausible" centers to speed up search?
    # Actually, for q=3, n=8, the space is 6561. 
    # A swap involves removing one center and adding one.
    # To reduce size, we must find a sequence of swaps that allows a removal, or find a direct removal.
    # Direct removal: Remove c if min(counts[w] for w in N(c)) > 1.
    
    # Strategy: 
    # 1. Try to remove centers.
    # 2. If stuck, perform a swap (remove c1, add c2) that maintains coverage but potentially improves "removability".
    #    A good swap is one that increases the minimum count of words covered by the removed center's neighbors?
    #    Or simply a random valid swap to escape local minima.
    
    # We will use a hill-climbing with random restarts/swaps.
    # Objective: Minimize |code_set|.
    # Constraint: min(counts) >= 1.
    
    # To make swaps faster, we precompute for each center c, the set of words it covers.
    # We also need to know, for a candidate replacement c_add, if it covers all "critical" words of c_remove.
    
    # Let's use a more aggressive local search:
    # Iterate:
    #   1. Find all removable centers.
    #   2. If any, remove one (randomly or the one that minimizes "loss" of redundancy).
    #   3. If none, find a swap (c_remove, c_add) such that:
    #      - c_add not in code_set
    #      - For all w in N(c_remove) with counts[w]==1, w is in N(c_add)
    #      - AND (Heuristic) The swap increases the number of words with count==2 or more?
    #        Or simply: accept any valid swap with probability p (annealing).
    
    # To speed up finding c_add:
    # If c_remove has critical words W_crit = {w in N(c_remove) | counts[w]==1},
    # c_add must be in intersection of N(w) for all w in W_crit.
    # If W_crit is empty, c_add can be any valid center (any word not in code_set).
    # If W_crit is not empty, c_add is restricted.
    
    # Let's precompute "critical words" for each center? No, counts change.
    
    # Optimization: 
    # Instead of checking all 6561 candidates for c_add, we can intersect the neighbor lists of critical words.
    # Since |N(w)| = 18, and |W_crit| is small (usually 1-3), the intersection is small.
    
    def get_valid_swaps(c_remove):
        """Returns list of c_add candidates that allow removal of c_remove without breaking coverage."""
        # Find critical words for c_remove
        critical_words = []
        for w in neighbors[c_remove]:
            if counts[w] == 1:
                critical_words.append(w)
        
        if not critical_words:
            # No critical words. Any word not in code_set is a valid swap target?
            # Wait, if we swap c_remove with c_add, we remove c_remove and add c_add.
            # If c_remove has no critical words, removing it is safe.
            # So we can just remove it? 
            # Ah, if we just remove it, size decreases by 1. That's a "removal", not a "swap".
            # My loop structure should handle removals first.
            # If we are in the "swap" phase, it means no single removal is possible.
            # So c_remove MUST have critical words.
            # Thus this case shouldn't happen if we check removability first.
            return []
        
        # Intersection of neighbors of all critical words
        # Start with neighbors of first critical word
        candidates = set(neighbors[critical_words[0]])
        
        for w in critical_words[1:]:
            candidates.intersection_update(neighbors[w])
            if not candidates:
                break
        
        # Remove current code members
        candidates -= code_set
        
        return list(candidates)

    # Main Loop
    # We alternate between "Removal" and "Swap" phases.
    # To avoid getting stuck in a specific local minimum, we use randomization.
    
    current_len = len(code_set)
    stuck_counter = 0
    max_stuck = 2000 # If we do 2000 swaps/removals without reducing size, restart or break?
    # Actually, if we do swaps, size stays same. We only reduce on removal.
    # So "stuck" means "no removal found".
    
    while time.time() < deadline - 2:
        # Phase 1: Try to remove a center
        removed = False
        # Find all removable centers
        # A center c is removable if min(counts[w] for w in N(c)) > 1
        # To speed up, we can sample centers.
        # Or scan all? 729 centers * 18 checks = ~13k ops. Fast.
        
        removable_centers = []
        for c in code_set:
            # Check if any neighbor has count 1
            is_removable = True
            for w in neighbors[c]:
                if counts[w] == 1:
                    is_removable = False
                    break
            if is_removable:
                removable_centers.append(c)
                
        if removable_centers:
            # Pick one to remove
            # Heuristic: Remove the one that leaves the "best" redundancy profile?
            # For now, random.
            c_remove = random.choice(removable_centers)
            update_counts(c_remove, -1)
            code_set.remove(c_remove)
            current_len -= 1
            stuck_counter = 0 # Reset stuck counter on success
            
            if current_len < best_len:
                best_len = current_len
                best_code = set(code_set)
                best_counts = counts.copy()
            
            # Continue to try more removals
            continue
        
        # Phase 2: No removal possible. Try a swap.
        # Pick a random center to remove
        c_remove = random.choice(list(code_set))
        
        # Find valid swaps
        valid_swaps = get_valid_swaps(c_remove)
        
        if valid_swaps:
            # Pick a swap
            # Heuristic: Prefer swaps that increase "safety margin"?
            # For now, random.
            c_add = random.choice(valid_swaps)
            
            # Perform swap
            update_counts(c_remove, -1)
            code_set.remove(c_remove)
            update_counts(c_add, 1)
            code_set.add(c_add)
            
            stuck_counter += 1
            
            # If stuck for too long, try a "kick" (random perturbation) or break?
            # If we break, we might miss a better solution.
            # Let's limit stuck swaps to prevent infinite loops in a local minimum.
            # But a local minimum might require many swaps to escape.
            # Let's allow a larger budget.
            if stuck_counter > 5000:
                # Reset to best and try again? Or just break?
                # If we reset, we lose progress.
                # Let's just break if we've been swapping for a long time without reduction.
                # 5000 swaps is fast.
                break
        else:
            # No valid swap for this c_remove.
            # This means c_remove is "locked" by its critical words.
            # Try another center.
            # If we try all and none work, we are in a local minimum where no single swap helps.
            # This is a "2-opt" or larger move situation.
            # For simplicity, we break.
            # To be more robust, we could try a double-swap, but that's complex.
            # Let's break and return best.
            break

    # Convert best_code to list of lists
    result = [list(idx_to_word[idx]) for idx in best_code]
    
    return result
