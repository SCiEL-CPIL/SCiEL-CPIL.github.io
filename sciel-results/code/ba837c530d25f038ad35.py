import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s

    n = 16
    all_points = list(range(n))

    # Precompute all triples as masks
    all_triples = []
    triple_to_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                mask = (1 << i) | (1 << j) | (1 << k)
                idx = len(all_triples)
                all_triples.append(mask)
                triple_to_idx[mask] = idx

    num_triples = len(all_triples)  # 560

    # Precompute conflicts: for each triple, which other triples conflict?
    # Conflict if intersection size >= 2
    conflicts = [[] for _ in range(num_triples)]
    for i in range(num_triples):
        mi = all_triples[i]
        for j in range(i + 1, num_triples):
            mj = all_triples[j]
            inter = mi & mj
            if bin(inter).count('1') >= 2:
                conflicts[i].append(j)
                conflicts[j].append(i)

    # Precompute masks by size for alpha calculation
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(1 << n):
        k = bin(mask).count('1')
        masks_by_size[k].append(mask)

    def calc_alpha(triples_indices, max_check=n):
        """Calculate the independence number (largest set containing no triple).
        Starts checking from max_check downwards."""
        if not triples_indices:
            return n

        t_masks = [all_triples[i] for i in triples_indices]

        # Check sizes descending from max_check
        for k in range(min(max_check, n), 0, -1):
            for s in masks_by_size[k]:
                is_ind = True
                for t in t_masks:
                    if (s & t) == t:
                        is_ind = False
                        break
                if is_ind:
                    return k
        return 0

    def greedy_max_packing():
        """Greedy construction aiming for maximum number of triples (37)."""
        current_triples = set()
        # Randomize the order of candidates
        available = list(range(num_triples))
        random.shuffle(available)
        
        # To get a dense packing, we should prioritize triples that don't conflict
        # but also maybe cover "rare" points? 
        # Standard greedy: just add if valid.
        # To ensure maximality, we might need to be careful.
        
        for idx in available:
            if idx not in current_triples:
                can_add = True
                for c in conflicts[idx]:
                    if c in current_triples:
                        can_add = False
                        break
                if can_add:
                    current_triples.add(idx)
                    
        return list(current_triples)

    def local_search(initial_triples, max_time=10.0):
        """Local search to minimize alpha."""
        current_triples = set(initial_triples)
        current_alpha = calc_alpha(list(current_triples))
        best_triples = list(current_triples)
        best_alpha = current_alpha

        T = 2.0
        cooling_rate = 0.995
        iterations = 0
        max_iterations = 20000

        while time.time() - start_time < max_time and time.time() < deadline:
            new_triples = set(current_triples)
            move_type = random.random()

            if move_type < 0.5:
                # Reconfiguration: Pick a random triple not in the set, 
                # remove all conflicting triples, add the new one.
                # This allows "jumping" to a different part of the search space.
                candidates = [i for i in range(num_triples) if i not in current_triples]
                if candidates:
                    new_idx = random.choice(candidates)
                    # Find conflicts in current set
                    to_remove = []
                    for c in conflicts[new_idx]:
                        if c in new_triples:
                            to_remove.append(c)
                    
                    if len(to_remove) <= 2: # Only accept swaps that don't remove too many
                         for r in to_remove:
                             new_triples.remove(r)
                         new_triples.add(new_idx)
                    else:
                         # If too many conflicts, just try a simple swap or skip
                         # Let's try a simple swap: remove one random, add one random valid
                         if new_triples:
                             to_remove = random.choice(list(new_triples))
                             new_triples.remove(to_remove)
                             valid_adds = [i for i in range(num_triples) if i not in new_triples and all(c not in new_triples for c in conflicts[i])]
                             if valid_adds:
                                 new_triples.add(random.choice(valid_adds))

            elif move_type < 0.8:
                # Simple swap: remove one, add one valid
                if new_triples:
                    to_remove = random.choice(list(new_triples))
                    new_triples.remove(to_remove)
                    valid_adds = []
                    for idx in range(num_triples):
                        if idx not in new_triples:
                            if all(c not in new_triples for c in conflicts[idx]):
                                valid_adds.append(idx)
                    if valid_adds:
                        new_triples.add(random.choice(valid_adds))

            else:
                # Add if possible (rare in maximal packing)
                valid_adds = [i for i in range(num_triples) if i not in new_triples and all(c not in new_triples for c in conflicts[i])]
                if valid_adds:
                    new_triples.add(random.choice(valid_adds))

            if new_triples == current_triples:
                continue

            # Optimization: Only check alpha down from current_alpha
            # If we added a triple, alpha can decrease.
            # If we removed a triple, alpha can increase.
            # We want to minimize alpha.
            
            # Heuristic: If we removed more than we added, alpha likely increased.
            # If we added more than we removed, alpha likely decreased.
            
            # Just calculate new alpha, but start from current_alpha
            # If we suspect it increased, we might need to check higher?
            # But we track best_alpha.
            
            new_alpha = calc_alpha(list(new_triples), max_check=current_alpha)
            
            # If new_alpha is still current_alpha, it might have increased?
            # No, calc_alpha returns the LARGEST independent set.
            # If we removed triples, the largest independent set can only grow or stay same.
            # So if we removed triples, new_alpha >= current_alpha.
            # If we added triples, new_alpha <= current_alpha.
            
            # The calc_alpha above only checks down from current_alpha.
            # If we removed triples, we might miss a larger alpha.
            # But we are minimizing alpha. If it increased, it's worse (unless it allows a future move to drop it lower?).
            # Usually, we want to stay in low-alpha regions.
            
            # To be safe, if the move involved removal, we should check if alpha increased.
            # But checking up to n is expensive.
            # Let's assume we only accept moves that don't increase alpha, 
            # or we check a limited range.
            
            # For now, let's just use the calculated new_alpha.
            # If new_alpha > current_alpha, it's a bad move (usually).
            
            delta = new_alpha - current_alpha

            if delta <= 0 or (delta > 0 and random.random() < (0.1 ** delta)):
                current_triples = new_triples
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
                    T = max(T, 1.5)

            iterations += 1
            T *= cooling_rate
            if T < 0.05:
                T = 0.05

        return best_triples, best_alpha

    best_alpha = n
    best_triples = None

    # Phase 1: Multiple greedy starts to find a dense packing
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        triples = greedy_max_packing()
        alpha = calc_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            if alpha <= 6:
                break

    # Phase 2: Local search from best and some random starts
    if best_triples is not None:
        # Local search from the best greedy result
        ls_triples, ls_alpha = local_search(best_triples, max_time=30.0)
        if ls_alpha < best_alpha:
            best_alpha = ls_alpha
            best_triples = ls_triples

        # Additional local searches from different greedy starts
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            triples = greedy_max_packing()
            alpha = calc_alpha(triples)
            if alpha <= best_alpha + 1:
                ls_triples, ls_alpha = local_search(triples, max_time=5.0)
                if ls_alpha < best_alpha:
                    best_alpha = ls_alpha
                    best_triples = ls_triples

    # Phase 3: More aggressive local search if time remains
    if best_triples is not None and time.time() < deadline - 5:
        # Try to improve further with longer local search
        ls_triples, ls_alpha = local_search(best_triples, max_time=deadline - time.time() - 2)
        if ls_alpha < best_alpha:
            best_alpha = ls_alpha
            best_triples = ls_triples

    if best_triples is None:
        # Fallback: simple construction
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    # Convert indices back to points
    result = []
    for idx in best_triples:
        mask = all_triples[idx]
        pts = []
        for i in range(n):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)

    return result
