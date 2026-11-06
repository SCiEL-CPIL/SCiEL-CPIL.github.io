import time
import random
import sys

def construct():
    start_time = time.time()
    # Safety margin: 90s limit, use 80s for computation
    deadline = start_time + 80.0
    
    N_MAX = 4096
    candidates = list(range(1, N_MAX + 1))
    
    def time_up():
        return time.time() > deadline

    def can_add(S, x, sums_set):
        """
        Check if x can be added to S without creating duplicate triple sums.
        Returns (valid, new_sums_to_add) or (False, None).
        Uses set intersection for speed.
        """
        n = len(S)
        new_sums = set()
        
        # (x, x, x)
        new_sums.add(3 * x)
        
        # (x, x, s) for s in S
        for s in S:
            new_sums.add(2 * x + s)
            
        # (x, s, t) for s <= t in S
        # We iterate i, j to generate unique pairs (s,t)
        for i in range(n):
            si = S[i]
            for j in range(i, n):
                tj = S[j]
                new_sums.add(x + si + tj)
                
        # Check for collision with existing sums
        if new_sums & sums_set:
            return False, None
            
        return True, new_sums

    def build_sums(S):
        """Rebuild the set of all triple sums for a given list S."""
        sums = set()
        n = len(S)
        for i in range(n):
            si = S[i]
            for j in range(i, n):
                sj = S[j]
                for k in range(j, n):
                    sk = S[k]
                    sums.add(si + sj + sk)
        return sums

    def greedy_fill(S, order, sums_set=None):
        """
        Start with set S (list) and try to add elements from 'order' 
        that are not already in S, maintaining validity.
        Returns the final list S and its sums_set.
        """
        if sums_set is None:
            sums_set = build_sums(S)
            
        S = list(S) # Work on a copy
        S_set = set(S)
        
        for x in order:
            if time_up():
                break
            if x in S_set:
                continue
            valid, new_sums = can_add(S, x, sums_set)
            if valid:
                S.append(x)
                S_set.add(x)
                sums_set.update(new_sums)
                
        return S, sums_set

    def local_search_refill(S, order, sums_set=None):
        """
        Given a valid set S, try to add any element from 'order' 
        that fits. This is essentially greedy_fill but usually 
        used on a set that is already "full" in one ordering 
        to see if another ordering allows more.
        """
        return greedy_fill(S, order, sums_set)

    best_set = []
    best_len = 0
    
    # Phase 1: Basic Greedy Strategies
    # 1. Increasing
    S_inc, sums_inc = greedy_fill([], candidates, None)
    if len(S_inc) > best_len:
        best_set = S_inc
        best_len = len(S_inc)
        
    # 2. Decreasing
    S_dec, sums_dec = greedy_fill([], candidates[::-1], None)
    if len(S_dec) > best_len:
        best_set = S_dec
        best_len = len(S_dec)

    # 3. Randomized Greedy
    random.seed(42)
    for _ in range(20):
        if time_up(): break
        order = candidates[:]
        random.shuffle(order)
        S_rand, sums_rand = greedy_fill([], order, None)
        if len(S_rand) > best_len:
            best_set = S_rand
            best_len = len(S_rand)
            
    # Phase 2: Perturbation + Refill
    # Idea: Take the best set, remove k random elements, and refill.
    # This allows "reshuffling" the set to escape local optima.
    
    max_perturbations = 100
    for _ in range(max_perturbations):
        if time_up(): break
        
        # Start from the current best
        current_S = best_set[:]
        if not current_S:
            continue
            
        # Remove 1 to 3 random elements
        k = random.randint(1, 3)
        if len(current_S) <= k:
            continue
        to_remove = random.sample(current_S, k)
        remove_set = set(to_remove)
        current_S = [x for x in current_S if x not in remove_set]
        
        # Rebuild sums for the reduced set
        sums_current = build_sums(current_S)
        
        # Refill using a random order
        order = candidates[:]
        random.shuffle(order)
        
        # Try to add elements
        new_S, new_sums = greedy_fill(current_S, order, sums_current)
        
        if len(new_S) > best_len:
            best_set = new_S
            best_len = len(best_set)
            # If we improved, we might want to try more perturbations from this new best
            # Or just continue loop which restarts from best_set
            
    # Phase 3: Final Polish
    # Try to add any remaining valid elements to the best set
    # We do this by iterating through all candidates and adding if valid
    # This is just a final greedy_fill with a specific order (e.g., increasing)
    # to catch any "easy" additions.
    final_order = candidates[:]
    # Actually, the order matters. Let's try increasing, then decreasing.
    if best_set:
        sums_best = build_sums(best_set)
        # Try increasing
        S_polished, _ = greedy_fill(best_set, candidates, sums_best)
        if len(S_polished) > best_len:
            best_set = S_polished
            best_len = len(best_set)
            
        # Try decreasing
        sums_best = build_sums(best_set)
        S_polished, _ = greedy_fill(best_set, candidates[::-1], sums_best)
        if len(S_polished) > best_len:
            best_set = S_polished
            best_len = len(best_set)
            
    # Final Verification
    if best_set:
        final_sums = build_sums(best_set)
        if len(final_sums) != len(set(final_sums)): # This check is redundant as build_sums returns a set
             pass
        # Check uniqueness explicitly just in case of logic errors
        # n = len(best_set)
        # count = 0
        # for i in range(n):
        #   for j in range(i, n):
        #     for k in range(j, n):
        #       count += 1
        # if len(final_sums) != count:
        #   # Collision found, should not happen with correct can_add
        #   pass
        if len(best_set) > 0:
            return best_set
    else:
        # Fallback: Powers of 2
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# EVOLVE-BLOCK-END
