import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    N = 32
    MAX_VAL = 512
    bit_masks = [1 << i for i in range(MAX_VAL + 1)]

    def score_mask(mask):
        count = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def mask_to_list(mask):
        return [x for x in range(1, MAX_VAL + 1) if mask & bit_masks[x]]

    # 1. Generate all possible triplets (a, d) such that a+2d <= MAX_VAL
    # Each triplet is (a, d). It covers difference d.
    # We want to select a set of triplets such that the union of their elements has size <= 32
    # and maximizes the number of distinct d's.
    # Actually, we can just greedily build the set.
    
    # A better structural approach:
    # The score is the number of distinct d.
    # If we include a triplet (a, a+d, a+2d), we gain d.
    # We want to maximize the count of distinct d.
    # Let's try a greedy algorithm that adds elements one by one, but prioritizes
    # creating new differences.
    
    # Alternative: Start with a known good structure or use a specific heuristic.
    # The inspiration mentioned "Cantor-like base-4 digit constructions".
    # Let's try a deterministic construction based on binary representations or similar.
    # However, the prompt asks for a strategy materially different from GA.
    # A "Greedy Triplet Selection" is distinct.
    
    # Let's refine the Greedy Triplet Selection:
    # 1. Maintain current set S (bitmask).
    # 2. Maintain set of covered differences D.
    # 3. At each step, consider adding a new element x to S.
    #    Calculate new differences formed by x with existing elements.
    #    Prioritize x that creates the most NEW differences.
    #    If tie, prioritize x that is part of a triplet with two existing elements (completing a triplet).
    
    def greedy_triplet_search():
        S_mask = 0
        S_set = set()
        covered_d = set()
        
        for _ in range(N):
            best_x = -1
            best_new_d = -1
            best_completions = -1
            
            for x in range(1, MAX_VAL + 1):
                if x in S_set:
                    continue
                
                # Check differences formed by x
                new_d = set()
                completions = 0
                
                for y in S_set:
                    d = abs(x - y)
                    if d > 0:
                        # Check if x, y, and another element form a triplet
                        # Case 1: y is middle. Need y - d and y + d.
                        # Case 2: y is start. Need y + d and y + 2d. (x is y+2d or y+d?)
                        # Let's just check all triplets involving x.
                        
                        # Triplet (x-d, x, x+d) -> d
                        if (x - d) in S_set and (x + d) in S_set:
                            # This forms a triplet with difference d.
                            # Wait, if x-d and x+d are in S, then (x-d, x, x+d) is a triplet.
                            # The difference is d.
                            if d not in covered_d:
                                new_d.add(d)
                            completions += 1
                            
                        # Triplet (x, x+d, x+2d) -> d
                        # This is covered if we check pairs.
                        # Let's iterate over pairs (y, z) in S such that x completes a triplet?
                        # No, x is new.
                        # Triplets involving x:
                        # 1. x, x+d, x+2d (x is start)
                        # 2. x-d, x, x+d (x is middle)
                        # 3. x-2d, x-d, x (x is end)
                        
                        # It's easier to just check if adding x creates any new d.
                        # A new d is created if there exists a, b in S (or x) such that 
                        # {a, a+d, a+2d} is a subset of S U {x}.
                        
                        # Let's just calculate the score difference.
                        # But that's expensive.
                        
                        # Heuristic: Count how many existing elements y are such that
                        # adding x completes a triplet with y.
                        # If y and x are distance d apart:
                        # - If y-d in S: triplet (y-d, y, x) or (x, y, y+d)? No.
                        #   If y-d in S, then (y-d, y, x) is a triplet if x = y+d.
                        #   Then d is the difference.
                        # - If y+d in S: triplet (y, x, y+d) is NOT a 3-term AP with diff d unless x-y = y+d-x => 2x = 2y+d.
                        #   Actually, if y, x, z form AP, then x-y = z-x.
                        
                        # Let's stick to the simple score-based greedy but optimized.
                        # The previous GA used score_mask.
                        # Let's try a "Greedy with Lookahead" or "Greedy with Random Restart".
                        
                        # Since GA stagnated, maybe a simple Greedy with multiple random restarts is better/faster.
                        # Or, a specific constructive algorithm.
                        
                        # Let's try: Greedy that adds the element maximizing the number of NEW differences.
                        # To find new differences for x:
                        # For each d in 1..255:
                        #   Does S U {x} contain a, a+d, a+2d?
                        #   If S contains a, a+d, a+2d, d is already covered.
                        #   If S contains a, a+d but not a+2d, and x == a+2d, then d is NEW.
                        #   If S contains a, a+2d but not a+d, and x == a+d, then d is NEW.
                        #   If S contains a+d, a+2d but not a, and x == a, then d is NEW.
                        
                        # This is O(|S| * 256) per x. |S| <= 32. 32*256 = 8192 ops per x.
                        # 512 x's. 512 * 8192 = 4M ops per step.
                        # 32 steps. 128M ops. Too slow for Python in 110s? Maybe.
                        # We can optimize by precomputing potential differences.
                        
                        pass # Placeholder for complex logic, let's use a simpler heuristic first.
                        
                # Simpler Heuristic:
                # Score = number of y in S such that (x-y) is a "good" difference?
                # No, let's just use the bitmask score but only for the top candidates?
                
                # Let's try a different structural idea:
                # **Base-4 / Cantor set construction**
                # The inspiration mentioned this.
                # A set defined by digits in base 4 being only 0 or 1 (Cantor set) has no 3-term APs?
                # No, we WANT 3-term APs.
                # The complement of a Cantor set?
                # Or a set with many small differences.
                
                # Let's try a **Greedy Algorithm that explicitly targets differences**.
                # We want to cover d=1, 2, 3...
                # To cover d, we need a, a+d, a+2d.
                # Greedy:
                # 1. Start with empty set.
                # 2. For d from 1 to 255:
                #    If d is not covered:
                #       Find a such that adding a, a+d, a+2d (or subsets) minimizes new elements added.
                #       Prefer a where a, a+d, a+2d are already in S or can be added cheaply.
                
                # This is a Set Cover problem approximation.
                
                pass

        return mask_to_list(S_mask)
    
    # Let's implement the "Set Cover" style greedy.
    # State: S (set of elements), Covered (set of d)
    # Goal: Maximize len(Covered)
    
    def set_cover_greedy():
        S = set()
        Covered = set()
        
        # Precompute all possible triplets (a, d)
        # We will iterate through d's in increasing order of "difficulty" or just 1..255.
        # Actually, small d's are easier to cover with dense sets.
        
        # To make it fast, we'll just do a standard greedy:
        # At each step, pick the element x that maximizes the number of NEW d's it enables.
        # "Enables" means: for any d, if S U {x} has a triplet with diff d, but S did not, it's a new d.
        
        for _ in range(N):
            best_x = -1
            best_gain = -1
            
            for x in range(1, MAX_VAL + 1):
                if x in S:
                    continue
                
                gain = 0
                # Check all d
                for d in range(1, 256):
                    if d in Covered:
                        continue
                    
                    # Check if adding x completes a triplet for d
                    # Triplet forms: (a, a+d, a+2d)
                    # x must be one of these.
                    # 1. x = a. Need a+d, a+2d in S.
                    # 2. x = a+d. Need a, a+2d in S.
                    # 3. x = a+2d. Need a, a+d in S.
                    
                    # Case 1: x = a
                    if (x + d) in S and (x + 2*d) in S:
                        gain += 1
                        # Once we know it can cover d, we don't need to check other a's for this d?
                        # No, we just need to know if d becomes covered.
                        # If any triplet for d is formed, d is covered.
                        break # Found one, d is covered.
                    
                    # Case 2: x = a+d => a = x-d, a+2d = x+d
                    if (x - d) > 0 and (x - d) in S and (x + d) in S:
                        gain += 1
                        break
                        
                    # Case 3: x = a+2d => a = x-2d, a+d = x-d
                    if (x - 2*d) > 0 and (x - 2*d) in S and (x - d) in S:
                        gain += 1
                        break
                        
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
                    if best_gain == 0:
                         # If no gain, break early? No, we must add an element.
                         # We'll just pick the first one or random if all 0.
                         # But we continue to find if any has gain > 0.
                         pass
            
            if best_x != -1:
                S.add(best_x)
                # Update Covered
                # We can re-calculate or just add the gained ones.
                # For simplicity and correctness, let's just re-calculate Covered from S if S is small?
                # No, that's O(32^2 * 256) = 262k. Fine.
                # But we can just add the d's that were gained.
                # Actually, we didn't track WHICH d's were gained, just the count.
                # Let's just re-calculate Covered set.
                new_covered = set()
                for d in range(1, 256):
                    # Check if S has a triplet for d
                    # a in S, a+d in S, a+2d in S
                    for a in S:
                        if (a + d) in S and (a + 2*d) in S:
                            new_covered.add(d)
                            break
                Covered = new_covered
            else:
                # No element improves score? Add random.
                candidates = [x for x in range(1, MAX_VAL + 1) if x not in S]
                if candidates:
                    S.add(random.choice(candidates))
                    # Recalculate Covered
                    new_covered = set()
                    for d in range(1, 256):
                        for a in S:
                            if (a + d) in S and (a + 2*d) in S:
                                new_covered.add(d)
                                break
                    Covered = new_covered

        S_mask = 0
        for x in S:
            S_mask |= bit_masks[x]
        return S_mask

    # Run multiple restarts of the greedy algorithm
    best_mask = 0
    best_score = -1
    
    # One deterministic run
    m = set_cover_greedy()
    s = score_mask(m)
    if s > best_score:
        best_score = s
        best_mask = m
        
    # Randomized restarts
    # We can randomize the tie-breaking or the order of x consideration.
    # Or just shuffle the candidate list.
    
    for i in range(50):
        if time.time() > deadline:
            break
        # Randomize the greedy by shuffling the order we check x?
        # Or just run a random local search on the result?
        
        # Let's try a simple Local Search on the greedy result
        current_mask = best_mask
        current_score = best_score
        current_set = mask_to_list(current_mask)
        current_set_set = set(current_set)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try swapping one element for another
            for i_idx, x_out in enumerate(current_set):
                for x_in in range(1, MAX_VAL + 1):
                    if x_in in current_set_set:
                        continue
                    new_set = current_set.copy()
                    new_set[i_idx] = x_in
                    new_mask = 0
                    for x in new_set:
                        new_mask |= bit_masks[x]
                    new_score = score_mask(new_mask)
                    if new_score > current_score:
                        current_mask = new_mask
                        current_score = new_score
                        current_set = new_set
                        current_set_set = set(new_set)
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                        break # Restart loop after improvement? Or continue?
                # if improved: break # To speed up, break out of inner loop and restart outer?
                # Actually, just breaking inner and continuing outer with updated state is fine.
                # But we need to update current_set_set.
                # Let's just break both loops and restart the while loop for a cleaner "First Improvement" strategy.
                if improved:
                    break
            
    return mask_to_list(best_mask)
