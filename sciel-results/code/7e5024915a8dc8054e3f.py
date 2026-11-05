"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    """
    Construct a large subset of [1, 512] with no 4-term AP.
    
    Strategy:
    1. Precompute all 4-APs.
    2. Use a greedy construction that prioritizes elements based on a heuristic:
       - Start with a random valid set.
       - Or, start with a "3-AP-free" set (which is 4-AP-free) and try to add elements.
       - Actually, a better heuristic: Greedy by "degree" in the 4-AP hypergraph.
       - But random greedy is often strong. Let's use a "Weighted Greedy" where weight 
         is inversely proportional to the number of 4-APs the element participates in.
         Elements in fewer 4-APs are "safer" to add.
    3. Local Search: 
       - Identify "bad" elements: elements that are part of many "almost 4-APs" 
         (triples that would form a 4-AP if one more element were added).
       - Remove the worst element.
       - Try to add the best available element.
       - Repeat until no improvement.
    4. Multiple restarts with different random seeds/orderings.
    
    Key Insight for Speed:
    - Precompute for each element x, the list of 4-APs containing x.
    - Precompute for each pair (x, y) or triple? No, just use the conflict masks.
    - To find "bad" elements, we can count for each x in S, how many triples T in S 
      exist such that T U {x} is a 4-AP. This is expensive.
    - Alternative: Just use the standard "remove one, add best" local search.
    - To make it fast, we maintain a list of "candidate additions" and "candidate removals".
    
    Let's stick to a robust, fast local search:
    1. Generate initial set via Greedy (random order, or sorted by conflict count).
    2. Local Search Loop:
       a. Try to add any element not in S that is valid. If found, add it. Repeat until stuck.
       b. If stuck, remove a random element from S (or the one with highest "degree" in S).
       c. Go to (a).
       d. Keep track of best set found.
       e. Stop when time limit is reached.
    """
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 512
    
    # Precompute conflicts
    # conflicts[i] is a list of masks for element i+1
    conflicts = [[] for _ in range(N)]
    
    # Also, for each element, count total 4-APs it's in (for greedy heuristic)
    ap_count = [0] * N
    
    # Iterate all 4-APs
    for a in range(1, N - 2):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # Elements are a, b, c, e
            # For a: others are b, c, e
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            conflicts[a - 1].append(mask_a)
            ap_count[a-1] += 1
            
            # For b: others are a, c, e
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            conflicts[b - 1].append(mask_b)
            ap_count[b-1] += 1
            
            # For c: others are a, b, e
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
            conflicts[c - 1].append(mask_c)
            ap_count[c-1] += 1
            
            # For e: others are a, b, c
            mask_e = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflicts[e - 1].append(mask_e)
            ap_count[e-1] += 1

    def is_valid_addition(x, S):
        """Check if element x (1-indexed) can be added to bitmask S."""
        idx = x - 1
        for m in conflicts[idx]:
            if (S & m) == m:
                return False
        return True

    def greedy_construct(use_heuristic=False):
        """
        Greedy construction.
        If use_heuristic, pick elements with fewer APs first (safer).
        """
        S = 0
        if use_heuristic:
            # Sort by ap_count ascending (fewer conflicts first)
            order = list(range(1, N + 1))
            order.sort(key=lambda x: ap_count[x-1])
            # Add some randomness to break ties
            random.shuffle(order)
            # Stable sort by ap_count? No, just shuffle then sort? 
            # Better: bucket sort by ap_count, shuffle within buckets.
            buckets = defaultdict(list)
            for x in order:
                buckets[ap_count[x-1]].append(x)
            final_order = []
            for k in sorted(buckets.keys()):
                random.shuffle(buckets[k])
                final_order.extend(buckets[k])
            order = final_order
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)
            
        for x in order:
            if is_valid_addition(x, S):
                S |= (1 << (x - 1))
        return S

    def local_search(S, time_limit):
        """
        Improve S using local search.
        Returns best S found.
        """
        current_S = S
        current_size = bin(current_S).count('1')
        best_S = current_S
        best_size = current_size
        
        # List of elements not in S
        # We'll just iterate 1..N for additions
        
        # To speed up, we can maintain a list of "active" candidates?
        # For N=512, linear scan is fast enough if we do it smartly.
        
        while time.time() < time_limit:
            # Phase 1: Try to add any valid element
            improved = False
            # Randomize the order of checking to avoid bias
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                if not (current_S & (1 << (x - 1))):
                    if is_valid_addition(x, current_S):
                        current_S |= (1 << (x - 1))
                        current_size += 1
                        improved = True
                        # Restart candidate loop? Or continue?
                        # Continuing is fine, we might find more.
                        # But checking all is O(N*K).
                        # Let's just add the first one found and restart the phase?
                        # No, let's add all possible in this pass.
                        # But adding one might invalidate another? No, adding only creates new conflicts, 
                        # so if x was valid before, it's still valid unless it conflicts with a newly added y.
                        # But we check is_valid_addition against the *current* S, which updates.
                        # So this loop is correct.
                        pass
            
            if current_size > best_size:
                best_size = current_size
                best_S = current_S
            
            # Phase 2: If we didn't improve (or even if we did, but we are stuck at local opt),
            # we need to remove an element to allow new additions.
            # How to choose which element to remove?
            # Heuristic: Remove the element that is "most central" to the current structure?
            # Or just random?
            # Let's try: Remove a random element from S.
            
            # If S is empty, we are in trouble. But S starts non-empty.
            if current_size == 0:
                break
                
            # Pick a random element to remove
            # To make it smarter, we could count for each x in S, how many "almost 4-APs" it blocks.
            # But that's expensive.
            # Let's stick to random removal for now, but maybe bias towards elements with higher ap_count?
            # Elements with higher ap_count are "more connected", so removing them might open up more space.
            
            elements_in_S = [i+1 for i in range(N) if current_S & (1 << i)]
            if not elements_in_S:
                break
                
            # Bias: Higher ap_count -> higher chance of removal
            # Weights: 1 + ap_count[x-1]
            weights = [1 + ap_count[x-1] for x in elements_in_S]
            x_remove = random.choices(elements_in_S, weights=weights, k=1)[0]
            
            current_S &= ~(1 << (x_remove - 1))
            current_size -= 1
            
            # Check time
            if time.time() > time_limit:
                break

        return best_S, best_size

    # Main Loop: Multiple restarts
    best_overall_S = 0
    best_overall_size = 0
    
    # Run several restarts
    # Alternate between heuristic and random greedy
    num_restarts = 200 # We will limit by time
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Allocate time for this restart
        # If we have 80s left, and 200 restarts, that's 0.4s per restart.
        # Local search might be fast.
        
        # Choose strategy
        use_heuristic = (i % 2 == 0)
        
        S_init = greedy_construct(use_heuristic)
        
        # Run local search for a short burst
        # Time budget per restart: min(1.0, (deadline - time.time()) / remaining_restarts)
        # Let's just give each restart 0.5s max, or until time is up.
        restart_deadline = time.time() + 0.5
        
        S_best, size_best = local_search(S_init, restart_deadline)
        
        if size_best > best_overall_size:
            best_overall_size = size_best
            best_overall_S = S_best
            
    # Final polish on the best set found
    # Do one more pass of greedy addition
    S_final = best_overall_S
    for x in range(1, N + 1):
        if not (S_final & (1 << (x - 1))):
            if is_valid_addition(x, S_final):
                S_final |= (1 << (x - 1))
    
    # Convert to list
    result = []
    for i in range(N):
        if S_final & (1 << i):
            result.append(i + 1)
            
    # Verification
    def has_4ap(s):
        s_set = set(s)
        for a in s_set:
            for b in s_set:
                if a >= b:
                    continue
                d = b - a
                if d == 0:
                    continue
                c = b + d
                e = c + d
                if c in s_set and e in s_set:
                    return True
        return False
    
    if has_4ap(result):
        return [1]
        
    return sorted(result)

# EVOLVE-BLOCK-END
