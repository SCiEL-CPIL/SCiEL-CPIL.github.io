# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 256
    max_size = 64
    rng = random.Random(42)
    
    best_total = 0
    best_A = []
    best_B = []
    
    def get_state(A, B):
        sums_A = set()
        diffs_A = set()
        for i in range(len(A)):
            ai = A[i]
            sums_A.add(ai + ai)
            for j in range(i + 1, len(A)):
                aj = A[j]
                sums_A.add(ai + aj)
                diffs_A.add(ai - aj)
                diffs_A.add(aj - ai)
                
        sums_B = set()
        diffs_B = set()
        for i in range(len(B)):
            bi = B[i]
            sums_B.add(bi + bi)
            for j in range(i + 1, len(B)):
                bj = B[j]
                sums_B.add(bi + bj)
                diffs_B.add(bi - bj)
                diffs_B.add(bj - bi)
        return sums_A, diffs_A, sums_B, diffs_B

    def can_add_A(x, A, sums_A, diffs_A, diffs_B):
        if x + x in sums_A:
            return False
        for a in A:
            if x + a in sums_A:
                return False
            d1 = x - a
            d2 = a - x
            if d1 in diffs_B or d2 in diffs_B:
                return False
        return True

    def can_add_B(x, A, sums_A, diffs_A, diffs_B, B, sums_B):
        if x + x in sums_B:
            return False
        for b in B:
            if x + b in sums_B:
                return False
            d1 = x - b
            d2 = b - x
            if d1 in diffs_A or d2 in diffs_A:
                return False
        return True

    def greedy_simultaneous(order_A, order_B):
        A = []
        B = []
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()
        
        # Interleave based on random priority or just alternate?
        # Let's try a "best fit" greedy: at each step, pick the element that allows the most future additions?
        # Too slow. Just use randomized order and try to add to the set that has more space or is "easier".
        
        # We will iterate through a merged list of (element, target_set) pairs?
        # Or simply:
        # 1. Try to build A and B independently in a round-robin fashion.
        
        # Let's use a simple greedy that tries to add to A if possible, else B, in a random order of elements.
        # But we have separate orders. Let's merge them into a single stream of decisions.
        
        # Actually, the previous approach of building A then B is biased.
        # Let's try: For each element in 1..N (randomized), decide if it goes to A, B, or neither.
        # This is hard because adding to A blocks B.
        
        # Alternative: Greedy on the union of candidates.
        # We will maintain a set of "available" candidates for A and B.
        
        # Let's stick to the "Greedy with Randomized Order" but alternating.
        # At each step, pick a random element from 1..N not yet "processed" in this pass?
        # No, elements can be in both.
        
        # Let's use the "Conflict Graph" inspiration more directly.
        # We want to select a set of pairs (x, 'A') and (x, 'B').
        # We will use a greedy algorithm that picks the "least conflicting" node.
        
        # For simplicity and speed, we will use a heuristic:
        # 1. Randomly shuffle all elements.
        # 2. For each element x, try to add it to A. If successful, do it.
        # 3. Then for each element x, try to add it to B. If successful, do it.
        # 4. This is biased towards A.
        
        # To be symmetric, we can randomize which set we try to fill first for each element?
        # Or better:
        # Run multiple rounds. In each round, try to add to A, then B.
        # But "Local Search" will fix asymmetries.
        
        # Let's just use the existing `greedy_construct` but with more randomization in the local search.
        
        A = []
        B = []
        
        # We'll use a priority queue or just a list.
        # Let's try to add elements in a random order, trying A first, then B.
        # If we can't add to A, we don't force it to B immediately, we leave it for B's turn.
        
        # Actually, the most effective simple strategy is:
        # 1. Generate random permutation P.
        # 2. For x in P:
        #    If can_add_A(x): add to A
        #    Else if can_add_B(x): add to B
        # This is biased to A.
        
        # To fix bias, we can do:
        # 1. Generate random permutation P.
        # 2. For x in P:
        #    If can_add_A(x) and can_add_B(x):
        #        If len(A) < len(B): add to A
        #        Else: add to B
        #    Elif can_add_A(x): add to A
        #    Elif can_add_B(x): add to B
        
        # This balances the sizes.
        
        for x in order_A: # Reusing order_A as the master random order
            if len(A) < max_size and can_add_A(x, A, sums_A, diffs_A, diffs_B):
                # Add to A
                A.append(x)
                sums_A.add(x + x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
            elif len(B) < max_size and can_add_B(x, A, sums_A, diffs_A, diffs_B, B, sums_B):
                # Add to B
                B.append(x)
                sums_B.add(x + x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
            else:
                # Try the other set if the first was full or failed?
                # The logic above: if A fails, try B.
                # But if A succeeds, we don't try B.
                # This is fine.
                pass
        
        return A, B

    def local_search_swaps(A, B, max_iters=1000):
        A = list(A)
        B = list(B)
        
        for _ in range(max_iters):
            improved = False
            sums_A, diffs_A, sums_B, diffs_B = get_state(A, B)
            
            # 1. Try additions
            if len(A) < max_size:
                for x in range(1, n + 1):
                    if x not in A and can_add_A(x, A, sums_A, diffs_A, diffs_B):
                        A.append(x)
                        improved = True
                        break
            if improved:
                continue # Restart loop to update state

            if len(B) < max_size:
                sums_A, diffs_A, sums_B, diffs_B = get_state(A, B)
                for x in range(1, n + 1):
                    if x not in B and can_add_B(x, A, sums_A, diffs_A, diffs_B, B, sums_B):
                        B.append(x)
                        improved = True
                        break
            if improved:
                continue

            # 2. Try Swaps
            # Remove an element from A, see if we can add 2 to B or 1 to A (different one)
            # Or simply: Remove a, try to add x to A. If x != a and valid, swap.
            # This doesn't increase size. We need to increase size.
            # So: Remove a from A. Check if we can add x to B (that was blocked).
            # Or: Remove a from A. Check if we can add x to A (different) AND y to B?
            
            # Simplest Swap that increases size:
            # Remove a from A.
            # If we can now add x to B (where x was not in B and couldn't be added before), 
            # and we can also add y to A (where y was not in A and couldn't be added before)?
            # That's complex.
            
            # Let's try: Remove a from A.
            # Try to add any x to A. If successful, size is same.
            # Try to add any x to B. If successful, size is same (A lost 1, B gained 1).
            # We want size to INCREASE.
            # So, Remove a from A.
            # If we can add TWO elements to B? Unlikely in one step.
            # If we can add ONE element to B, and ONE element to A?
            
            # Alternative: 
            # If we are stuck, maybe we should swap an element in A with an element in B?
            # No, they are independent sets.
            
            # Let's try "Kick-out":
            # For each a in A:
            #   Temporarily remove a.
            #   Try to add any x to A. If added, size same.
            #   Try to add any x to B. If added, size same.
            #   Try to add x to A and y to B?
            
            # Given the time limit, simple additions might be enough if we have many restarts.
            # But "Swaps" in the directive usually implies 1-for-1 to allow 2-for-0 later?
            # Or 1-for-1 that opens up space for a 2nd addition?
            
            # Let's implement a simple 1-for-1 swap that might lead to improvement in the next iteration.
            # Actually, if we are at a local maximum for additions, any 1-for-1 swap keeps the size same.
            # It only helps if the new configuration allows a 2nd addition in the next step.
            
            # Let's try:
            # For each a in A:
            #   Remove a.
            #   Find x that can be added to A.
            #   If found, swap a->x.
            #   This is just a shuffle.
            
            # Better:
            # For each a in A:
            #   Remove a.
            #   Try to add x to B.
            #   If added, we have A-1, B+1. Total size same.
            #   But maybe now we can add y to A?
            #   If we can add y to A, then we have A, B+1. Total size +1.
            
            # So:
            # 1. Remove a from A.
            # 2. If can_add_B(x) for some x not in B:
            #      Add x to B.
            #      If can_add_A(y) for some y not in A (and y != x if sets are disjoint? No, overlap allowed):
            #          Add y to A.
            #          Size increased by 1. Keep this state.
            #      Else:
            #          Keep the state (A-1, B+1) for a few iterations? 
            #          Or revert if no immediate gain?
            #          Reverting is safer to avoid oscillation.
            
            # Let's try to find a sequence: Remove a, Add x to B, Add y to A.
            
            for i, a in enumerate(A):
                # Try removing a
                A_temp = A[:i] + A[i+1:]
                sums_A_t, diffs_A_t, sums_B, diffs_B = get_state(A_temp, B)
                
                # Try to add x to B
                added_to_B = False
                x_added = -1
                for x in range(1, n + 1):
                    if x not in B and can_add_B(x, A_temp, sums_A_t, diffs_A_t, diffs_B, B, sums_B):
                        B_temp = B + [x]
                        x_added = x
                        added_to_B = True
                        break
                        
                if added_to_B:
                    # Now try to add y to A_temp
                    sums_A_t2, diffs_A_t2, sums_B_t2, diffs_B_t2 = get_state(A_temp, B_temp)
                    for y in range(1, n + 1):
                        if y not in A_temp and can_add_A(y, A_temp, sums_A_t2, diffs_A_t2, diffs_B_t2):
                            # Success: A_temp + [y], B_temp
                            A = A_temp + [y]
                            B = B_temp
                            improved = True
                            break
                            
                if improved:
                    break
                    
            if improved:
                continue
                
            # If no improvement, break
            break
            
        return A, B

    for seed in range(100):
        rng.seed(seed)
        
        order_A = list(range(1, n + 1))
        rng.shuffle(order_A)
        
        A, B = greedy_simultaneous(order_A, order_A)
        
        # Alternate greedy strategies
        # Sometimes building B first is better
        if seed % 2 == 0:
             A, B = greedy_simultaneous(order_A, order_A)
        else:
             # Swap roles in greedy?
             # Just run again with different seed logic inside greedy?
             # For now, just run local search.
             pass
             
        A, B = local_search_swaps(A, B, max_iters=50)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
            
        if best_total >= 120: # Heuristic threshold
            break

    # Final validation and fallback
    if not best_A or not best_B:
         best_A = [1, 2, 4, 8, 16, 32, 64, 128]
         best_B = [3, 5, 9, 17, 33, 65, 129]
         
    return [best_A, best_B]
# EVOLVE-BLOCK-END
