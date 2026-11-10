import time
import random

def construct():
    N = 1000
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    best_set = []
    best_size = 0
    
    # Helper: Check if x can be added to A
    def can_add(A, counts, x):
        # A is list, counts is dict
        # Check sums x+y for y in A where y <= x, and 2x
        # Note: We only need to check if counts[s] + 1 > 2
        # i.e., counts[s] >= 2
        
        # Optimization: Only check y in A such that y <= x
        # But A is not sorted. We can just iterate all y.
        for y in A:
            s = x + y
            if s > 2 * N: continue # sums are bounded, but counts dict handles it
            if counts.get(s, 0) >= 2:
                return False
        s = 2 * x
        if counts.get(s, 0) >= 2:
            return False
        return True

    def update_counts(A, counts, x, sign):
        # sign = 1 for add, -1 for remove
        for y in A:
            s = x + y
            counts[s] = counts.get(s, 0) + sign
            if counts[s] == 0:
                del counts[s]
        s = 2 * x
        counts[s] = counts.get(s, 0) + sign
        if counts[s] == 0:
            del counts[s]

    def greedy_random(seed):
        random.seed(seed)
        A = []
        counts = {}
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(A, counts, x):
                A.append(x)
                update_counts(A, counts, x, 1)
        return A, counts

    # Phase 1: Multiple Greedy Restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline: break
        A, counts = greedy_random(i)
        if len(A) > best_size:
            best_size = len(A)
            best_set = A
            best_counts = counts

    # Phase 2: Local Search on Best Set
    A = best_set[:]
    counts = best_counts.copy()
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try Insertions
        # Shuffle complement to randomize
        comp = [x for x in range(1, N+1) if x not in A]
        random.shuffle(comp)
        
        for x in comp:
            if can_add(A, counts, x):
                A.append(x)
                update_counts(A, counts, x, 1)
                improved = True
                break
        
        if improved:
            if len(A) > best_size:
                best_size = len(A)
                best_set = A[:]
                best_counts = counts.copy()
            continue # Restart insertion loop
        
        # 2. Try Swaps
        # Find all valid swaps
        valid_swaps = []
        # Optimization: We can break early if we find a good one?
        # Let's just find one valid swap to save time.
        # Random order of A and comp
        
        A_list = A[:]
        random.shuffle(A_list)
        
        found_swap = False
        for a in A_list:
            if found_swap: break
            # Try random b from comp
            # To be efficient, just pick a few random b's for each a?
            # Or iterate all b?
            # Iterating all b is 920 checks. 80 * 920 = 73600 checks.
            # Each check is O(|A|) = 80. Total 6M ops.
            # This is acceptable.
            
            comp_list = comp[:]
            random.shuffle(comp_list)
            
            for b in comp_list:
                # Check if (A - {a}) U {b} is valid
                # 1. Remove a
                update_counts(A, counts, a, -1)
                
                # 2. Check if b can be added to (A - {a})
                # Note: A still contains a in the list, but counts are updated.
                # can_add uses A list. We must be careful.
                # Let's create a temporary list or handle it.
                # Simpler: 
                # Remove a from A list temporarily? 
                # Or modify can_add to take a set?
                
                # Let's just use a helper that checks validity of a set given counts.
                # Actually, can_add(A, counts, x) assumes x is NOT in A.
                # Here b is not in A.
                # But A still has a.
                # The sums involving a are removed from counts.
                # The sums involving b are NOT in counts yet.
                # can_add checks if counts[s] >= 2 for new sums.
                # This is correct because counts reflects A \ {a}.
                
                if can_add(A, counts, b):
                    # Valid swap!
                    # Commit: Remove a from A list, Add b to A list
                    A.remove(a)
                    A.append(b)
                    # Update counts for b
                    update_counts(A, counts, b, 1)
                    # Note: a was already removed from counts.
                    found_swap = True
                    break
                else:
                    # Revert counts for a
                    update_counts(A, counts, a, 1)

        if found_swap:
            # We swapped. Now loop back to try insertions.
            # To avoid immediate undo, we could record last swap, but randomization helps.
            continue
        else:
            # No valid swaps found. Stuck.
            # Try a different random seed for the next restart?
            # Or just stop?
            # Let's try one more greedy restart with a new seed.
            seed = int(time.time() * 1000) % 10000
            A_new, counts_new = greedy_random(seed)
            if len(A_new) > len(A):
                A = A_new
                counts = counts_new
                best_size = len(A)
                best_set = A[:]
                best_counts = counts.copy()
            else:
                # If we are stuck and no improvement, break to save time for other strategies?
                # Or just break.
                break

    return best_set
