import sys
import time

def construct():
    n = 192
    candidates = list(range(1, n + 1))
    
    # Precomputation for fast validity check
    # For each x, we need to check if there exist a,b,c in S such that x | a+b+c.
    # And for each y in S, if y | x+a+b for some a,b in S\{y}.
    
    # We will use a list-based approach for S in the check, but optimize the inner loop.
    
    def is_valid_add(x, S_list):
        # S_list is a list of selected integers
        K = len(S_list)
        
        # Check 1: x divides sum of any 3 in S_list
        if K >= 3:
            # Use set of elements for O(1) lookup? 
            # We need to check if c = k*x - a - b is in S_list.
            # Iterate a, b in S_list
            # a, b distinct. c distinct from a, b.
            S_set = set(S_list)
            # To avoid O(K^3), we do O(K^2)
            # For each a, b:
            #   target_sum_mod = (-a-b) % x
            #   We need c in S_list such that c % x == target_sum_mod AND c != a AND c != b.
            #   Precompute for each x, a list of elements in S_list grouped by mod x?
            #   This depends on S_list, so we can't precompute globally.
            #   But K is small.
            #   Let's just do the O(K^2) check.
            
            # Optimization: If x > sum(S_list), x cannot divide sum of 3 (unless sum is 0, impossible).
            # Max sum of 3 is 3*192 = 576.
            # If x > 576, always valid for Check 1.
            if x <= 576:
                # Group S_list by value? No, just iterate.
                # To speed up, we can break early.
                for i in range(K):
                    a = S_list[i]
                    for j in range(i+1, K):
                        b = S_list[j]
                        # We need c in S_list, c != a, c != b
                        # s = a+b
                        # We need c such that (s+c) % x == 0
                        # c % x == (-s) % x
                        rem = (- (a+b)) % x
                        # Check if any c in S_list has c % x == rem and c != a, c != b
                        # This is still O(K) inside O(K^2) -> O(K^3).
                        # Can we do better?
                        # Precompute counts of residues for S_list for modulus x?
                        # Since x changes, we can't precompute globally.
                        # But we can build a dict for current S_list once per call?
                        # O(K) to build dict. Then O(K^2) to check.
                        # Total O(K^2).
                        
                        pass # Placeholder for logic below
        
        # Let's implement the O(K^2) check properly.
        
        # Check 1
        if K >= 3 and x <= 576:
            # Build residue counts for S_list mod x
            res_counts = {}
            for v in S_list:
                r = v % x
                res_counts[r] = res_counts.get(r, 0) + 1
            
            for i in range(K):
                a = S_list[i]
                for j in range(i+1, K):
                    b = S_list[j]
                    rem = (- (a+b)) % x
                    count = res_counts.get(rem, 0)
                    # Subtract a and b if they have this residue
                    if a % x == rem:
                        count -= 1
                    if b % x == rem:
                        count -= 1
                    if count > 0:
                        return False
        
        # Check 2: Any y in S_list divides sum of x and 2 others
        for y in S_list:
            if y > 192 + 192 + 192: # y > 576, always valid
                continue
            # We need a, b in S_list \ {y} such that (x + a + b) % y == 0
            # a, b distinct, a,b != y
            others = [v for v in S_list if v != y]
            if len(others) < 2:
                continue
            
            # Build residue counts for others mod y
            res_counts_y = {}
            for v in others:
                r = v % y
                res_counts_y[r] = res_counts_y.get(r, 0) + 1
            
            for i in range(len(others)):
                a = others[i]
                for j in range(i+1, len(others)):
                    b = others[j]
                    rem = (- (x + a + b)) % y
                    # Wait, the sum is x + a + b.
                    # We need (x + a + b) % y == 0.
                    # This is fixed for a given a,b.
                    # So we just check if (x + a + b) % y == 0.
                    # No need for residue count for the *target*, just check the value.
                    # But we need to ensure a,b are distinct and not y.
                    # The loop above ensures a,b from others (so != y) and i<j (distinct).
                    if (x + a + b) % y == 0:
                        return False
                        
        return True

    def greedy(order):
        S = []
        for x in order:
            if is_valid_add(x, S):
                S.append(x)
        return S

    # Initial Greedy
    best_set = []
    orders = [
        list(range(192, 0, -1)),
        list(range(1, 193)),
    ]
    for order in orders:
        S = greedy(order)
        if len(S) > len(best_set):
            best_set = S
    
    best_size = len(best_set)
    
    # B&B
    # Use the best_set as initial lower bound.
    # Search for larger set.
    
    search_order = sorted(range(1, 193), reverse=True)
    
    global_best_size = best_size
    global_best_set = best_set[:]
    
    start_time = time.time()
    time_limit = 50
    
    def bnb(S, start_idx):
        nonlocal global_best_size, global_best_set
        
        if time.time() - start_time > time_limit:
            return
            
        if len(S) + (192 - start_idx + 1) <= global_best_size:
            return
            
        if len(S) >= 48:
            return

        for i in range(start_idx, 193):
            if time.time() - start_time > time_limit:
                return
            x = i
            # Check if x is already in S? 
            # We are iterating i from start_idx, and S contains elements < i?
            # Not necessarily. We should maintain S as a list and check membership.
            # Better: Iterate through search_order.
            
            # Let's refactor to use search_order index.
            pass

    # Refactored B&B
    def bnb_refactored(S, start_idx):
        nonlocal global_best_size, global_best_set
        
        if time.time() - start_time > time_limit:
            return
            
        # Pruning
        if len(S) + (len(search_order) - start_idx) <= global_best_size:
            return
            
        if len(S) >= 48:
            return

        for i in range(start_idx, len(search_order)):
            if time.time() - start_time > time_limit:
                return
            
            x = search_order[i]
            
            # Check if x is in S
            if x in S:
                continue
                
            if is_valid_add(x, S):
                new_S = S + [x]
                if len(new_S) > global_best_size:
                    global_best_size = len(new_S)
                    global_best_set = new_S[:]
                    
                bnb_refactored(new_S, i + 1)

    bnb_refactored([], 0)
    
    return global_best_set
