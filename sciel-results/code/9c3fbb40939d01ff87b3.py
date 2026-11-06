"""Rewritten program with optimized incremental validity check."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s kill
    
    def is_valid_set(s):
        """Check if the entire set s is valid."""
        n = len(s)
        if n < 4:
            return True
        for i in range(n):
            x = s[i]
            # Check if x divides sum of any 3 distinct others
            for j in range(n):
                if j == i: continue
                a = s[j]
                for k in range(n):
                    if k == i or k == j: continue
                    b = s[k]
                    for l in range(n):
                        if l == i or l == j or l == k: continue
                        c = s[l]
                        if (a + b + c) % x == 0:
                            return False
        return True

    def can_add(selected_list, candidate):
        """Check if candidate can be added to selected_list."""
        s = selected_list
        n = len(s)
        
        # 1. Check: does candidate divide sum of any 3 from s?
        # Generate sums of 3 distinct from s
        for i in range(n):
            a = s[i]
            for j in range(i+1, n):
                b = s[j]
                ab = a + b
                for k in range(j+1, n):
                    c = s[k]
                    if (ab + c) % candidate == 0:
                        return False
        
        # 2. Check: for each a in s, does a divide sum of any 3 from (s\{a}) U {candidate}?
        # The new triples involving candidate are {candidate, u, v} where u, v in s\{a}
        for i, a in enumerate(s):
            # u, v are distinct elements from s, neither equal to a
            # Iterate pairs from s, skipping a
            for j in range(n):
                if j == i: continue
                u = s[j]
                for k in range(j+1, n):
                    if k == i: continue
                    v = s[k]
                    if (candidate + u + v) % a == 0:
                        return False
        
        return True

    def greedy_search(order):
        """Run greedy selection on a given order of candidates."""
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if len(selected) >= 48:
                break
            if can_add(selected, x):
                selected.append(x)
        return selected

    # Heuristic 1: Descending order (Large numbers first)
    candidates = list(range(192, 0, -1))
    best_set = greedy_search(candidates)
    best_len = len(best_set)
    
    # Heuristic 2: Ascending order
    if time.time() < deadline - 5:
        candidates_asc = list(range(1, 193))
        set_asc = greedy_search(candidates_asc)
        if len(set_asc) > best_len:
            best_set = set_asc
            best_len = len(best_set)

    # Heuristic 3: Randomized restarts
    # We can try multiple random orders
    # To make it effective, we can bias towards larger numbers or numbers with fewer divisors
    # For now, simple random shuffles of the full range
    num_restarts = 50
    for _ in range(num_restarts):
        if time.time() > deadline - 10: # Leave 10s for local search or final check
            break
        if best_len >= 48:
            break
            
        random.shuffle(candidates)
        # Bias: sort by value descending with some randomness?
        # Or just random shuffle
        current_set = greedy_search(candidates)
        if len(current_set) > best_len:
            best_set = current_set
            best_len = len(best_set)
            # If we found a good one, we can try to improve it with local search
            
    # Local Search on best_set
    # Try 1-swap: remove a, add b
    # To speed up, we only check validity of the new set if it looks promising
    # Or we can use a faster check for the specific change?
    # Given time constraints, we might skip complex local search if greedy is fast enough.
    # However, a simple local search might help if we are stuck at 47.
    
    if best_len < 48 and time.time() < deadline - 5:
        # Try to swap elements in best_set
        # We will try to replace small elements with large ones
        # Sort best_set ascending
        best_set_sorted = sorted(best_set)
        # Try to remove the smallest element and add the largest available
        # This is a heuristic local search
        
        # Build a set of available candidates
        selected_set = set(best_set)
        available = [x for x in range(192, 0, -1) if x not in selected_set]
        
        # Try swapping the smallest element in best_set with the largest in available
        # Repeat until no improvement or time out
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Find the smallest element in best_set
            min_idx = 0
            min_val = best_set[0]
            for i in range(1, len(best_set)):
                if best_set[i] < min_val:
                    min_val = best_set[i]
                    min_idx = i
            
            # Try to replace min_val with candidates from available (largest first)
            for b in available:
                # Create new set
                new_set = best_set[:]
                new_set[min_idx] = b
                # Check if new_set is valid
                # We can use is_valid_set but it's O(N^4) if naive, or O(N^3) if optimized.
                # Let's use a faster check for the specific change?
                # Actually, we can just call can_add logic? No, we removed one and added one.
                # We need to check the whole set.
                
                # Optimized check for new_set
                # Since we only changed one element, we can check:
                # 1. Does b divide sum of 3 from (best_set \ {min_val})?
                # 2. Does min_val divide sum of 3 from (best_set \ {min_val}) U {b}? -> No, min_val is gone.
                # 3. Do other elements in best_set \ {min_val} divide sum of 3 involving b?
                
                # Let's just use the full is_valid_set for simplicity if it's fast enough.
                # N=48, N^4 is too slow.
                # Let's use the can_add logic on the fly?
                # We can reconstruct the check.
                
                # Check 1: b vs best_set \ {min_val}
                # Check 2: Others vs (best_set \ {min_val}) U {b}
                
                # This is essentially checking validity of the new set.
                # Let's write a helper to check validity of a set given that most of it was valid.
                
                # For now, let's just try a few swaps and use a quick full validation.
                # If it's too slow, we break.
                
                # Quick full validation
                valid = True
                s = new_set
                n = len(s)
                for i in range(n):
                    x = s[i]
                    # Check x divides sum of 3 others
                    found = False
                    for j in range(n):
                        if j == i: continue
                        a = s[j]
                        for k in range(n):
                            if k == i or k == j: continue
                            b_val = s[k]
                            for l in range(n):
                                if l == i or l == j or l == k: continue
                                c_val = s[l]
                                if (a + b_val + c_val) % x == 0:
                                    valid = False
                                    found = True
                                    break
                            if found: break
                        if found: break
                    if not valid: break
                
                if valid:
                    best_set = new_set
                    selected_set.add(b)
                    selected_set.remove(min_val)
                    available.remove(b)
                    available.append(min_val)
                    improved = True
                    break # Restart swap loop with new set

    return best_set

# EVOLVE-BLOCK-END
