import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    n = 256
    
    def is_valid_set(S):
        # S is a set of integers
        # Check: for all a < b <= c in S, a does not divide b+c
        S_list = list(S)
        S_list.sort()
        for i, a in enumerate(S_list):
            # b must be > a
            # c must be >= b
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # 1. Find the best interval [L, 256]
    # Note: The "L check" logic in the original code had a flaw: it only checked if a multiple of L
    # exists in the sum range, but didn't ensure that the specific pairs (b,c) exist in the set.
    # For an interval [L, N], if a=L, b,c in [L+1, N].
    # Sums b+c range from 2(L+1) to 2N.
    # If there is a multiple of L in this range, is it always representable?
    # Yes, any integer S in [2(L+1), 2N] can be written as b+c with L+1 <= b <= c <= N.
    # Proof: Let b = max(L+1, S-N). Then c = S-b.
    # If S <= N + (N-L-1) = 2N-L-1, we can pick b=L+1, c=S-L-1. Since S >= 2L+2, c >= L+1.
    # Since S <= 2N-L-1, c <= N-L-1+L = N? No.
    # Actually, the set of sums is exactly [2(L+1), 2N].
    # So the original check `if k_min <= k_max: valid = False` is correct for a=L.
    # However, we still need to check a > L.
    
    # A better strategy: Iterate L from high to low? Or just find the largest valid interval.
    # The largest valid interval gives the base set.
    # Then we can try to add smaller elements.
    
    best_L = 1
    best_size = 0
    best_set = set()
    
    # We want to maximize size, so we want smallest L.
    # But we also want a valid set.
    # Let's find the smallest L such that [L, 256] is valid.
    # If [L, 256] is valid, then [L+1, 256] is also valid (subset).
    # So we can binary search or just iterate.
    # Since n=256, iterating is fast.
    
    for L in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Quick check for a=L
        k_min = (2 * (L + 1) + L - 1) // L
        k_max = (2 * n) // L
        if k_min <= k_max:
            # If L fails, larger L might pass? No, if L fails, [L, 256] is invalid.
            # But [L+1, 256] might be valid.
            # We are looking for the *smallest* L that yields a valid interval.
            # If L fails, we must try L+1.
            continue
        
        # If L passes the quick check, verify full validity for [L, 256]
        S_test = set(range(L, n + 1))
        if is_valid_set(S_test):
            # Since we iterate L from 1 upwards, the first valid L gives the largest set.
            # Wait, is it guaranteed that if [L, 256] is valid, [L-1, 256] is invalid?
            # Not necessarily, but we want the smallest L.
            # So the first L we find that is valid is the best interval.
            best_L = L
            best_size = n - L + 1
            best_set = S_test
            break # Found the smallest L, so largest set. No need to check smaller L?
            # Actually, smaller L would be a superset. If [L, 256] is valid, 
            # [L-1, 256] might be invalid. We want the *largest* valid set.
            # The largest valid set containing the tail [L, 256] is [L, 256] itself 
            # (since adding L-1 might break validity).
            # So finding the smallest L such that [L, 256] is valid is the correct base.
            pass

    # If no interval was found (shouldn't happen, [256, 256] is valid), fallback
    if not best_set:
        return list(range(192, 257))

    # 2. Greedy Expansion
    # Try to add elements from best_L - 1 down to 1
    current_set = set(best_set)
    
    # Sort candidates
    candidates = list(range(best_L - 1, 0, -1))
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        # Tentatively add x
        # Check validity.
        # x is smaller than all elements in current_set.
        # So x acts as 'a'.
        # We need to ensure x does not divide b+c for any b, c in current_set with b <= c.
        
        valid = True
        c_list = list(current_set)
        for i, b in enumerate(c_list):
            for k in range(i, len(c_list)): # b <= c
                c = c_list[k]
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            current_set.add(x)

    # 3. Local Search / Polishing
    # Try to remove some elements and add others? Or just try to add more?
    # The greedy approach might get stuck.
    # We can try a simple local search: try to swap an existing element with a smaller one?
    # Or just try to add elements again in a different order?
    # Given the time budget, we can try a few restarts or a more sophisticated search.
    
    # Let's try to add elements that were skipped, maybe in a different order?
    # Or try to remove an element from the tail to add two smaller ones?
    # Example: Remove 256, add 1 and 2? Unlikely to be valid.
    
    # A better polish: Try to add elements from 1 to best_L-1 again, but maybe we can add
    # multiple elements at once?
    
    # Let's try a simple local search: 
    # For each x in [1, best_L-1], if it can be added, add it.
    # We already did this.
    
    # What if we try to remove an element y from current_set and see if we can add x?
    # This is expensive.
    
    # Let's just return the current set.
    return list(current_set)

# EVOLVE-BLOCK-END
