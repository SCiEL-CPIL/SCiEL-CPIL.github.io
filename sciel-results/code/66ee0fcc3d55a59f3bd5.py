import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute: for each element x in [1,100], the set of "complement" elements
    # that together with x could form a subset summing to 200.
    # Actually, we need a different approach. The constraint is: no subset of S sums to 200.
    # This is a global constraint, not pairwise.
    
    # Key insight: If all elements in S are > 100, then no subset can sum to 200 
    # (minimum element is 101, so minimum 2-element sum is 101+102=203 > 200, 
    # and minimum 1-element sum is 101 < 200 but we need exactly 200).
    # Wait: {101} sums to 101, not 200. {100, 100} but elements are distinct.
    # Actually if S ⊆ {101, ..., 100} that's empty. Let me reconsider.
    
    # If S ⊆ {67, 68, ..., 100}: 
    #   Min 1-element sum: 67
    #   Min 2-element sum: 67+68 = 135
    #   Min 3-element sum: 67+68+69 = 204 > 200
    # So we need to check if any subset sums to exactly 200.
    # With elements 67..100, can we get 200? 
    #   100+100 not allowed (distinct). 100+99+1? No, 1 not in set.
    #   100+67+33? 33 not in set.
    #   99+101? 101 not in set.
    #   Let me check: 100+99 = 199, need 1 more, not available.
    #   100+98 = 198, need 2, not available.
    #   100+97 = 197, need 3, not available.
    #   ...
    #   100+67 = 167, need 33, not available.
    #   99+98 = 197, need 3, not available.
    #   99+97 = 196, need 4, not available.
    #   ...
    #   99+67 = 166, need 34, not available.
    #   98+97 = 195, need 5, not available.
    #   ...
    #   67+68 = 135, need 65, not available.
    #   67+68+69 = 204 > 200.
    # So {67, ..., 100} has size 34 and is valid!
    
    # Can we do better? What about including 66?
    # {66, 67, ..., 100}: size 35.
    #   66+67+68 = 201 > 200.
    #   66+134? 134 not in set.
    #   66+67+67 not allowed.
    #   66+68+66 not allowed.
    #   66+74+60? 74 in set, 60 not.
    #   66+73+61? 61 not in set.
    #   66+72+62? 62 not in set.
    #   66+71+63? 63 not in set.
    #   66+70+64? 64 not in set.
    #   66+69+65? 65 not in set.
    #   66+68+66 not allowed.
    #   66+67+67 not allowed.
    #   66+100+34? 34 not in set.
    #   66+99+35? 35 not in set.
    #   ...
    #   66+67+68 = 201, too big.
    #   So no 3-element subset with 66 sums to 200.
    #   2-element: 66+x=200 => x=134, not in set.
    #   1-element: 66≠200.
    #   4-element: min is 66+67+68+69=270>200.
    # So {66, ..., 100} is valid with size 35!
    
    # Can we add 65?
    # {65, 66, ..., 100}: size 36.
    #   65+66+69 = 200! So this is NOT valid.
    #   65+67+68 = 200! Also not valid.
    # So we can't simply add 65.
    
    # Strategy: Start with {66,...,100} (size 35) and try to add elements < 66 
    # while removing conflicting elements to maintain validity.
    
    def is_valid(S):
        """Check if no subset of S sums to 200 using bitset DP."""
        # Use a boolean array for subset sums up to 200
        possible = [False] * 201
        possible[0] = True
        for x in S:
            if x > 200:
                continue
            # Update in reverse to avoid using x multiple times
            for s in range(200, x - 1, -1):
                if possible[s - x]:
                    possible[s] = True
                    if s == 200:
                        return False
        return True
    
    def find_witness(S):
        """Find a subset of S summing to 200, return it or None."""
        S_list = sorted(S)
        n = len(S_list)
        # dp[s] = bitmask of indices used to achieve sum s
        # Use parent tracking for efficiency
        parent = {}  # (s, i) -> (s-x, i-1) for reconstruction
        possible = {0}
        
        for i, x in enumerate(S_list):
            if x > 200:
                continue
            new_sums = set()
            for s in list(possible):
                if s + x <= 200:
                    new_sums.add(s + x)
                    parent[(s + x, i)] = (s, i - 1)
            possible |= new_sums
            if 200 in possible:
                break
        
        if 200 not in possible:
            return None
        
        # Reconstruct
        witness = set()
        s = 200
        i = len(S_list) - 1
        while s > 0:
            while i >= 0:
                if (s, i) in parent:
                    prev_s, prev_i = parent[(s, i)]
                    witness.add(S_list[i])
                    s = prev_s
                    i = prev_i
                    break
                i -= 1
            else:
                break
        return witness
    
    def local_search(S, dl):
        """Improve S by trying to add elements and do swaps."""
        S = set(S)
        changed = True
        while changed:
            changed = False
            if time.time() > dl:
                break
            
            candidates = sorted([x for x in range(1, 101) if x not in S])
            
            # Try additions in order of how "safe" they are
            # Prefer larger elements (fewer ways to combine to 200)
            for x in reversed(candidates):
                if time.time() > dl:
                    break
                S_trial = S | {x}
                if is_valid(S_trial):
                    S = S_trial
                    changed = True
            
            # Try swaps: remove one from S, add one not in S
            if time.time() > dl:
                break
            S_list = sorted(S)
            for x_remove in S_list:
                if time.time() > dl:
                    break
                S_trial = S - {x_remove}
                candidates_now = sorted([x for x in range(1, 101) if x not in S_trial])
                for x_add in reversed(candidates_now):
                    if time.time() > dl:
                        break
                    if x_add == x_remove:
                        continue
                    S_trial2 = S_trial | {x_add}
                    if is_valid(S_trial2):
                        S = S_trial2
                        changed = True
                        break
        
        return S
    
    # Start with the known valid set {66, ..., 100}
    S = set(range(66, 101))
    
    # Try multiple random restarts with different initial sets
    best_set = S.copy()
    best_size = len(S)
    
    # Strategy 1: Start from {66,...,100} and local search
    if time.time() < deadline - 10:
        S1 = set(range(66, 101))
        S1 = local_search(S1, deadline - 10)
        if len(S1) > best_size:
            best_size = len(S1)
            best_set = S1.copy()
    
    # Strategy 2: Greedy construction from scratch
    if time.time() < deadline - 10:
        S2 = set()
        # Add elements in order of "least conflict" - larger elements first
        for x in range(100, 0, -1):
            if time.time() > deadline - 10:
                break
            S_trial = S2 | {x}
            if is_valid(S_trial):
                S2 = S_trial
        # Local search to improve
        S2 = local_search(S2, deadline - 5)
        if len(S2) > best_size:
            best_size = len(S2)
            best_set = S2.copy()
    
    # Strategy 3: Greedy with different order
    if time.time() < deadline - 5:
        S3 = set()
        # Try adding from both ends
        for x in [100, 1, 99, 2, 98, 3, 97, 4, 96, 5, 95, 6, 94, 7, 93, 8, 92, 9, 91, 10,
                  90, 11, 89, 12, 88, 13, 87, 14, 86, 15, 85, 16, 84, 17, 83, 18, 82, 19, 81, 20,
                  80, 21, 79, 22, 78, 23, 77, 24, 76, 25, 75, 26, 74, 27, 73, 28, 72, 29, 71, 30,
                  70, 31, 69, 32, 68, 33, 67, 34, 66, 35, 65, 36, 64, 37, 63, 38, 62, 39, 61, 40,
                  60, 41, 59, 42, 58, 43, 57, 44, 56, 45, 55, 46, 54, 47, 53, 48, 52, 49, 51, 50]:
            if time.time() > deadline - 5:
                break
            S_trial = S3 | {x}
            if is_valid(S_trial):
                S3 = S_trial
        S3 = local_search(S3, deadline - 2)
        if len(S3) > best_size:
            best_size = len(S3)
            best_set = S3.copy()
    
    # Strategy 4: Randomized greedy
    if time.time() < deadline - 5:
        random.seed(42)
        S4 = set()
        elems = list(range(1, 101))
        random.shuffle(elems)
        for x in elems:
            if time.time() > deadline - 5:
                break
            S_trial = S4 | {x}
            if is_valid(S_trial):
                S4 = S_trial
        S4 = local_search(S4, deadline - 2)
        if len(S4) > best_size:
            best_size = len(S4)
            best_set = S4.copy()
    
    # Strategy 5: Start with {67,...,100} (size 34) and try to add more
    if time.time() < deadline - 5:
        S5 = set(range(67, 101))
        S5 = local_search(S5, deadline - 2)
        if len(S5) > best_size:
            best_size = len(S5)
            best_set = S5.copy()
    
    # Strategy 6: More aggressive local search with multiple passes
    if time.time() < deadline - 2:
        S6 = best_set.copy()
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            S6 = local_search(S6, deadline - 2)
        if len(S6) > best_size:
            best_size = len(S6)
            best_set = S6.copy()
    
    # Final validation
    if not is_valid(best_set):
        # Fallback to known valid set
        best_set = set(range(66, 101))
    
    if not best_set:
        best_set = set(range(66, 101))
    
    return list(best_set)
