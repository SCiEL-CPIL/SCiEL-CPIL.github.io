"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 256
    deadline = time.time() + 82.0  # Safety margin before 90s kill
    
    # The proven safe tail: {k, ..., n} is valid if for all a<b<=c in tail, a ∤ (b+c)
    # For the tail [k, n], the worst case is a=k, b=k+1, c=n (or similar)
    # Actually the constraint for the pure tail [k,n]: 
    #   For a=k, we need k ∤ (b+c) for all k<b<=c<=n
    #   b+c ranges from (k+1)+(k+2)=2k+3 to (n-1)+n=2n-1
    #   So we need k ∤ s for all s in [2k+3, 2n-1]
    #   The multiples of k in that range: ceil((2k+3)/k)*k to floor((2n-1)/k)*k
    #   ceil((2k+3)/k) = 2 + ceil(3/k) = 3 for k>=2
    #   So multiples are 3k, 4k, ..., floor((2n-1)/k)*k
    #   We need 3k > 2n-1, i.e., 3k >= 2n, i.e., k >= ceil(2n/3) = ceil(512/3) = 171
    # So [171, 256] is the maximal pure tail.
    
    tail_start = 171
    base_set = set(range(tail_start, n + 1))
    base_size = len(base_set)  # 86
    
    # Precompute: for each a in [1, n], the set of sums s=a*(a+1) is irrelevant.
    # We need: for a, forbidden sums are multiples of a.
    # For checking if we can add x to a set S:
    #   x as 'a': for all b,c in S with x<b<=c, need x ∤ (b+c)
    #   x as 'b': for all a,c in S with a<x<=c, need a ∤ (x+c)
    #   x as 'c': for all a,b in S with a<b<=x, need a ∤ (b+x)
    
    # Key insight for speed: use bitsets or precomputed conflict maps.
    # Since n=256, we can use numpy bitsets for fast set operations.
    import numpy as np
    
    # Represent sets as boolean arrays of length n+1
    def set_to_arr(S):
        arr = np.zeros(n + 1, dtype=bool)
        for x in S:
            arr[x] = True
        return arr
    
    def arr_to_set(arr):
        return set(np.where(arr)[0].tolist())
    
    base_arr = set_to_arr(base_set)
    
    # Precompute for each a: which sums are multiples of a (for sum range up to 2n)
    # For a given a, forbidden sums are a*2, a*3, ..., a*floor(2n/a)
    # But we only care about sums that can be formed by pairs in S.
    
    # Faster approach: for each candidate x, check validity using bit operations
    def can_add(arr, x):
        """Check if x can be added to the set represented by arr."""
        if arr[x]:
            return False
        n_local = n
        
        # 1. x as smallest (a): need x ∤ (b+c) for all b,c in set with x<b<=c
        #    i.e., for all b in set with b>x, for all c in set with c>=b, (b+c) % x != 0
        #    Equivalently: no pair (b,c) in set with x<b<=c such that (b+c) is a multiple of x.
        #    Let's check: for each multiple m of x where m = b+c, b>x, c>=b, b,c in set
        #    b+c = m, b > x, c >= b => b <= m/2, c = m-b >= b => b <= m/2
        #    So b in (x, m/2], c = m-b >= b, both in set
        #    
        #    Efficient check: for each multiple m of x with m >= 2*(x+1) (min b+c) and m <= 2*n:
        #    Check if exists b in set with x < b <= m//2 and (m-b) in set and (m-b) >= b
        #    
        #    Alternative: iterate b in set with b > x, check if (b+c) % x == 0 for any c in set with c >= b
        #    i.e., for b in set, b > x: need no c in set with c >= b and (b+c) % x == 0
        #    c ≡ -b (mod x), c >= b, c in set
        #    The smallest such c is c0 = b + ((-b) % x) = b + x - (b % x) if b%x != 0, else b
        #    Actually: c ≡ -b (mod x) => c = b + k*x for integer k, c >= b
        #    k=0: c=b, valid if b%x==0 (then c=b, b<=c OK)
        #    k=1: c=b+x
        #    etc.
        
        # Let me just do it directly but efficiently with numpy
        S = arr_to_set(arr)  # This is O(n) each time, but n=256 so it's fine
        
        # Actually let's keep the set alongside for this
        return True  # placeholder, will implement properly below
    
    # Let me use a cleaner approach with sets for clarity, since n=256 is small
    # The bottleneck is the number of additions we can try, not individual checks
    
    def is_valid_addition(S, x):
        """Check if x can be added to set S (as a sorted set)."""
        # x as 'a' (smallest): x ∤ (b+c) for all b,c in S with x < b <= c
        for b in S:
            if x < b:
                for c in S:
                    if b <= c and (b + c) % x == 0:
                        return False
        # x as 'b' (middle): a ∤ (x+c) for all a,c in S with a < x <= c
        for a in S:
            if a < x:
                for c in S:
                    if x <= c and (x + c) % a == 0:
                        return False
        # x as 'c' (largest): a ∤ (b+x) for all a,b in S with a < b <= x
        for a in S:
            for b in S:
                if a < b <= x and (b + x) % a == 0:
                    return False
        return True
    
    # Optimize: precompute for each a, the residues that are "dangerous"
    # Actually, let's use a different strategy: iterative improvement with local search
    
    current_set = set(base_set)
    best_set = set(current_set)
    best_size = len(current_set)
    
    candidates = list(range(1, tail_start))
    
    # Greedy pass: try adding in ascending order (smaller numbers are harder to place)
    for x in sorted(candidates):
        if is_valid_addition(current_set, x):
            current_set.add(x)
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
    
    # Multiple random restarts with different orderings
    import random
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        current_set = set(base_set)
        cands = list(range(1, tail_start))
        random.shuffle(cands)
        
        for x in cands:
            if time.time() > deadline - 10:
                break
            if is_valid_addition(current_set, x):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
    
    # Local search: try swaps
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        
        improved = False
        current = set(best_set)
        
        for x in list(current):
            if time.time() > deadline - 5:
                break
            # Try removing x and adding a new element
            current.remove(x)
            for y in range(1, n + 1):
                if y in current:
                    continue
                if is_valid_addition(current, y):
                    current.add(y)
                    if len(current) > len(best_set):
                        best_set = set(current)
                        improved = True
                    current.remove(y)
                    break
            if not improved:
                current.add(x)  # put x back
                improved = True  # mark as processed
    
    # Final verification
    def is_valid_full(S):
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if a < b <= c and (b + c) % a == 0:
                        return False
        return True
    
    if not is_valid_full(best_set):
        # Fallback to safe tail
        best_set = set(range(192, 257))
    
    return sorted(list(best_set))
# EVOLVE-BLOCK-END
