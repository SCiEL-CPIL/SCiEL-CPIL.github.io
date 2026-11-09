import time
import random
import sys

def construct():
    """
    Finds a set of distinct integers in 1..256 maximizing the size,
    such that all non-empty subset sums are distinct and no two distinct
    subset sums divide one another.
    """
    start_time = time.time()
    time_limit = 58  # Keep 2s margin for evaluator overhead

    best_set = [31, 32, 34]
    best_size = len(best_set)

    def compute_sums(S):
        """Compute all subset sums of S (including 0)."""
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums = sums | new_sums
        return sums

    def check_divisibility_antichain(non_empty_sums):
        """
        Check if non_empty_sums forms an antichain under divisibility.
        No two distinct values u, v should satisfy u|v or v|u.
        """
        sl = sorted(non_empty_sums)
        n = len(sl)
        for i in range(n):
            u = sl[i]
            for j in range(i + 1, n):
                v = sl[j]
                if v % u == 0:
                    return False
        return True

    def is_valid_set(S):
        """
        Checks if set S is valid.
        1. All subset sums are distinct.
        2. No two distinct subset sums divide one another.
        """
        if not S:
            return True
        
        sums = compute_sums(S)
        non_empty_sums = {s for s in sums if s != 0}
        
        # Check distinctness: should have exactly 2^k - 1 non-empty sums
        if len(non_empty_sums) != (1 << len(S)) - 1:
            return False
            
        # Check divisibility antichain
        if not check_divisibility_antichain(non_empty_sums):
            return False
            
        return True

    def check_validity_incremental(S, x, sums_S):
        """
        Checks if S U {x} is valid, given S is valid.
        S: list of elements
        x: new element
        sums_S: precomputed subset sums of S (including 0)
        Returns (True, new_sums) if valid, (False, None) otherwise.
        """
        if x in S:
            return False, None
            
        # Generate new sums: x + s for s in sums_S
        new_sums = {x + s for s in sums_S}
        
        # Check distinctness: sums_S and new_sums must be disjoint
        if sums_S & new_sums:
            return False, None
        
        all_sums = sums_S | new_sums
        non_empty_all = {s for s in all_sums if s != 0}
        
        if not check_divisibility_antichain(non_empty_all):
            return False, None
        
        return True, all_sums

    # Search Strategy
    # We use a combination of greedy construction with different orderings and local search.
    # Key insight: To avoid divisibility, we want sums to be "close" to each other or in ranges where multiples are unlikely.
    # Large numbers are generally safer.
    
    random.seed(42)
    
    # Pre-generate several candidate orderings
    num_restarts = 10000
    for attempt in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        S = []
        sums_S = {0}
        
        # Choose candidate ordering
        candidates = list(range(1, 257))
        
        # Strategy diversity
        r = random.random()
        if r < 0.3:
            # Descending: prefer larger numbers
            candidates.sort(reverse=True)
        elif r < 0.6:
            # Random
            random.shuffle(candidates)
        elif r < 0.8:
            # Ascending (sometimes small numbers work well if they don't divide)
            candidates.sort()
        else:
            # Bias towards upper half
            upper = list(range(129, 257))
            lower = list(range(1, 129))
            random.shuffle(upper)
            random.shuffle(lower)
            candidates = upper + lower

        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            
            # Check if we can add x
            valid, new_sums = check_validity_incremental(S, x, sums_S)
            if valid:
                S.append(x)
                sums_S = new_sums
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                
                # If we reach a large size, continue trying to add more
                # But if check becomes too slow, we'll time out naturally
                # For k >= 14, 2^k = 16384, cross-check is expensive
                # Let's cap the size we try to build greedily in one pass
                if len(S) >= 16:
                    if time.time() - start_time > time_limit:
                        break

    # Local Search / Polish
    # Try to add more elements to best_set by scanning all 1..256
    if best_set:
        S = best_set[:]
        sums_S = compute_sums(S)
        improved = True
        while improved:
            improved = False
            for x in range(1, 257):
                if x in S:
                    continue
                valid, new_sums = check_validity_incremental(S, x, sums_S)
                if valid:
                    S.append(x)
                    sums_S = new_sums
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S[:]
            # If we added something, we might be able to add more, so loop again
            # But to prevent infinite loops or excessive time, we break after one pass of additions
            # Or we can continue until no more can be added.
            if time.time() - start_time > time_limit:
                break

    # Final verification of best_set
    if best_set and len(best_set) <= 14:
        if not is_valid_set(best_set):
            # Fallback to seed
            best_set = [31, 32, 34]
            best_size = 3
    elif len(best_set) > 14:
        # For larger sets, verification is O(2^k * k^2) which might be slow, 
        # but we built it incrementally so it should be valid.
        # However, just to be safe, if it's small enough, verify.
        # If it's large, trust the incremental check.
        pass

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
