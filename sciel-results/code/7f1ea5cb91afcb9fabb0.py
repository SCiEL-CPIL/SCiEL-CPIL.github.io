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
    time_limit = 55  # Keep 5s margin for evaluator overhead
    
    best_set = [31, 32, 34]
    best_size = len(best_set)
    
    def is_valid_set(S):
        """
        Checks if set S is valid.
        1. All subset sums are distinct.
        2. No two distinct subset sums divide one another.
        """
        if not S:
            return True
            
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums = sums | new_sums
        
        non_empty_sums = {s for s in sums if s != 0}
        
        # Check distinctness of sums
        if len(non_empty_sums) != (1 << len(S)) - 1:
            return False
            
        # Check divisibility using sorted list for early termination
        sum_list = sorted(non_empty_sums)
        n = len(sum_list)
        for i in range(n):
            u = sum_list[i]
            for j in range(i + 1, n):
                v = sum_list[j]
                if v % u == 0:
                    return False
        return True

    def check_validity_incremental(sums_S, S, x):
        """
        Checks if S U {x} is valid, given S is valid and sums_S is the set of all subset sums of S.
        Returns (new_sums, valid) where new_sums is sums of S U {x} if valid.
        """
        if x in S:
            return None, False
            
        # Generate new sums: x + s for s in sums_S
        new_sums = {x + s for s in sums_S}
        
        # Check distinctness: intersection must be empty (excluding 0 which is only in sums_S)
        # If any value appears in both sums_S and new_sums, two different subsets have same sum
        if sums_S & new_sums:
            return None, False
        
        # Combined sums
        all_sums = sums_S | new_sums
        non_empty_all = {s for s in all_sums if s != 0}
        
        # Check divisibility
        # We know sums_S \ {0} is an antichain.
        # Need to check:
        # a) new_sums \ {0} is an antichain internally
        # b) No u in sums_S \ {0} divides v in new_sums \ {0}, and vice versa
        
        non_empty_new = sorted(s for s in new_sums if s != 0)
        non_empty_old = sorted(s for s in sums_S if s != 0)
        
        # Internal check for new_sums
        for i in range(len(non_empty_new)):
            u = non_empty_new[i]
            for j in range(i + 1, len(non_empty_new)):
                v = non_empty_new[j]
                if v % u == 0:
                    return None, False
        
        # Cross check between old and new
        # For each u in non_empty_new, check if any v in non_empty_old divides u or u divides v
        for u in non_empty_new:
            for v in non_empty_old:
                if v > u:
                    if u > 0 and v % u == 0:
                        return None, False
                else:
                    if v > 0 and u % v == 0:
                        return None, False
        
        return all_sums, True

    # Search Strategy
    # 1. Greedy with random restarts
    # 2. Try to build a set of size up to 12-15
    
    random.seed(42)
    
    max_attempts = 5000
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
        
        S = []
        sums_S = {0}
        
        # Start with a random small valid set or empty
        # Let's start empty and add greedily from a shuffled list
        # Use larger elements first to reduce divisibility conflicts
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        # Sort in descending order to try larger elements first
        candidates.sort(reverse=True)
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            
            # Check if we can add x
            new_sums, valid = check_validity_incremental(sums_S, S, x)
            if valid:
                S.append(x)
                sums_S = new_sums
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
            
            # If S becomes too large to check quickly, stop this branch
            # check_validity_incremental complexity is roughly O(2^k * 2^k) for cross check
            # k=10 -> ~1M. Fast.
            # k=12 -> ~16M. Slow in Python (~1s).
            # k=14 -> ~268M. Too slow.
            
            if len(S) > 14:
                break
    
    # Final verification of best_set to ensure no bugs in incremental logic
    # If best_set is small enough, verify fully.
    if len(best_set) <= 12:
        if not is_valid_set(best_set):
            # Fallback to seed
            best_set = [31, 32, 34]
    
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
