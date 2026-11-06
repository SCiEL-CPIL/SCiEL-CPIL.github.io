# EVOLVE-BLOCK-START
import time
import random

def _check_candidate_fast(S, cand):
    """
    Fast check: is cand the mean of any subset of >=2 elements of S?
    S is a sorted list of distinct ints. cand is not in S.
    """
    n = len(S)
    if n < 2:
        return True
    # For each k in 2..n, check if there exist k distinct elements in S summing to k*cand
    # This is subset sum with cardinality constraint.
    # DP: dp[j] = set of sums achievable by j elements.
    # To save memory, we can use a list of sets.
    # n <= 48. k up to 48. Sums up to 48*512 ~ 24576.
    # Actually, sums can be negative? No, elements are positive.
    # But relative to cand, we can shift.
    # Let's just use sets of sums.
    # dp[k] = set of sums of k elements.
    # We want to know if k*cand is in dp[k] for any k >= 2.
    
    # Optimization: early termination.
    # dp[0] = {0}
    # dp[k] = {s + x for s in dp[k-1] for x in S if x > last_x_used}
    # This is O(n^2 * |S|) which is bad.
    
    # Better: 1D DP for each k? No.
    # Standard 2D DP: dp[i][j] is not enough, we need sums.
    
    # Since n is small (~48), and we do this many times, we need it very fast.
    # Alternative: Randomized check? No, must be exact.
    
    # Let's try a different DP:
    # reachable[k] = set of sums.
    # reachable[0] = {0}
    # for x in S:
    #   for k in range(n, 0, -1):
    #     reachable[k] |= {s + x for s in reachable[k-1]}
    
    # This is O(n^2) set operations. Set sizes can grow large.
    # But max sum is 48*512 = 24576. Min sum is 1.
    # So the range of sums is small. We can use a bitset or boolean array.
    # Let's use a list of boolean arrays? Or just a set of integers.
    # A set of integers is fine if the number of distinct sums is not too large.
    # Max distinct sums is 24576. So sets are bounded by 24576.
    # 48 * 48 * 24576 operations? No, the set union is efficient in C.
    
    # Let's use a list of sets.
    reach = [set() for _ in range(n + 1)]
    reach[0].add(0)
    
    for x in S:
        # Update in reverse order of k to avoid using x multiple times
        for k in range(n, 0, -1):
            if reach[k-1]:
                # Optimization: if k*cand is already in reach[k], we can return False?
                # No, we are building reach[k].
                # We can check after updating.
                for s in reach[k-1]:
                    reach[k].add(s + x)
        
        # Check if k*cand is in reach[k] for any k >= 2
        # We only need to check up to n.
        # To avoid checking all k every time, we can check at the end.
        # Or check only the current k? No, sums from previous x might form the target.
        # Let's check at the end.
    
    for k in range(2, n + 1):
        if k * cand in reach[k]:
            return False
    return True

def _check_existing_elements(S, cand):
    """
    Check if adding cand invalidates any existing element a in S.
    i.e., is a the mean of a subset of (S U {cand}) of size >= 2?
    Since S is valid, a is not the mean of any subset of S.
    So the only new subsets are those containing cand.
    Let T be a subset of S. The new subset is T U {cand}.
    Size = |T| + 1. We need |T| + 1 >= 2 => |T| >= 1.
    Mean condition: (sum(T) + cand) / (|T| + 1) == a
    => sum(T) + cand = a(|T| + 1)
    => sum(T) = a|T| + a - cand
    Let k = |T|. We need a subset T of S of size k such that sum(T) = a*k + a - cand.
    
    For each a in S:
      target_sum(k) = a*k + a - cand
      Check if there is a subset of S of size k summing to target_sum(k).
      k ranges from 1 to len(S)-1 (since T is subset of S, and S has len n, max size n-1? No, T can be size n-1?
      Wait, T is a subset of S. The full set is S U {cand}. The subset is T U {cand}.
      T can be any subset of S.
      So k ranges from 1 to n.
    """
    n = len(S)
    for a in S:
        # We need to check for k = 1..n
        # Is there a subset of S of size k summing to a*k + a - cand?
        # Let's precompute reachable sums for S?
        # That would be expensive to do for each a.
        # But we can precompute a global DP for S:
        # reach_S[k] = set of sums of k elements from S.
        # This is O(n^2) set ops.
        # Then for each a, we just check membership.
        
        pass # We will compute reach_S once outside this loop in the main check function.
    
    return True

def _full_check_add(S, cand):
    """
    Full check for adding cand to S.
    1. cand is not mean of any subset of S (size>=2).
    2. No a in S is mean of any subset of S U {cand} (size>=2) that includes cand.
    """
    n = len(S)
    if n == 0:
        return True
    
    # Precompute reach_S[k] = set of sums of k elements from S.
    reach_S = [set() for _ in range(n + 1)]
    reach_S[0].add(0)
    for x in S:
        for k in range(n, 0, -1):
            if reach_S[k-1]:
                for s in reach_S[k-1]:
                    reach_S[k].add(s + x)
    
    # Check 1: cand is not mean of any subset of S (size >= 2)
    # Subset U of S, |U| >= 2, sum(U) = |U| * cand.
    for k in range(2, n + 1):
        if k * cand in reach_S[k]:
            return False
    
    # Check 2: For each a in S, no subset T of S (|T|>=1) such that
    # sum(T) + cand = a(|T|+1)  => sum(T) = a|T| + a - cand.
    for a in S:
        for k in range(1, n + 1):
            target = a * k + a - cand
            if target in reach_S[k]:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin for 90s limit
    
    best_set = []
    best_len = 0
    
    # Heuristic: Start with a small valid set and grow it.
    # Or random greedy.
    
    # Let's try multiple random greedy constructions.
    # Also, we can try to sort candidates in a specific order.
    # "Spread out" numbers might be better?
    # Or just random shuffle of 1..512.
    
    # Optimization: Pre-filter candidates?
    # If cand is very close to existing elements, it might be more likely to cause issues?
    # Not necessarily.
    
    # Strategy:
    # 1. Generate a random permutation of 1..512.
    # 2. Iterate through it, adding if valid.
    # 3. Repeat multiple times.
    # 4. Local search: try to swap out an element and in a new one.
    
    # To speed up, we can limit the number of checks per time step.
    
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Random permutation
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        current_set = []
        current_set_set = set()
        
        for cand in candidates:
            if time.time() > deadline - 2.0:
                break
            
            # Check if cand can be added
            if _full_check_add(current_set, cand):
                current_set.append(cand)
                current_set_set.add(cand)
        
        if len(current_set) > best_len:
            best_set = current_set
            best_len = len(current_set)
            # Sort best_set for consistency? Not required, but nice.
            # best_set.sort()
            
            # If we reached the maximum possible (48), we can stop?
            # The task says max 48.
            if best_len >= 48:
                return best_set
    
    # Local search on best_set
    # Try to improve best_set by swapping elements.
    # This is complex. Given time, maybe just return best_set.
    # But let's try a simple local search:
    # Try to remove each element from best_set and see if we can add more.
    # This might not help much if the greedy was good.
    
    # Alternative: Try to add any remaining number to best_set.
    if best_set:
        remaining = [x for x in range(1, 513) if x not in best_set]
        random.shuffle(remaining)
        for cand in remaining:
            if time.time() > deadline - 1.0:
                break
            if _full_check_add(best_set, cand):
                best_set.append(cand)
    
    return best_set

# EVOLVE-BLOCK-END
