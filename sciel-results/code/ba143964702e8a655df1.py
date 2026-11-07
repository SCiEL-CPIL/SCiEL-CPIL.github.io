import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Finds a set of distinct integers in 1..256 such that:
    1. All non-empty subset sums are unique.
    2. No two distinct subset sums divide each other.
    Maximize the size of the set.
    """
    deadline = time.time() + 85  # Safety margin
    
    # Precompute forbidden bitsets for divisibility
    # Max sum for 30 elements of 256 is 7680. 
    # But we are looking for small sets first. 
    # Let's assume max sum 5000 for precomputation to be safe and fast.
    MAX_SUM = 5000
    # forbidden[v] is a bitset where bit k is 1 if v divides k and k != v
    # We only care about k > v for "v divides k" check.
    # And for "k divides v", we check if k is in forbidden[v]? No.
    # If we have sums S, and v in S, we need to ensure no u in S (u != v) such that u | v or v | u.
    # Let's define conflict[v] as bitset of all u such that u | v or v | u (u != v).
    # Then check if S & conflict[v] is non-zero for any v in S.
    
    # This precomputation is O(MAX_SUM^2 / log MAX_SUM) roughly.
    # MAX_SUM = 5000. 5000^2 = 25e6. This might be too slow for Python if done naively.
    # We can do it smarter.
    
    # Alternative: Just check on the fly for the specific sums present.
    # The number of sums is 2^n. For n=10, 1024 sums.
    # Checking divisibility for 1024 sums is fast.
    # The bottleneck is generating the sums.
    # Bitset generation is O(1) per element addition (shift and OR).
    
    def get_sums_bitset(nums):
        """Returns a bitset of all subset sums."""
        s = 0
        for x in nums:
            s |= (s << x)
        # s has bit 0 set. We want non-empty sums.
        # But for uniqueness check, we can just count bits.
        # Total subsets = 2^n.
        # If bit_count(s) != 2^n, there are collisions.
        return s

    def is_valid(nums):
        """Check validity using bitset for uniqueness and explicit loop for divisibility."""
        n = len(nums)
        if n == 0:
            return True
        s = get_sums_bitset(nums)
        # Uniqueness check
        if s.bit_count() != (1 << n):
            return False
        
        # Extract sums
        sums = []
        temp = s
        while temp:
            lsb = temp & -temp
            val = lsb.bit_length() - 1
            if val > 0: # Ignore 0
                sums.append(val)
            temp ^= lsb
        
        # Divisibility check
        # For each pair (u, v) in sums, check if u | v or v | u
        # Optimized: For each v in sums, check if any u in sums (u != v) divides v or is divided by v.
        # Since sums are sorted, we can just check multiples.
        sums.sort()
        for i, v in enumerate(sums):
            # Check if any u < v divides v
            # u must be a divisor of v.
            # We can just check all u < v.
            for j in range(i):
                u = sums[j]
                if v % u == 0:
                    return False
            # Check if any w > v is a multiple of v
            # w = k * v
            # We can break early if w > max_sum
            w = 2 * v
            # Use binary search or just iterate?
            # The list is small (<= 2^15 = 32768).
            # For v=1, w=2, 3, ...
            # For v=100, w=200, 300...
            # This is O(N * MaxSum/v).
            # Sum of MaxSum/v for all v is MaxSum * H_N.
            # This is acceptable for N=2^10 ~ 1000.
            while w <= MAX_SUM:
                if w in sums: # O(N) lookup if list, O(1) if set
                    return False
                w += v
        return True

    def is_valid_fast(nums):
        """
        Faster validity check.
        1. Bitset uniqueness.
        2. Divisibility:
           Precompute a set of sums for O(1) lookup.
           For each v in sums, check multiples.
        """
        n = len(nums)
        if n == 0:
            return True
        s = get_sums_bitset(nums)
        if s.bit_count() != (1 << n):
            return False
        
        sums_set = set()
        temp = s
        while temp:
            lsb = temp & -temp
            val = lsb.bit_length() - 1
            if val > 0:
                sums_set.add(val)
            temp ^= lsb
        
        sums_list = sorted(sums_set)
        
        for i, v in enumerate(sums_list):
            # Check divisors u < v
            # Only need to check u that are in sums_set
            # We can iterate through sums_list[:i]
            for j in range(i):
                u = sums_list[j]
                if v % u == 0:
                    return False
            # Check multiples w > v
            w = 2 * v
            while w <= MAX_SUM:
                if w in sums_set:
                    return False
                w += v
        return True

    def greedy_build(pool, rng, deadline):
        chosen = []
        available = list(pool)
        rng.shuffle(available)
        
        while len(chosen) < 30:
            if time.time() > deadline:
                break
            added = False
            # Try to add an element
            # Heuristic: try larger elements first to avoid small divisors?
            # Or smaller to keep sums small?
            # Small sums are more likely to divide large sums.
            # Large sums are less likely to be divided by small sums if they are not multiples.
            # Actually, if we have a small sum v, it forbids 2v, 3v...
            # So having small sums is "expensive" in terms of forbidden values.
            # So we should avoid creating small sums.
            # This suggests starting with larger numbers.
            
            for x in available:
                cand = chosen + [x]
                if is_valid_fast(cand):
                    chosen.append(x)
                    available.remove(x)
                    added = True
                    break
            if not added:
                break
        return chosen

    best_set = []
    
    # Strategy 1: Greedy with large numbers
    rng = random.Random(42)
    pool = list(range(1, 257))
    
    # Try different orders
    for strategy in range(20):
        if time.time() > deadline:
            break
        r = random.Random(strategy)
        # Sort by some heuristic
        # e.g., random shuffle, or sorted descending
        p = list(pool)
        if strategy % 2 == 0:
            p.sort(reverse=True)
        else:
            r.shuffle(p)
        
        cand = greedy_build(p, r, deadline)
        if len(cand) > len(best_set):
            best_set = cand
            if len(best_set) >= 10:
                break # Good enough for now? No, keep trying
        
    # Local search on best_set
    if best_set:
        cur = list(best_set)
        pool = [x for x in range(1, 257) if x not in cur]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add
            for x in pool:
                cand = cur + [x]
                if is_valid_fast(cand):
                    cur.append(x)
                    pool.remove(x)
                    improved = True
                    break
            if improved:
                continue
            
            # Try to swap
            for i in range(len(cur)):
                if time.time() > deadline:
                    break
                for x in pool:
                    cand = cur[:i] + cur[i+1:] + [x]
                    if is_valid_fast(cand):
                        # Perform swap
                        removed = cur.pop(i)
                        cur.append(x)
                        pool.remove(x)
                        pool.append(removed)
                        improved = True
                        break
                if improved:
                    break
        
        if len(cur) > len(best_set):
            best_set = cur

    # Final polish
    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-END
