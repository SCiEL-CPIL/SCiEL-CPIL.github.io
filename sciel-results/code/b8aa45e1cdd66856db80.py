import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 512
    MAXC = 48
    cache = {}

    def build_dp(S):
        """
        Builds DP bitsets for subset sums of S.
        dp[k] is a bitset where bit s is 1 if s is the sum of exactly k elements from S.
        """
        n = len(S)
        # dp[k] corresponds to using k elements
        # We only need up to n elements
        dp = [0] * (n + 1)
        dp[0] = 1  # Sum 0 with 0 elements
        for a in S:
            # Update in reverse to avoid using the same element multiple times in one step
            # k ranges from current max used down to 0
            # Since we add one element 'a', we can increase the count by 1
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= dp[k] << a
        return dp

    def get_dps(S):
        """
        Caches the full DP and the DPs for S without each element x.
        Returns: (dp_full, max_sum_full, {x: (dp_without_x, max_sum_without_x)})
        """
        key = tuple(sorted(S))
        r = cache.get(key)
        if r is not None:
            return r
        
        dp_full = build_dp(S)
        maxfull = sum(S)
        dp_without = {}
        n = len(S)
        
        # Precompute DP for S \ {x} for each x in S
        # This is expensive but cached.
        # Optimization: If n is small, this is fast. If n is large, we rely on the cache hit rate
        # or the fact that we only call this when the set changes.
        for i in range(n):
            x = S[i]
            sub = S[:i] + S[i + 1:]
            # We only need dp up to len(sub)
            dp_sub = build_dp(sub)
            max_sub = maxfull - x
            dp_without[x] = (dp_sub, max_sub)
            
        r = (dp_full, maxfull, dp_without)
        cache[key] = r
        return r

    def test_add(S, x, dp_full, maxfull, dp_without):
        """
        Checks if x can be added to S without violating the condition.
        Condition: No element in S U {x} is the mean of a subset of at least 2 OTHER elements.
        
        This splits into two cases:
        1. x is the mean of a subset of S.
           k * x = sum(subset of S), k >= 2.
        2. Some a in S becomes the mean of a subset of (S \ {a}) U {x}.
           The subset must include x (otherwise it was already a violation in S).
           So, k * a = sum(subset of (S \ {a})) + x.
           Let j be the number of elements from (S \ {a}) in the subset.
           Total elements in subset = j + 1 (including x).
           k = j + 1.
           (j + 1) * a = sum(subset of S \ {a}) + x
           sum(subset of S \ {a}) = (j + 1) * a - x
           We need to check if (j+1)*a - x is a sum of j elements from S \ {a}.
           j can range from 1 to len(S) - 1.
           Note: The problem says "subset of at least two OTHER elements".
           If the subset is {x, y1, ..., yj}, the size is j+1.
           The "other" elements for 'a' are S \ {a}.
           The subset of OTHER elements is {x, y1, ..., yj}.
           Size >= 2 => j+1 >= 2 => j >= 1.
        """
        n = len(S)
        
        # Check 1: x is mean of subset of S
        # k * x = sum of k elements from S.
        # k >= 2.
        for k in range(2, n + 1):
            t = k * x
            if t > maxfull:
                break
            if (dp_full[k] >> t) & 1:
                return False

        # Check 2: a in S is mean of subset involving x
        for a in S:
            dpo, maxo = dp_without[a]
            # j is count of elements from S \ {a}
            # j >= 1
            # Max j is n - 1
            for j in range(1, n):
                # Total elements in "other" subset = j + 1 (the x plus j others)
                # Mean condition: (j+1) * a = sum(others)
                # sum(others) = (j+1)*a - x
                # This sum must be formed by j elements from S \ {a}
                t = (j + 1) * a - x
                if t < 0 or t > maxo:
                    continue
                # Check if t is a sum of exactly j elements from S \ {a}
                if (dpo[j] >> t) & 1:
                    return False
        return True

    def verify(S):
        """
        Independent verification of the set S.
        """
        S = list(S)
        n = len(S)
        for i in range(n):
            a = S[i]
            sub = S[:i] + S[i + 1:]
            m = len(sub)
            if m < 2:
                continue
            dp = build_dp(sub)
            maxs = sum(sub)
            for k in range(2, m + 1):
                t = k * a
                if t > maxs:
                    break
                if (dp[k] >> t) & 1:
                    return False
        return True

    best = []
    best_sz = 0

    # Generate candidate list
    # Prioritize Cantor-like numbers (base 3 digits 0,1) as they are less likely to form sums
    candidates = []
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 3 == 2:
                valid = False
                break
            n //= 3
        if valid:
            candidates.append(i)
    
    # Add all other numbers
    all_nums = list(range(1, N + 1))
    other_nums = [x for x in all_nums if x not in candidates]
    
    # Shuffle other numbers for randomization
    random.shuffle(other_nums)
    
    # Combined list: Cantor numbers first, then others
    # We will shuffle the Cantor numbers too? No, let's try a few orders.
    # Strategy: Multi-order greedy.
    
    # To maximize speed, we will run a few greedy passes with different orderings.
    # Ordering 1: Cantor ascending
    # Ordering 2: Cantor descending
    # Ordering 3: Random mix
    # Ordering 4: Random mix
    
    orders = []
    orders.append(candidates + other_nums)
    orders.append(candidates[::-1] + other_nums)
    
    # Add a few random shuffles of the full set
    for _ in range(4):
        mixed = all_nums[:]
        random.shuffle(mixed)
        # Bias towards Cantor? 
        # Let's just use pure random for diversity
        orders.append(mixed)

    for order in orders:
        if time.time() > deadline - 10:
            break
        if best_sz >= MAXC:
            break
            
        cur = []
        dps = None
        for x in order:
            if len(cur) >= MAXC:
                break
            if time.time() > deadline - 10:
                break
            if dps is None:
                dps = get_dps(cur)
            if test_add(cur, x, dps[0], dps[1], dps[2]):
                cur.append(x)
                dps = None # Invalidate cache for next step
        
        if len(cur) > best_sz:
            # Verify to be safe
            if verify(cur):
                best_sz = len(cur)
                best = cur[:]

    # Phase 2: Local Search (Swap)
    # If we didn't reach max, try to improve the best set found.
    if best_sz < MAXC and time.time() < deadline - 5:
        S = best[:]
        if not S:
            S = [1, 2, 4]
            if not verify(S):
                S = []
            if not S:
                return []

        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        random.shuffle(unused)
        
        improved = True
        no_improve_counter = 0
        dps = None
        
        while improved and time.time() < deadline - 2:
            improved = False
            if dps is None:
                dps = get_dps(S)
            
            # Try to add
            for u in unused[:]:
                if len(S) >= MAXC:
                    break
                if time.time() > deadline - 2:
                    break
                if test_add(S, u, dps[0], dps[1], dps[2]):
                    S.append(u)
                    Sset.add(u)
                    unused.remove(u)
                    improved = True
                    no_improve_counter = 0
                    dps = None # Invalidate
                    break
            
            if len(S) >= MAXC:
                break
                
            if not improved:
                # Try swaps
                random.shuffle(unused)
                swapped = False
                max_swaps_to_try = 50 if time.time() > deadline - 10 else len(unused)
                
                if dps is None:
                    dps = get_dps(S)
                    
                for u in unused[:max_swaps_to_try]:
                    if time.time() > deadline - 2:
                        break
                    
                    # Find candidates s_out such that u is NOT mean of S \ {s_out}
                    # This is a necessary condition for the swap to be valid.
                    # If u IS a mean of S \ {s_out}, then in S' = S \ {s_out} U {u},
                    # u is a mean of a subset of OTHER elements (S \ {s_out}), so invalid.
                    valid_swaps = []
                    n_S = len(S)
                    for i, s_out in enumerate(S):
                        dpo, maxo = dps[2][s_out]
                        # Check if u is mean of subset of S \ {s_out}
                        is_mean = False
                        # k >= 2
                        for k in range(2, n_S): # max subset size is n_S - 1
                            t = k * u
                            if t > maxo:
                                break
                            if (dpo[k] >> t) & 1:
                                is_mean = True
                                break
                        if not is_mean:
                            valid_swaps.append(i)
                            
                    for i in valid_swaps:
                        if time.time() > deadline - 2:
                            break
                        s_out = S[i]
                        newS = S[:i] + S[i + 1:] + [u]
                        # Full verification
                        dps2 = get_dps(newS)
                        if test_add(newS, u, dps2[0], dps2[1], dps2[2]):
                            S = newS
                            Sset.remove(s_out)
                            Sset.add(u)
                            unused.remove(u)
                            unused.append(s_out)
                            improved = True
                            swapped = True
                            no_improve_counter = 0
                            dps = dps2
                            break
                    if swapped:
                        break
                
                if not swapped:
                    no_improve_counter += 1
                    if no_improve_counter > 5:
                        break

        if verify(S) and len(S) > best_sz:
            best_sz = len(S)
            best = S[:]

    # Final safety check
    if not verify(best):
        # Fallback: remove elements until valid
        # This is slow, but should rarely trigger
        for sz in range(len(best), 0, -1):
            if verify(best[:sz]):
                best = best[:sz]
                break
        else:
            best = []

    return best

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} elements")
    print(res)
