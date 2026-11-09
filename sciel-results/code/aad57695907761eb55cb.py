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
        dp = [0] * (n + 1)
        dp[0] = 1  # Sum 0 with 0 elements
        for a in S:
            # Update in reverse to avoid using the same element multiple times in one step
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
        
        for i in range(n):
            sub = S[:i] + S[i + 1:]
            dp_without[S[i]] = (build_dp(sub), maxfull - S[i])
        
        r = (dp_full, maxfull, dp_without)
        cache[key] = r
        return r

    def test_add(S, x, dp_full, maxfull, dp_without):
        """
        Checks if x can be added to S without violating the mean condition.
        1. x is not the mean of any subset of S (size >= 2).
        2. No element a in S becomes the mean of any subset of S U {x} containing x.
        """
        n = len(S)
        # Check 1: x not the mean of any subset of S (size>=2)
        for k in range(2, n + 1):
            t = k * x
            if t > maxfull:
                break
            if (dp_full[k] >> t) & 1:
                return False
        
        # Check 2: no a in S becomes mean of a subset containing x
        # Subset T of S U {x} \ {a} has size j+1, sum (j+1)*a.
        # T contains x. Let T' = T \ {x}. T' is subset of S \ {a} of size j.
        # Sum(T') = (j+1)*a - x.
        for a in S:
            dpo, maxo = dp_without[a]
            m = n - 1  # Size of S \ {a}
            # j can range from 1 to m (since T size >= 2, and T contains x, T' size >= 1)
            # Max T size is n (since S U {x} has n+1 elements, remove a -> n elements).
            # So j can be up to n-1.
            for j in range(1, m + 1):
                t = (j + 1) * a - x
                if t < 0 or t > maxo:
                    continue
                if (dpo[j] >> t) & 1:
                    return False
        return True

    def verify(S):
        """
        Verifies the entire set S.
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

    # Precompute conflict scores (degree in 3-AP conflict graph)
    # deg[x] counts how many other numbers y are in a 3-AP with x
    # This serves as a proxy for "how constrained" x is.
    deg = [0] * (N + 1)
    for x in range(1, N + 1):
        # Count y such that x is mean of y, z -> 2x = y+z
        # y and z must be distinct, != x, in [1, N]
        # y = 2x - z
        # z ranges from 1 to N, z != x
        # y must be in [1, N], y != x
        # We just count the number of valid z
        # Actually, we want the degree in the graph where edge (u,v) exists if they form a 3-AP with some w.
        # Simpler: Count number of 3-APs containing x.
        count = 0
        # Case 1: x is the mean. 2x = a+b. a != b, a,b != x.
        # a+b = 2x. Pairs (a,b) with a < b.
        # a ranges from 1 to x-1. b = 2x - a.
        # b must be <= N and b != x (implied if a != x) and a != b (implied if a != x).
        # If a=x, b=x, which is not allowed (distinct elements).
        for a in range(1, x):
            b = 2 * x - a
            if b <= N and b != x:
                count += 1
        # Case 2: x is an endpoint. 
        # Subcase 2a: x is the smaller end. 2b = x + c -> c = 2b - x.
        # b > x. c > b.
        # Subcase 2b: x is the larger end. 2b = a + x -> a = 2b - x.
        # a < b < x.
        
        # It's easier to just iterate all pairs (u, v) and if they form a 3-AP, increment deg for all 3.
        # But that is O(N^2) or O(N^3). N=512 is small enough for O(N^2) precomputation.
        pass # We'll do a more direct calculation below if needed, or just use a simpler heuristic.
        
    # Let's use a simpler heuristic for ordering: 
    # Numbers with fewer divisors or specific modular properties might be better?
    # Actually, the 3-AP count is the most direct "conflict" metric.
    # Let's compute 3-AP involvement properly.
    is_3ap = [[False]*(N+1) for _ in range(N+1)] # is_3ap[a][b] is True if a,b are in a 3-AP
    # This is too memory heavy? 512x512 booleans is fine.
    # Actually, we just need the degree.
    deg = [0] * (N + 1)
    # Iterate all possible 3-APs (a, b, c) with a < b < c and 2b = a+c
    # a from 1 to N-2
    # c from a+2 to N
    # b = (a+c)/2 must be integer
    for a in range(1, N - 1):
        # c must have same parity as a
        for c in range(a + 2, N + 1, 2):
            b = (a + c) // 2
            if 1 <= b <= N:
                # a, b, c form a 3-AP
                deg[a] += 1
                deg[b] += 1
                deg[c] += 1

    best = []
    best_sz = 0

    # Phase 1: Multi-order Greedy with Conflict-Awareness
    # We create a list of candidates. We will try different strategies for ordering.
    # Strategy A: Sort by deg[x] (ascending), then randomize ties.
    # Strategy B: Random shuffle.
    
    candidates = list(range(1, N + 1))
    
    # Pre-sort by degree for Strategy A
    candidates_by_deg = sorted(candidates, key=lambda x: (deg[x], random.random()))
    
    num_restarts = 60 # Increased restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        # Alternate between pure random and degree-ordered
        if i % 2 == 0:
            order = candidates_by_deg[:]
            # Add some noise to the degree ordering to avoid getting stuck in one local min
            # Shuffle the first 10% or just randomize ties more aggressively
            # Actually, let's just randomize the entire list but bias towards low deg?
            # Simple approach: Sort by (deg[x] + random.uniform(0, 5), random.random())
            noisy_order = sorted(candidates, key=lambda x: (deg[x] + random.uniform(0, 10), random.random()))
            order = noisy_order
        else:
            order = candidates[:]
            random.shuffle(order)

        for rev in [False, True]:
            if time.time() > deadline - 15:
                break
            current_order = order[::-1] if rev else order
            cur = []
            dps = None
            for x in current_order:
                if len(cur) >= MAXC:
                    break
                if dps is None:
                    dps = get_dps(cur)
                if test_add(cur, x, dps[0], dps[1], dps[2]):
                    cur.append(x)
                    dps = None
            
            if verify(cur) and len(cur) > best_sz:
                best_sz = len(cur)
                best = cur[:]
        
        if best_sz >= MAXC:
            break

    # Phase 2: Local Search (add + swap)
    if best_sz < MAXC and time.time() < deadline - 5:
        S = best[:]
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        
        # Sort unused by degree as well, to try "safer" swaps first
        unused.sort(key=lambda x: deg[x])
        
        improved = True
        no_improve_counter = 0
        
        while improved and time.time() < deadline - 2:
            improved = False
            dps = get_dps(S)
            
            # Try to add
            # We iterate through unused. Since unused is sorted by deg, we try safe ones first.
            for u in unused[:]:
                if len(S) >= MAXC:
                    break
                if test_add(S, u, dps[0], dps[1], dps[2]):
                    S.append(u)
                    Sset.add(u)
                    unused.remove(u)
                    improved = True
                    no_improve_counter = 0
                    break
            
            if len(S) >= MAXC:
                break
                
            if not improved:
                # Try swaps
                # We want to swap s in S with u in unused.
                # To improve, we should remove a "high conflict" s and add a "low conflict" u.
                # So iterate u (low deg) and s (high deg).
                
                # Sort S by deg descending to try removing the most constrained elements first
                S_sorted_by_deg = sorted(S, key=lambda x: deg[x], reverse=True)
                
                swapped = False
                # Limit swap search for speed
                max_swaps_to_try = len(unused)
                if time.time() > deadline - 8:
                    max_swaps_to_try = min(len(unused), 50)

                for u in unused[:max_swaps_to_try]:
                    if time.time() > deadline - 2:
                        break
                    for s in S_sorted_by_deg:
                        # Find index of s in S
                        # S is small, so linear search is fine
                        idx = S.index(s)
                        newS = S[:idx] + S[idx+1:]
                        # Check if u can be added to newS
                        dps2 = get_dps(newS)
                        if test_add(newS, u, dps2[0], dps2[1], dps2[2]):
                            S = newS + [u]
                            Sset.remove(s)
                            Sset.add(u)
                            unused.remove(u)
                            unused.append(s)
                            # Re-sort unused to keep "safe" ones at the front? 
                            # This is O(N log N). N=464. Might be slow if done every swap.
                            # Let's just append and let the next iteration handle it, or re-sort occasionally.
                            improved = True
                            swapped = True
                            no_improve_counter = 0
                            break
                    if swapped:
                        break
                
                if not swapped:
                    no_improve_counter += 1
                    if no_improve_counter > 10:
                        break

        if verify(S) and len(S) > best_sz:
            best_sz = len(S)
            best = S[:]

    # Final safety
    if not verify(best):
        for sz in range(len(best), 0, -1):
            if verify(best[:sz]):
                best = best[:sz]
                break
        else:
            best = []

    return best
