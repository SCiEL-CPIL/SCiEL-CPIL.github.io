import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 512
    MAXC = 48

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

    def verify(S):
        """
        Verifies the entire set S.
        No element in S is the mean of any subset of at least 2 OTHER elements.
        Returns True if valid, False otherwise.
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

    def find_violation(S):
        """
        Finds a violation in S. Returns (element, subset) or None.
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
                    return (a, k)
        return None

    # Precompute 3-AP involvement degrees
    deg = [0] * (N + 1)
    for a in range(1, N - 1):
        for c in range(a + 2, N + 1, 2):
            b = (a + c) // 2
            if 1 <= b <= N:
                deg[a] += 1
                deg[b] += 1
                deg[c] += 1

    pair_conflict = [0] * (N + 1)
    for x in range(1, N + 1):
        for u in range(1, x):
            v = 2 * x - u
            if v > x and v <= N:
                pair_conflict[x] += 1

    best = []
    best_sz = 0

    def fast_check_add(S_set, S_list, x):
        """
        Checks if x can be added to S without creating a 3-AP (pair mean).
        S_set: set of elements in S
        S_list: list of elements in S
        x: candidate
        Returns True if no pair (u,v) in S such that (u+v)/2 == x or x is mean of u,v?
        Actually:
        1. x is not mean of any pair in S: (u+v)/2 != x for all u,v in S.
           Equivalent to: 2x != u+v.
        2. No u in S is mean of x and v in S: u != (x+v)/2 => 2u != x+v => v != 2u-x.
        """
        # Check 1: 2x is not sum of any pair in S
        # This is hard to do fast without precomputation.
        # But we can just iterate?
        # For speed, we can check if any v in S_set such that 2x-v in S_set and v != 2x-v?
        # Actually, if u+v=2x, then v=2x-u.
        # We just need to know if there exists u in S such that 2x-u in S.
        # This is O(|S|).
        
        for u in S_list:
            v = 2 * x - u
            if v == x:
                continue # u=x is not in S
            if v in S_set:
                # Found pair u, v in S with mean x
                return False
        
        # Check 2: For each u in S, is 2u-x in S?
        for u in S_list:
            v = 2 * u - x
            if v == x:
                continue
            if v in S_set:
                # Found pair x, v in S with mean u
                return False
                
        return True

    def try_greedy(order):
        """Greedily build a set following the given order, using fast check then slow verify."""
        cur = []
        cur_set = set()

        for x in order:
            if len(cur) >= MAXC:
                break
            if time.time() > deadline - 10:
                break
            
            # Fast check
            if fast_check_add(cur_set, cur, x):
                cur.append(x)
                cur_set.add(x)
        
        # Slow verify the result
        if cur:
            if verify(cur):
                return cur
            else:
                # If verification fails, remove violating elements
                # Simple strategy: remove elements involved in violations
                S = cur[:]
                Sset = set(S)
                changed = True
                while changed and S:
                    viol = find_violation(S)
                    if not viol:
                        break
                    a, k = viol
                    # Remove a
                    S.remove(a)
                    Sset.discard(a)
                    changed = True
                if S and verify(S):
                    return S
                else:
                    return []
        return []

    def local_search(S):
        """Try to improve S by adding elements or swapping."""
        if time.time() > deadline - 5:
            return S

        S = list(S)
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        unused.sort(key=lambda x: pair_conflict[x])

        improved = True
        no_improve = 0

        while improved and time.time() < deadline - 3:
            improved = False
            
            # Try to add an unused element
            added = False
            for u in unused:
                if len(S) >= MAXC:
                    break
                if time.time() > deadline - 3:
                    break
                if fast_check_add(Sset, S, u):
                    # Tentative add
                    S.append(u)
                    Sset.add(u)
                    if verify(S):
                        improved = True
                        no_improve = 0
                        added = True
                        break
                    else:
                        # Revert
                        S.pop()
                        Sset.discard(u)

            if len(S) >= MAXC:
                break

            if not added:
                # Try swaps
                S_sorted = sorted(S, key=lambda x: pair_conflict[x], reverse=True)
                swapped = False
                max_swaps_to_try = min(len(unused), 20)
                
                for u in unused[:max_swaps_to_try]:
                    if time.time() > deadline - 3:
                        break
                    for s in S_sorted:
                        idx = S.index(s)
                        newS = S[:idx] + S[idx + 1:]
                        newSset = set(newS)
                        
                        # Check if u can replace s
                        # First, remove s, then try to add u
                        if fast_check_add(newSset, newS, u):
                            newS.append(u)
                            if verify(newS):
                                S = newS
                                Sset = set(S)
                                unused.remove(u)
                                unused.append(s)
                                unused.sort(key=lambda x: pair_conflict[x])
                                improved = True
                                swapped = True
                                break
                    if swapped:
                        break

                if not swapped:
                    no_improve += 1
                    if no_improve > 5:
                        break

        return S

    # Phase 1: Multiple greedy restarts
    candidates = list(range(1, N + 1))
    order1 = sorted(candidates, key=lambda x: (pair_conflict[x], deg[x], random.random()))
    order2 = sorted(candidates, key=lambda x: (deg[x], pair_conflict[x], random.random()))

    num_restarts = 100 # Increased restarts due to faster greedy
    
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break

        if i % 3 == 0:
            order = order1[:]
        elif i % 3 == 1:
            order = order2[:]
        else:
            order = candidates[:]
            random.shuffle(order)

        cur = try_greedy(order)

        if len(cur) > best_sz:
            best_sz = len(cur)
            best = cur[:]

        if best_sz >= MAXC:
            break

    # Phase 2: Local search on the best found
    if best_sz < MAXC and time.time() < deadline - 10:
        S = local_search(best[:])
        if len(S) > best_sz:
            best_sz = len(S)
            best = S[:]

    # Final safety check
    if best and not verify(best):
        # Try to fix
        S = best[:]
        Sset = set(S)
        changed = True
        while changed and S:
            viol = find_violation(S)
            if not viol:
                break
            a, k = viol
            S.remove(a)
            changed = True
        if S and verify(S):
            best = S
        else:
            best = []

    return best
