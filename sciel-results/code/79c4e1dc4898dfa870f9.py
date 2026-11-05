"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random
import sys

N = 256
MAX_SUM = 2 * N

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 28.0  # Safety margin
    
    # Bitset for sums: index is sum value, value is count or boolean
    # We only need to know if a sum is present.
    
    def get_sums_array(S):
        """Returns a boolean array of size MAX_SUM+1 where arr[s] is True if s is in sumset(S)."""
        sums = [False] * (MAX_SUM + 1)
        n = len(S)
        for i in range(n):
            ai = S[i]
            sums[2 * ai] = True
            for j in range(i + 1, n):
                sums[ai + S[j]] = True
        return sums

    def can_add_fast(S, sums_arr, x):
        """Check if x can be added to S given its sum array."""
        if sums_arr[2 * x]:
            return False
        for a in S:
            if sums_arr[x + a]:
                return False
        return True

    def is_maximal_fast(S, sums_arr):
        """Check if S is inclusion-maximal."""
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if can_add_fast(S, sums_arr, x):
                return False
        return True

    def greedy_construct(order):
        """Build a Sidon set by greedily adding elements in given order."""
        S = []
        sums_arr = [False] * (MAX_SUM + 1)
        for x in order:
            if can_add_fast(S, sums_arr, x):
                S.append(x)
                sums_arr[2 * x] = True
                for a in S[:-1]:
                    sums_arr[x + a] = True
        return S, sums_arr

    def prune_and_swap(S, sums_arr, time_limit):
        """Iteratively remove elements if possible, or swap to maintain maximality with smaller/equal size."""
        S = list(S)
        S.sort()
        sums_arr = get_sums_array(S)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            n = len(S)
            
            # 1. Try to remove elements
            for i in range(n - 1, -1, -1):
                if time.time() > time_limit:
                    return S, get_sums_array(S)
                u = S[i]
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                new_sums = get_sums_array(new_S)
                if is_maximal_fast(new_S, new_sums):
                    S = new_S
                    sums_arr = new_sums
                    improved = True
                    # Restart loop as indices changed
                    break
            
            if improved:
                continue
                
            # 2. Try swaps: replace u with v
            # We only care if we can maintain maximality. 
            # Ideally we want to find a swap that allows future removals, 
            # but for now, any swap that keeps it maximal and potentially leads to smaller set is good.
            # Actually, if we just swap, size stays same. 
            # The power is in the subsequent removals.
            
            for i in range(len(S)):
                if time.time() > time_limit:
                    break
                u = S[i]
                S_without_u = S[:i] + S[i+1:]
                sums_without_u = get_sums_array(S_without_u)
                
                # Find v that can be added to S_without_u
                # And check if resulting set is maximal
                for v in range(1, N + 1):
                    if v == u:
                        continue
                    if v in S_without_u:
                        continue
                    if can_add_fast(S_without_u, sums_without_u, v):
                        new_S = S_without_u + [v]
                        new_S.sort()
                        new_sums = get_sums_array(new_S)
                        if is_maximal_fast(new_S, new_sums):
                            # Perform swap
                            S = new_S
                            sums_arr = new_sums
                            improved = True
                            break
                if improved:
                    break
        return S, sums_arr

    best_size = N + 1
    best_set = None

    # Phase 1: Greedy baselines
    # Ascending
    S0, sums0 = greedy_construct(range(1, N + 1))
    S0, _ = prune_and_swap(S0, sums0, deadline - 5)
    if len(S0) < best_size:
        best_size = len(S0)
        best_set = list(S0)

    # Descending
    S1, sums1 = greedy_construct(range(N, 0, -1))
    S1, _ = prune_and_swap(S1, sums1, deadline - 5)
    if len(S1) < best_size:
        best_size = len(S1)
        best_set = list(S1)

    # Phase 2: Randomized greedy with aggressive pruning/swapping
    rng = random.Random(42)
    
    # We have ~20s left.
    # Each trial: Greedy (fast) + Prune/Swap (slower).
    # We need to balance.
    
    for trial in range(10000):
        if time.time() > deadline - 2:
            break
        
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        S, sums = greedy_construct(order)
        S.sort()
        
        # Prune and Swap
        # Limit time per trial to ensure we finish
        trial_limit = min(deadline - 2, time.time() + 0.05) # 50ms per trial approx
        S, _ = prune_and_swap(S, sums, trial_limit)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)

    # Phase 3: Final polish on best_set
    if best_set is not None and time.time() < deadline - 1:
        S_best, sums_best = prune_and_swap(best_set, get_sums_array(best_set), deadline - 0.5)
        if len(S_best) < best_size:
            best_size = len(S_best)
            best_set = list(S_best)

    # Fallback
    if best_set is None:
        S0, _ = greedy_construct(range(1, N + 1))
        best_set = S0

    # Final verification
    S = sorted(best_set)
    sums = get_sums_array(S)
    
    # Check Sidon property
    seen = set()
    valid = True
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in seen:
                valid = False
                break
            seen.add(s)
    if not valid:
        # Fallback to a known valid one (re-run greedy and prune minimally)
        S0, _ = greedy_construct(range(1, N + 1))
        S0, _ = prune_and_swap(S0, get_sums_array(S0), deadline - 0.1)
        return S0

    # Check maximality
    if not is_maximal_fast(S, sums):
        # Repair: add missing blockers
        S_set = set(S)
        sums_arr = sums
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if can_add_fast(S, sums_arr, x):
                    S.append(x)
                    S_set.add(x)
                    sums_arr[2 * x] = True
                    for a in S[:-1]:
                        sums_arr[x + a] = True
                    changed = True
                    break
        S = sorted(S)
        best_set = S

    return best_set
