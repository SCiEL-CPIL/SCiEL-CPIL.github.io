import random
import time
import sys

def construct():
    """
    Constructs a set of up to 48 distinct positive integers <= 512.
    No element is the mean of any subset of at least two other elements.
    Uses incremental DP + local search to maximize size.
    """
    N = 512
    MAX_SIZE = 48
    MAX_SUM = N * MAX_SIZE  # 24576
    
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def check_validity_with_dp(S_list):
        """Full DP validity check. Returns True if valid."""
        m = len(S_list)
        if m < 3:
            return True
        for a in S_list:
            others = [x for x in S_list if x != a]
            m_others = len(others)
            if m_others < 2:
                continue
            dp = [0] * (m_others + 1)
            dp[0] = 1
            for x in others:
                for k in range(m_others, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            for k in range(2, m_others + 1):
                target = k * a
                if target > MAX_SUM:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True
    
    def find_violation(S_list):
        """Find one violation. Returns (a, subset_list) or (None, None)."""
        for a in S_list:
            others = [x for x in S_list if x != a]
            m_others = len(others)
            if m_others < 2:
                continue
            # Check k=2
            for i in range(m_others):
                for j in range(i+1, m_others):
                    if others[i] + others[j] == 2 * a:
                        return a, [others[i], others[j]]
            # Check k=3
            others_set = set(others)
            for i in range(m_others):
                for j in range(i+1, m_others):
                    rem = 3 * a - others[i] - others[j]
                    if rem > others[j] and rem in others_set and rem != others[i] and rem != others[j]:
                        return a, [others[i], others[j], rem]
            # Check k>=4 with DP
            dp = [0] * (m_others + 1)
            dp[0] = 1
            for x in others:
                for k in range(m_others, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            for k in range(4, m_others + 1):
                target = k * a
                if target > MAX_SUM:
                    break
                if (dp[k] >> target) & 1:
                    return a, None
        return None, None
    
    def incremental_greedy(order):
        """
        Greedily build a valid set by adding elements in 'order'.
        Uses incremental DP for speed.
        Returns the set built.
        """
        selected = []
        n = 0
        
        # dp_all[k]: bitset of sums achievable with k elements from selected
        dp_all = [1]  # dp_all[0] = 1 (sum 0)
        
        # For each element a in selected, dp_without[a][k]: bitset of sums with k elements from selected\{a}
        dp_without = {}
        
        for x in order:
            if time.time() > deadline:
                break
            if n >= MAX_SIZE:
                break
            
            # Check 1: Is x the mean of any subset of selected (size >= 2)?
            # Need: exists k >= 2, subset of selected with sum = k*x
            valid = True
            for k in range(2, n + 1):
                target = k * x
                if target > MAX_SUM:
                    break
                if k < len(dp_all) and (dp_all[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: For each a in selected, is a the mean of a subset involving x?
            # a is mean of subset T where x in T, T subset of selected ∪ {x}, |T| >= 2
            # sum(T) = |T| * a, x in T, T \ {x} subset of selected \ {a}
            # Let U = T \ {x}, then sum(U) = |T|*a - x = (|U|+1)*a - x
            # U subset of selected \ {a}, |U| >= 1 (since |T| >= 2 and x in T)
            for a in selected:
                if not valid:
                    break
                if a not in dp_without:
                    continue
                dp_w = dp_without[a]
                for k in range(1, n):  # k = |U|, U has k elements from selected\{a}
                    target = (k + 1) * a - x
                    if target < 0 or target > MAX_SUM:
                        continue
                    if k < len(dp_w) and (dp_w[k] >> target) & 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Add x to selected
            selected.append(x)
            
            # Update dp_all
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = dp_all[k] if k < len(dp_all) else 0
                if k - 1 >= 0 and k - 1 < len(dp_all):
                    val |= (dp_all[k-1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all
            
            # Update dp_without for existing elements
            new_dp_without = {}
            for a in selected[:-1]:
                if a not in dp_without:
                    continue
                old_dp_w = dp_without[a]
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = old_dp_w[k] if k < len(old_dp_w) else 0
                    if k - 1 >= 0 and k - 1 < len(old_dp_w):
                        val |= (old_dp_w[k-1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list
            
            # For new element x, dp_without[x] is the old dp_all (selected before adding x)
            new_dp_without[x] = dp_all[:]
            
            dp_without = new_dp_without
            n += 1
        
        return selected
    
    def local_search_improve(S, max_iters=200):
        """
        Try to improve set S by adding elements or swapping.
        Uses full DP check for simplicity (slower but correct).
        """
        S = set(S)
        all_vals = set(range(1, N + 1))
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add an element
            added = False
            candidates_to_add = [x for x in all_vals if x not in S]
            random.shuffle(candidates_to_add)
            for x in candidates_to_add[:30]:
                new_S = list(S | {x})
                if check_validity_with_dp(new_S):
                    S = S | {x}
                    added = True
                    break
            
            if added:
                continue
            
            # Try a swap
            if S:
                a = random.choice(list(S))
                candidates_to_swap = [x for x in all_vals if x not in S]
                random.shuffle(candidates_to_swap)
                for x in candidates_to_swap[:30]:
                    new_S = list((S - {a}) | {x})
                    if check_validity_with_dp(new_S):
                        S = (S - {a}) | {x}
                        break
        
        return S
    
    def repair_set(S, max_steps=500):
        """Repair an invalid set by removing elements causing violations."""
        S = set(S)
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            
            S_list = list(S)
            if len(S_list) < 3:
                break
            
            a, T = find_violation(S_list)
            if a is None:
                break  # Valid
            
            # Remove an element to fix the violation
            r = random.random()
            if T is not None and len(T) > 0 and r < 0.5:
                victim = random.choice(T)
                S.discard(victim)
            elif r < 0.8:
                S.discard(a)
            else:
                others = [x for x in S if x != a]
                if others:
                    S.discard(random.choice(others))
                else:
                    S.discard(a)
        
        return S
    
    best_set = set()
    best_size = 0
    
    candidates = list(range(1, N + 1))
    
    # Multiple strategies with different orderings
    strategies = []
    
    # Strategy 1: Random order
    for _ in range(3):
        order = candidates[:]
        random.shuffle(order)
        strategies.append(order)
    
    # Strategy 2: Large-first (fewer small numbers that can be means)
    order_large_first = list(range(N, 0, -1))
    strategies.append(order_large_first)
    
    # Strategy 3: Small-first (dense at low end)
    order_small_first = list(range(1, N + 1))
    strategies.append(order_small_first)
    
    # Strategy 4: Mixed - alternate large and small
    order_mixed = []
    for i in range(N // 2):
        order_mixed.append(N - i)
        order_mixed.append(i + 1)
    strategies.append(order_mixed)
    
    # Strategy 5: Modular pattern (avoid arithmetic progressions)
    # Prefer numbers that are powers of 2 or near powers
    order_modular = []
    for i in range(10):
        order_modular.append(2 ** i)
    for i in range(1, N + 1):
        if i not in order_modular:
            order_modular.append(i)
    random.shuffle(order_modular[10:])  # Keep powers at front, shuffle rest
    strategies.append(order_modular)
    
    for order in strategies:
        if time.time() > deadline:
            break
        
        # Incremental greedy
        S = incremental_greedy(order)
        S = set(S)
        
        # Verify
        if S:
            S_list = list(S)
            if check_validity_with_dp(S_list):
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
        
        # Local search to improve
        if len(S) < MAX_SIZE:
            S = local_search_improve(S, max_iters=100)
            S = set(S)
            if S:
                S_list = list(S)
                if check_validity_with_dp(S_list):
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S.copy()
        
        if best_size == MAX_SIZE:
            break
    
    # If we have a good set, try more local search
    if best_set and len(best_set) < MAX_SIZE:
        best_set = local_search_improve(best_set, max_iters=300)
        best_set = set(best_set)
        if best_set:
            S_list = list(best_set)
            if check_validity_with_dp(S_list):
                best_size = len(best_set)
            else:
                # Shouldn't happen, but safety
                best_set = set()
                best_size = 0
    
    # If still not at MAX_SIZE, try repair-based approach
    if best_size < MAX_SIZE and time.time() < deadline - 5:
        for restart in range(10):
            if time.time() > deadline:
                break
            
            # Start with random set of size MAX_SIZE
            S = set(random.sample(candidates, MAX_SIZE))
            
            # Repair
            S = repair_set(S, max_steps=300)
            
            if S:
                S_list = list(S)
                if check_validity_with_dp(S_list):
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S.copy()
                    
                    # Try to add more
                    if len(S) < MAX_SIZE:
                        S = local_search_improve(S, max_iters=100)
                        S = set(S)
                        if S:
                            S_list = list(S)
                            if check_validity_with_dp(S_list):
                                if len(S) > best_size:
                                    best_size = len(S)
                                    best_set = S.copy()
            
            if best_size == MAX_SIZE:
                break
    
    # Final verification
    if best_set:
        S_list = list(best_set)
        if check_validity_with_dp(S_list):
            return list(best_set)
    
    # Fallback: small valid set
    return [1, 3, 7, 15, 31, 63, 127, 255, 511]

# EVOLVE-BLOCK-END
