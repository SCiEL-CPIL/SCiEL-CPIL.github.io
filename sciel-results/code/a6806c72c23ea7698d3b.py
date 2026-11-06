import random
import time
import sys

def construct():
    """
    Constructs a set of up to 48 distinct positive integers <= 512.
    No element is the mean of any subset of at least two other elements.
    Uses Greedy Initialization followed by Local Search with Repair.
    """
    N = 512
    MAX_SIZE = 48
    
    # Fast DP verification
    def is_valid_and_count(S):
        """
        Returns (is_valid, violation_count).
        Uses integer bitset DP with early termination.
        """
        S_list = list(S)
        m = len(S_list)
        if m < 3:
            return True, 0
        
        for a in S_list:
            others = [x for x in S_list if x != a]
            m_others = len(others)
            if m_others < 2:
                continue
            
            # Quick check: k=2 (arithmetic mean of 2 elements)
            # a is mean of x,y iff x+y = 2a
            others_set = set(others)
            for x in others:
                y = 2 * a - x
                if y != x and y in others_set:
                    return False, 1
            
            # DP for others
            max_sum = sum(others)
            dp = [0] * (m_others + 1)
            dp[0] = 1
            
            for x in others:
                for k in range(m_others, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            
            # Check violations for a
            for k in range(3, m_others + 1):
                target = k * a
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return False, 1
        
        return True, 0

    def get_violation(S):
        """
        Returns (a, T) if a is the mean of subset T of S\{a}, else (None, None).
        T is a list of elements (or None if found via DP for k>=4).
        """
        S_list = list(S)
        for a in S_list:
            others = [x for x in S_list if x != a]
            m_others = len(others)
            if m_others < 2:
                continue
            
            others_set = set(others)
            
            # Check k=2
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    if others[i] + others[j] == 2 * a:
                        return a, [others[i], others[j]]
            
            # Check k=3 using set lookup
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    rem = 3 * a - others[i] - others[j]
                    if rem > others[j] and rem in others_set:
                        if rem != others[i] and rem != others[j]:
                            return a, [others[i], others[j], rem]
            
            # Fallback: Use DP to check existence for k>=4
            max_sum = sum(others)
            dp = [0] * (m_others + 1)
            dp[0] = 1
            for x in others:
                for k in range(m_others, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            
            for k in range(4, m_others + 1):
                target = k * a
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return a, None
        
        return None, None

    def greedy_construct(order):
        """
        Builds a set by adding elements in the given order if valid.
        """
        S = set()
        for x in order:
            if len(S) >= MAX_SIZE:
                break
            S_trial = S.copy()
            S_trial.add(x)
            is_valid, _ = is_valid_and_count(S_trial)
            if is_valid:
                S = S_trial
        return S

    start_time = time.time()
    time_limit = 80.0 # Safety margin (90s kill, 10s buffer)
    
    candidates = list(range(1, N + 1))
    S_set = set(candidates)
    
    best_set = set()
    best_size = 0
    
    def update_best(s):
        nonlocal best_set, best_size
        if len(s) > best_size:
            best_set = s.copy()
            best_size = len(s)
    
    # Phase 1: Greedy Initialization with diverse orders
    # 1. Descending
    S_desc = greedy_construct(range(N, 0, -1))
    update_best(S_desc)
        
    # 2. Ascending
    S_asc = greedy_construct(range(1, N + 1))
    update_best(S_asc)
    
    # 3. Even numbers first, then odd
    evens = list(range(2, N + 1, 2))
    odds = list(range(1, N + 1, 2))
    S_eo = greedy_construct(evens + odds)
    update_best(S_eo)
    
    # 4. Odd first, then even
    S_oe = greedy_construct(odds + evens)
    update_best(S_oe)
    
    # 5. Modular ordering: by remainder mod 4
    mod_orders = []
    for r in [3, 1, 2, 0]:  # Try different mod orderings
        mod_orders.append([x for x in range(1, N+1) if x % 4 == r])
    S_mod = greedy_construct([x for group in mod_orders for x in group])
    update_best(S_mod)
    
    # 6. Random Orders
    for _ in range(20):
        if time.time() - start_time > time_limit:
            break
        random.shuffle(candidates)
        S_rand = greedy_construct(candidates)
        update_best(S_rand)
            
    # Phase 2: Local Search / Repair / Expansion
    S = best_set.copy()
    
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
        
        # Check if current S is valid
        is_valid, _ = is_valid_and_count(S)
        
        if not is_valid:
            # Repair S
            for step in range(500):
                if time.time() - start_time > time_limit:
                    break
                is_valid, _ = is_valid_and_count(S)
                if is_valid:
                    break
                a, T = get_violation(S)
                if a is None:
                    break
                r = random.random()
                if T is not None and len(T) > 0 and r < 0.5:
                    victim = random.choice(T)
                    S.remove(victim)
                elif r < 0.8:
                    S.remove(a)
                else:
                    others = [x for x in S if x != a]
                    if others:
                        S.remove(random.choice(others))
                    else:
                        S.remove(a)
        
        if len(S) > best_size:
            best_set = S.copy()
            best_size = len(S)
            
        if len(S) == MAX_SIZE:
            is_valid, _ = is_valid_and_count(S)
            if is_valid:
                return list(S)
        
        # Try to expand/swap to reach MAX_SIZE
        for step in range(300):
            if len(S) >= MAX_SIZE or time.time() - start_time > time_limit:
                break
            
            remaining = [x for x in candidates if x not in S]
            if not remaining:
                break
                
            # Pick a random candidate to add
            c = random.choice(remaining)
            S_trial = S.copy()
            S_trial.add(c)
            
            is_valid_trial, _ = is_valid_and_count(S_trial)
            
            if is_valid_trial:
                S = S_trial
                if len(S) > best_size:
                    best_set = S.copy()
                    best_size = len(S)
                if len(S) == MAX_SIZE:
                    is_valid_final, _ = is_valid_and_count(S)
                    if is_valid_final:
                        return list(S)
            else:
                # Try swap
                a, T = get_violation(S_trial)
                if a is not None:
                    candidates_to_remove = []
                    if T:
                        candidates_to_remove.extend([x for x in T if x in S])
                    if a in S and a != c:
                        candidates_to_remove.append(a)
                    
                    if candidates_to_remove:
                        victim = random.choice(candidates_to_remove)
                        S_trial.remove(victim)
                        is_valid_swap, _ = is_valid_and_count(S_trial)
                        if is_valid_swap:
                            if random.random() < 0.4:
                                S = S_trial
                                if len(S) > best_size:
                                    best_set = S.copy()
                                    best_size = len(S)

    # Return the best valid set found
    if best_set:
        is_valid, _ = is_valid_and_count(best_set)
        if is_valid:
            return list(best_set)
    
    # Fallback
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
