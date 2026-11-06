import random
import time
import sys

def construct():
    """
    Constructs a set of up to 48 distinct positive integers <= 512.
    No element is the mean of any subset of at least two other elements.
    Uses Greedy 3-AP-free construction + Repair.
    """
    N = 512
    MAX_SIZE = 48
    
    def is_valid_full(S):
        """
        Checks if S satisfies the full condition:
        No element a in S is the mean of any subset T of S\{a} with |T| >= 2.
        Uses DP for subset sums.
        """
        S_list = sorted(S)
        m = len(S_list)
        if m < 3:
            return True
        
        # Precompute subset sums for each possible subset size?
        # No, we need to check for each a.
        # For each a, we check if k*a is a subset sum of S\{a} for any k>=2.
        
        for a in S_list:
            others = [x for x in S_list if x != a]
            m_others = len(others)
            if m_others < 2:
                continue
            
            max_sum = sum(others)
            # dp[k] is a bitset of possible sums with k elements
            dp = [0] * (m_others + 1)
            dp[0] = 1
            
            for x in others:
                for k in range(m_others, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            
            for k in range(2, m_others + 1):
                target = k * a
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return False
        
        return True

    def get_violation(S):
        """
        Returns (a, k, T_sum) or None.
        """
        S_list = sorted(S)
        for a in S_list:
            others = [x for x in S_list if x != a]
            m_others = len(others)
            if m_others < 2:
                continue
            
            max_sum = sum(others)
            dp = [0] * (m_others + 1)
            dp[0] = 1
            
            for x in others:
                for k in range(m_others, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            
            for k in range(2, m_others + 1):
                target = k * a
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return a, k
        
        return None

    start_time = time.time()
    time_limit = 85.0
    
    best_set = set()
    best_size = 0
    
    # Greedy 3-AP-free construction
    def build_3ap_free():
        S = set()
        # Randomize order to find different structures
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(S) >= MAX_SIZE:
                break
            # Check if x creates a 3-AP with any pair in S
            # 1. x is the mean: 2x = a+b
            # 2. a is the mean: 2a = x+b => x = 2a-b
            # 3. b is the mean: 2b = x+a => x = 2b-a
            
            conflict = False
            for a in S:
                # Case 1: 2x = a + b => b = 2x - a
                b1 = 2 * x - a
                if b1 in S and b1 != a:
                    conflict = True
                    break
                
                # Case 2: 2a = x + b => b = 2a - x
                b2 = 2 * a - x
                if b2 in S and b2 != a:
                    conflict = True
                    break
                
                # Case 3: 2b = x + a => b = 2b - a ... wait.
                # 2b = x + a => b = (x+a)/2.
                # This is symmetric to Case 1 with roles of a and x swapped?
                # No, in Case 1, x is the middle.
                # In Case 3, b is the middle.
                # We need to check if there exists b in S such that 2b = x+a.
                # This is equivalent to (x+a) being even and (x+a)//2 in S.
                if (x + a) % 2 == 0:
                    b3 = (x + a) // 2
                    if b3 in S and b3 != a and b3 != x:
                        conflict = True
                        break
            
            if not conflict:
                S.add(x)
        
        return S

    for restart in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        S = build_3ap_free()
        
        # If we didn't get 48, try to add more?
        # The greedy might stop early.
        # But 3-AP-free sets of size 48 exist in 1..512.
        # If len(S) < 48, we can try to fill it with random numbers that don't break 3-AP freeness?
        # Actually, if we just shuffle and greedily add, we should get a maximal 3-AP-free set.
        # The size of a maximal 3-AP-free set in 1..512 is usually > 48.
        
        if len(S) > best_size:
            best_set = S.copy()
            best_size = len(S)
        
        if len(S) >= MAX_SIZE:
            # Check full validity
            if is_valid_full(S):
                return list(S)
        
        # If not valid, repair
        # We can try to repair S
        S_repair = S.copy()
        for step in range(100):
            if time.time() - start_time > time_limit:
                break
            
            if len(S_repair) < MAX_SIZE:
                break # Can't repair by removing if we want to keep size?
                # Actually, we can remove and then try to add back?
                # For now, just remove if invalid.
            
            viol = get_violation(S_repair)
            if viol is None:
                # Valid
                if len(S_repair) > best_size:
                    best_set = S_repair.copy()
                    best_size = len(S_repair)
                if len(S_repair) == MAX_SIZE:
                    return list(S_repair)
                break
            
            a, k = viol
            # Remove a
            S_repair.remove(a)
            
            # Try to add a new element to replace a?
            # Optional. For now, just remove.

    # Return best found
    if best_set:
        if is_valid_full(best_set):
            return list(best_set)
    
    # Fallback: Return a small valid set
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
