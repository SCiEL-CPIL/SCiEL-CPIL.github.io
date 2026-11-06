import time
import random

def construct():
    """
    Constructs a set of up to 48 distinct positive integers <= 512.
    No element is the mean of any subset of at least two other elements.
    Uses a two-phase approach:
    1. Greedy construction to build a large 3-AP-free set (satisfies k=2 condition).
    2. Local search to repair violations for k>=3 by swapping elements.
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
        Returns (a, k) if there's a violation, else None.
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

    def build_3ap_free():
        """
        Greedily constructs a 3-AP-free set.
        """
        S = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(S) >= MAX_SIZE:
                break
            
            conflict = False
            for a in S:
                # 1. x is middle: 2x = a + b => b = 2x - a
                b1 = 2 * x - a
                if b1 in S and b1 != a:
                    conflict = True
                    break
                
                # 2. a is middle: 2a = x + b => b = 2a - x
                b2 = 2 * a - x
                if b2 in S and b2 != a:
                    conflict = True
                    break
                
                # 3. b is middle: 2b = x + a => b = (x+a)/2
                if (x + a) % 2 == 0:
                    b3 = (x + a) // 2
                    if b3 in S and b3 != a and b3 != x:
                        conflict = True
                        break
            
            if not conflict:
                S.add(x)
        
        return S

    def local_search(S, time_limit):
        """
        Attempts to repair S by swapping elements to fix k>=3 violations.
        Returns a valid set or the best set found.
        """
        S = S.copy()
        start = time.time()
        
        while time.time() - start < time_limit:
            viol = get_violation(S)
            if viol is None:
                return S
            
            a, k = viol
            # Try to replace a with some other element
            removed = False
            for y in range(1, N + 1):
                if y in S:
                    continue
                
                S.add(y)
                S.remove(a)
                
                if is_valid_full(S):
                    return S
                else:
                    S.add(a)
                    S.remove(y)
            
            # If no swap worked, remove a and try to add a new element
            S.remove(a)
            for y in range(1, N + 1):
                if y in S:
                    continue
                S.add(y)
                if is_valid_full(S):
                    # We lost one, but maybe we can regain?
                    # For now, just return if we maintain size or improve validity
                    if len(S) >= MAX_SIZE:
                        return S
                else:
                    S.remove(y)
            
            # If we can't fix it, just remove the violator to ensure validity
            # and try to fill up again? This is complex. 
            # Simpler: if we can't fix, return the set without the violator?
            # But we want max size.
            # Let's just break and return current best if we can't improve.
            # Actually, the loop above tries to swap. If it fails, we removed 'a'.
            # Now we try to add 'y' back. If none work, we are left with size-1.
            # This is a dead end. We should probably restart or accept the loss.
            # Given time constraints, let's just return the set if it's valid (it is, after removing a)
            # and hope the caller restarts.
            if is_valid_full(S):
                return S
            
            # If even after removing 'a' it's invalid (shouldn't happen if 'a' was the only violator? 
            # No, removing an element can't create a new violation involving that element, 
            # but it might leave other elements violating? No, removing elements only reduces subset sums.
            # So S is valid after removing a.
            return S

    start_time = time.time()
    time_limit = 85.0
    
    best_set = set()
    best_size = 0
    
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
        
        S = build_3ap_free()
        
        if len(S) > best_size:
            best_set = S.copy()
            best_size = len(S)
        
        if len(S) >= MAX_SIZE:
            if is_valid_full(S):
                return list(S)
            
            # Try to repair
            S_repaired = local_search(S, 0.5)
            
            if is_valid_full(S_repaired):
                if len(S_repaired) > best_size:
                    best_set = S_repaired.copy()
                    best_size = len(S_repaired)
                
                if len(S_repaired) >= MAX_SIZE:
                    return list(S_repaired)
    
    # Return best found
    if best_set and is_valid_full(best_set):
        return list(best_set)
    
    # Fallback: Return a small valid set
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
