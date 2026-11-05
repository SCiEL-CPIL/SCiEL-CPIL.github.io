import time
import random
import numpy as np

def _subset_sum_bitset(elements, k, target):
    """Check if subset of exactly k elements sums to target using bitset DP."""
    n = len(elements)
    if k < 0 or k > n:
        return False
    if k == 0:
        return target == 0
    if target < 0:
        return False
    
    # Use complement if beneficial
    if n - k < k:
        total = sum(elements)
        if target > total:
            return False
        return _subset_sum_bitset(elements, n - k, total - target)
    
    if k == 1:
        return target in elements
    if k == 2:
        seen = set()
        for x in elements:
            if (target - x) in seen:
                return True
            seen.add(x)
        return False
    
    # Bitset DP: dp[j] is a bitset where bit s is 1 if sum s achievable with j elements
    max_sum = target
    dp = [0] * (k + 1)
    dp[0] = 1  # bit 0 set
    
    for e in elements:
        for j in range(k, 0, -1):
            shifted = (dp[j-1] << e)
            mask = (1 << (max_sum + 1)) - 1
            dp[j] |= (shifted & mask)
    
    return (dp[k] >> target) & 1 == 1

def _is_valid_set(S):
    """Check if set S is valid: no element is mean of any subset of >=2 other elements."""
    S = list(S)
    n = len(S)
    for i, x in enumerate(S):
        others = S[:i] + S[i+1:]
        m = len(others)
        for k in range(2, m + 1):
            if _subset_sum_bitset(others, k, k * x):
                return False
    return True

def _can_add(S, c):
    """Check if c can be added to S while maintaining validity."""
    S = list(S)
    n = len(S)
    
    # Condition 1: c is not mean of any k-subset of S (k >= 2)
    for k in range(2, n + 1):
        if _subset_sum_bitset(S, k, k * c):
            return False
    
    # Condition 2: no x in S becomes mean of subset including c
    for i, x in enumerate(S):
        others = S[:i] + S[i+1:]  # S without x
        m = len(others)
        # Need subset T of others with |T| = j >= 1 such that sum(T) + c = (j+1)*x
        # i.e., sum(T) = (j+1)*x - c, for j in 1..m
        for j in range(1, m + 1):
            target = (j + 1) * x - c
            if target < 0:
                continue
            if _subset_sum_bitset(others, j, target):
                return False
    
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 82.0  # Safety margin before 90s kill
    
    # Start with powers of 2 as a safe base
    base = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    best = base[:]
    best_sz = len(best)
    
    # Generate a structured candidate pool
    # Include powers of 2, multiples, and random fill
    candidates = set(range(1, 513))
    
    # Try multiple greedy orderings
    orderings = []
    
    # Ordering 1: ascending
    orderings.append(list(range(1, 513)))
    
    # Ordering 2: descending
    orderings.append(list(range(512, 0, -1)))
    
    # Ordering 3: random
    for _ in range(5):
        r = list(range(1, 513))
        random.shuffle(r)
        orderings.append(r)
    
    # Ordering 4: prioritize numbers coprime to many others
    def coprime_score(x):
        count = 0
        for y in range(1, 513):
            if y != x:
                import math
                if math.gcd(x, y) == 1:
                    count += 1
        return -count  # Higher coprime count = lower score (try first)
    
    orderings.append(sorted(range(1, 513), key= coprime_score))
    
    # Try each ordering
    for trial, ordering in enumerate(orderings):
        if time.time() > deadline - 15:
            break
        
        cur = base[:]
        cur_set = set(cur)
        
        for c in ordering:
            if c in cur_set:
                continue
            if time.time() > deadline - 10:
                break
            
            if _can_add(cur, c):
                cur.append(c)
                cur_set.add(c)
        
        if len(cur) > best_sz:
            best_sz = len(cur)
            best = cur[:]
            if best_sz >= 48:
                break
    
    # Local search: try to improve by swapping
    if best_sz < 48 and time.time() < deadline - 10:
        cur = best[:]
        cur_set = set(cur)
        improved = True
        
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try removing one element and adding two
            for i in range(len(cur)):
                if time.time() > deadline - 5:
                    break
                
                x = cur[i]
                rest = cur[:i] + cur[i+1:]
                rest_set = set(rest)
                
                # Try to find two elements to add
                found_pair = False
                for a in range(1, 513):
                    if found_pair or a in rest_set:
                        continue
                    if time.time() > deadline - 5:
                        break
                    
                    if _can_add(rest, a):
                        new_set = rest + [a]
                        new_set_s = rest_set | {a}
                        
                        for b in range(1, 513):
                            if b in new_set_s:
                                continue
                            if _can_add(new_set, b):
                                new_set.append(b)
                                if len(new_set) > len(cur):
                                    cur = new_set
                                    cur_set = set(cur)
                                    best = cur[:]
                                    best_sz = len(cur)
                                    improved = True
                                    found_pair = True
                                    if best_sz >= 48:
                                        break
                        if best_sz >= 48:
                            break
                
                if best_sz >= 48:
                    break
            
            # Try removing two elements and adding three
            if best_sz < 48 and time.time() < deadline - 5:
                for i in range(len(cur)):
                    if time.time() > deadline - 5:
                        break
                    for j in range(i+1, len(cur)):
                        if time.time() > deadline - 5:
                            break
                        
                        x1, x2 = cur[i], cur[j]
                        rest = [cur[k] for k in range(len(cur)) if k != i and k != j]
                        rest_set = set(rest)
                        
                        found_triple = False
                        for a in range(1, 513):
                            if found_triple or a in rest_set:
                                continue
                            if time.time() > deadline - 5:
                                break
                            
                            if _can_add(rest, a):
                                new_set = rest + [a]
                                new_set_s = rest_set | {a}
                                
                                for b in range(1, 513):
                                    if found_triple or b in new_set_s:
                                        continue
                                    if _can_add(new_set, b):
                                        new_set2 = new_set + [b]
                                        new_set_s2 = new_set_s | {b}
                                        
                                        for c in range(1, 513):
                                            if c in new_set_s2:
                                                continue
                                            if _can_add(new_set2, c):
                                                new_set2.append(c)
                                                if len(new_set2) > len(cur):
                                                    cur = new_set2
                                                    cur_set = set(cur)
                                                    best = cur[:]
                                                    best_sz = len(cur)
                                                    improved = True
                                                    found_triple = True
                                                    break
                                        if found_triple:
                                            break
                                    if found_triple:
                                        break
                            if found_triple:
                                break
                        
                        if found_triple:
                            break
                    if best_sz >= 48:
                        break
    
    return best[:48]
