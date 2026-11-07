import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_val = float('inf')
    best_list = None
    
    # Pre-check for very small max (score < 21)
    # Max <= 230
    # Integers in [211, 230]
    small_nums = list(range(211, 231))
    
    # Check k=1 to k=5
    # k=1
    for x in small_nums:
        P = x * 210
        s = math.isqrt(P)
        if s * s == P:
            val = x - 210
            if val < best_val:
                best_val = val
                best_list = [x]
                
    # k=2
    for i in range(len(small_nums)):
        for j in range(i+1, len(small_nums)):
            P = small_nums[i] * small_nums[j] * 210
            s = math.isqrt(P)
            if s * s == P:
                val = small_nums[j] - 210
                if val < best_val:
                    best_val = val
                    best_list = [small_nums[i], small_nums[j]]
                    
    # k=3
    for combo in combinations(small_nums, 3):
        P = 210
        for x in combo:
            P *= x
        s = math.isqrt(P)
        if s * s == P:
            val = combo[-1] - 210
            if val < best_val:
                best_val = val
                best_list = list(combo)
                
    # k=4
    for combo in combinations(small_nums, 4):
        P = 210
        for x in combo:
            P *= x
        s = math.isqrt(P)
        if s * s == P:
            val = combo[-1] - 210
            if val < best_val:
                best_val = val
                best_list = list(combo)
                
    # k=5
    for combo in combinations(small_nums, 5):
        P = 210
        for x in combo:
            P *= x
        s = math.isqrt(P)
        if s * s == P:
            val = combo[-1] - 210
            if val < best_val:
                best_val = val
                best_list = list(combo)

    # If we found a solution with max <= 230, we can return it immediately 
    # as it is better than any solution with max >= 231 (score >= 21)
    if best_val < 21:
        return best_list

    # General search for k=1
    x1 = 210 * 4
    if 210 < x1 <= 10210:
        val = x1 - 210
        if val < best_val:
            best_val = val
            best_list = [x1]

    # General search for k=2
    m_max = int(math.sqrt(10210**2 / 210))
    
    for m in range(15, m_max + 1):
        if time.time() > deadline:
            break
        P = 210 * m * m
        limit = int(math.isqrt(P))
        if limit < 211:
            continue
            
        found = False
        for d in range(limit, 210, -1):
            if P % d == 0:
                x2 = P // d
                if x2 > 210 and x2 != d:
                    val = x2 - 210
                    if val < best_val:
                        best_val = val
                        best_list = [d, x2]
                    found = True
                    break

    # General search for k=3 (only if we haven't found something very good)
    if best_val > 21:
        nums = list(range(211, 240))
        n = len(nums)
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                for k in range(j+1, n):
                    P = nums[i] * nums[j] * nums[k]
                    if P % 210 != 0:
                        continue
                    Q = P // 210
                    s = int(math.isqrt(Q))
                    if s * s == Q:
                        val = nums[k] - 210
                        if val < best_val:
                            best_val = val
                            best_list = [nums[i], nums[j], nums[k]]

    if best_list is None:
        return [840]
        
    return best_list
