import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_val = float('inf')
    best_list = None
    
    # Check k=1
    x1 = 210 * 4
    if 210 < x1 <= 10210:
        val = x1 - 210
        if val < best_val:
            best_val = val
            best_list = [x1]
    
    # Check k=2
    m_max = int(math.isqrt(10210**2 // 210))
    
    for m in range(15, m_max + 1):
        if time.time() > deadline:
            break
        P = 210 * m * m
        limit = int(math.isqrt(P))
        if limit < 211:
            continue
        
        for d in range(limit, 210, -1):
            if P % d == 0:
                x2 = P // d
                if x2 > 210 and x2 != d:
                    val = x2 - 210
                    if val < best_val:
                        best_val = val
                        best_list = [d, x2]
                    break
    
    # Check k=3: For each M, find s such that a*b*M = 210*s^2
    # i.e., a*b = 210*s^2/M must be integer, and we need factors a,b with 211<=a<b<M
    for M in range(211, 10211):
        if time.time() > deadline:
            break
        if M - 210 >= best_val:
            break
        
        # We need a*b*M = 210*s^2 for some s
        # So a*b = 210*s^2 / M
        # s^2 must make 210*s^2 divisible by M
        # Let's iterate over possible s values
        # a*b*M = 210*s^2, so s^2 = a*b*M/210
        # Since a,b < M, a*b*M < M^3, so s < M*sqrt(M/210)
        # Also a*b >= 211*212, so s^2 >= 211*212*M/210, s >= sqrt(211*212*M/210)
        
        min_s_sq = 211 * 212 * M // 210
        max_s_sq = (M - 1) * (M - 2) * M // 210
        
        min_s = max(1, math.isqrt(min_s_sq))
        if min_s * min_s < min_s_sq:
            min_s += 1
        max_s = math.isqrt(max_s_sq)
        
        for s in range(min_s, max_s + 1):
            if time.time() > deadline:
                break
            num = 210 * s * s
            if num % M != 0:
                continue
            ab = num // M
            # Find factor pair a*b = ab with 211 <= a < b < M
            # a must be a divisor of ab
            limit_a = int(math.isqrt(ab))
            found = False
            for a in range(211, limit_a + 1):
                if ab % a == 0:
                    b = ab // a
                    if b > a and b < M:
                        val = M - 210
                        if val < best_val:
                            best_val = val
                            best_list = [a, b, M]
                        found = True
                        break
            if found:
                break
    
    if best_list is None:
        return [840]
        
    return best_list
