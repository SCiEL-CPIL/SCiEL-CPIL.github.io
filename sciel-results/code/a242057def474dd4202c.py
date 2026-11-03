import itertools
import math

def construct():
    best_score = 0
    best_set = [0, 1]
    
    def calc_score(u):
        if not u:
            return 0
        max_u = u[-1]
        if max_u == 0:
            return 0
            
        sum_mask = 0
        diff_mask = 0
        
        for i in u:
            for j in u:
                sum_mask |= (1 << (i + j))
                diff_mask |= (1 << (i - j + max_u))
                
        num_sums = bin(sum_mask).count('1')
        num_diffs = bin(diff_mask).count('1')
        
        if num_sums == 0:
            return 0
            
        ratio = num_diffs / num_sums
        if ratio <= 1:
            return 0
            
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            return 0
            
        return 1 + math.log2(ratio) / denom

    # Search small M thoroughly
    # M up to 60
    # n up to 8
    for M in range(1, 61):
        # Max n such that C(M, n-1) is manageable
        # C(60, 5) is 5.4M. Too slow.
        # C(30, 5) is 142k. OK.
        # C(20, 6) is 38k. OK.
        # C(10, 5) is 252. OK.
        
        max_n = 8
        if M > 20:
            max_n = 6
        if M > 30:
            max_n = 5
        if M > 40:
            max_n = 4
        if M > 50:
            max_n = 3
            
        for n in range(2, max_n + 1):
            if n - 1 > M:
                continue
            for comb in itertools.combinations(range(1, M + 1), n - 1):
                u = [0] + list(comb)
                u.sort()
                s = calc_score(u)
                if s > best_score:
                    best_score = s
                    best_set = u

    return best_set
