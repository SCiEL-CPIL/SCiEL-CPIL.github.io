import math

def construct():
    best_score = 0
    best_set = [0, 1, 2, 5]
    
    def calc_score(u):
        if not u:
            return 0
        max_u = u[-1]
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

    # Strategy 1: U = {0, ..., n-2} U {K}
    # Optimize n and K
    for n in range(3, 30):
        # K must be > 2*(n-2) to keep sumsets mostly disjoint
        # But even if not disjoint, we can calculate exactly
        # We search K from n-1 to 2*(n-2) + 10
        min_k = n - 1
        max_k = 2 * (n - 2) + 20
        for K in range(min_k, max_k + 1):
            u = list(range(0, n - 1)) + [K]
            u.sort()
            s = calc_score(u)
            if s > best_score:
                best_score = s
                best_set = u

    # Strategy 2: Two blocks {0, ..., n-1} U {K, ..., K+n-1}
    for n in range(2, 15):
        for K in range(n + 1, 3 * n + 10):
            u = list(range(0, n)) + list(range(K, K + n))
            u.sort()
            s = calc_score(u)
            if s > best_score:
                best_score = s
                best_set = u

    return best_set
