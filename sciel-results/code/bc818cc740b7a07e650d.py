import math

def construct():
    # Difference-sparse construction: U = {0, d, 2d, ..., (n-1)d}
    # |U+U| = 2n-1, |U-U| = 2n-1 (for arithmetic progression)
    # This gives ratio 1, score 1. Not useful.
    
    # The directive says: "difference-sparse" construction where U is a subset of a short
    # arithmetic progression with a large common difference, optimizing the gap size to
    # balance linear growth of difference set against quadratic growth of sumset.
    
    # Let me think about this differently. Consider U = {0, a, b} where a < b.
    # |U+U|: 0, a, b, 2a, a+b, 2b. If all distinct, |U+U|=6.
    # |U-U|: 0, a, b, -a, -b, b-a, a-b. If all distinct (b != 2a), |U-U|=7.
    # Ratio = 7/6. Score = 1 + log2(7/6)/log2(2b+1).
    # To maximize, minimize b. b=2, a=1: U={0,1,2}. Sums: 0,1,2,3,4. |U+U|=5. Diffs: -2,-1,0,1,2. |U-U|=5. Ratio=1.
    # b=3, a=1: U={0,1,3}. Sums: 0,1,2,3,4,6. |U+U|=6. Diffs: -3,-2,-1,0,1,2,3. |U-U|=7. Ratio=7/6.
    # Score = 1 + log2(7/6)/log2(7) = 1 + 0.222/2.807 = 1.079.
    # b=3, a=2: U={0,2,3}. Sums: 0,2,3,4,5,6. |U+U|=6. Diffs: -3,-2,-1,0,1,2,3. |U-U|=7. Same ratio.
    # Score same.
    
    # For U = {0, 1, 3, 7}: 
    # Sums: 0,1,2,3,4,6,7,8,10,14 -> 10 elements
    # Diffs: -7,-6,-4,-3,-2,-1,0,1,2,3,4,6,7 -> 13 elements
    # Ratio = 13/10 = 1.3
    # Score = 1 + log2(1.3)/log2(15) = 1 + 0.3785/3.907 = 1.0969
    
    # For U = {0, 1, 3, 7, 15}:
    # Let me compute. This is like Mersenne numbers.
    # Actually, let me just do a small search for n up to 7, M up to 30.
    
    def calc_score(u):
        if not u or len(u) < 2:
            return 0
        max_u = max(u)
        sums = set()
        diffs = set()
        for i in u:
            for j in u:
                sums.add(i + j)
                diffs.add(i - j)
        num_sums = len(sums)
        num_diffs = len(diffs)
        if num_sums == 0:
            return 0
        ratio = num_diffs / num_sums
        if ratio <= 1:
            return 0
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            return 0
        return 1 + math.log2(ratio) / denom

    best_score = 0
    best_set = [0, 1, 3, 7, 15]
    
    # Search small sets
    for n in range(3, 8):
        for M in range(n, 35):
            # Generate all combinations of n-1 elements from 1..M
            import itertools
            for comb in itertools.combinations(range(1, M + 1), n - 1):
                u = [0] + list(comb)
                s = calc_score(u)
                if s > best_score:
                    best_score = s
                    best_set = u
    
    return best_set
