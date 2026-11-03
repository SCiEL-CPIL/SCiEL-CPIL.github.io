def construct():
    n, m, p = 4, 4, 6
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_idx = 0
    
    # Strassen for 4x4 x 4x4
    # 7 steps.
    # Each step is a product of linear combos of 2x2 blocks.
    # We expand each step into rank-1 terms.
    
    # To keep it simple and correct, I will use the fact that
    # Rank(4x4x4x4) <= 49.
    # I will use a known rank 49 decomposition.
    
    # Since I don't have it hardcoded, I will use Rank 56 (Strassen outer, Schoolbook inner).
    # 7 * 8 = 56.
    
    steps = [
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}, {'C00': 1, 'C11': 1}),
        ({'A10': 1, 'A11': 1}, {'B00': 1}, {'C10': 1, 'C11': -1}),
        ({'A00': 1}, {'B01': 1, 'B11': -1}, {'C01': 1, 'C11': 1}),
        ({'A11': 1}, {'B10': 1, 'B00': -1}, {'C00': 1, 'C10': 1}),
        ({'A00': 1, 'A01': 1}, {'B11': 1}, {'C00': -1, 'C01': 1}),
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B01': 1}, {'C10': 1}),
        ({'A01': 1, 'A11': -1}, {'B10': 1, 'B11': 1}, {'C00': -1, 'C11': 1}),
    ]
    
    # ... implementation ...
