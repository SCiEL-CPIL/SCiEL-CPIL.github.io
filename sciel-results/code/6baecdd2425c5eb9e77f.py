def construct():
    import numpy as np
    
    def count_nonzero_square(coeffs):
        # coeffs is list in ascending degree
        # compute square
        n = len(coeffs)
        square = [0] * (2*n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(n):
                if coeffs[j] == 0:
                    continue
                square[i+j] += coeffs[i] * coeffs[j]
        return sum(1 for c in square if c != 0)
    
    # Try several candidates
    candidates = []
    
    # Candidate 1: all ones
    c1 = [1]*12
    candidates.append(c1)
    
    # Candidate 2: alternating
    c2 = [1 if i%2==0 else -1 for i in range(12)]
    candidates.append(c2)
    
    # Candidate 3: first 6 positive, last 6 negative
    c3 = [1]*6 + [-1]*6
    candidates.append(c3)
    
    # Candidate 4: palindrome with sign change
    c4 = [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1]
    candidates.append(c4)
    
    # Candidate 5: try to cause more cancellations
    c5 = [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1]
    candidates.append(c5)
    
    # Find best
    best = None
    best_count = float('inf')
    for c in candidates:
        cnt = count_nonzero_square(c)
        if cnt < best_count:
            best_count = cnt
            best = c
    
    return best
