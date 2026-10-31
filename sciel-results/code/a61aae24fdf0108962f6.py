import math

def calculate_score(U):
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
        
    val = 1 + (math.log(ratio) / denom)
    return val

def construct():
    best_score = 0
    best_set = [0, 1, 2]
    
    # Strategy 1: Powers of 2
    # U = {0, 1, 2, 4, ..., 2^(k-1)}
    for k in range(3, 15):
        U = [0]
        val = 1
        for i in range(1, k):
            U.append(val)
            val *= 2
        score = calculate_score(U)
        if score > best_score:
            best_score = score
            best_set = U
            
    # Strategy 2: Small Arithmetic Progressions with a Gap
    # U = {0, 1, ..., k-1} U {M, M+1, ..., M+k-1}
    for k in range(2, 6):
        for M in range(k, 50):
            U = list(range(k))
            for i in range(k):
                U.append(M + i)
            if 0 not in U:
                U.append(0)
            U = sorted(set(U))
            score = calculate_score(U)
            if score > best_score:
                best_score = score
                best_set = U
                
    # Strategy 3: Fibonacci-like
    for k in range(3, 15):
        U = [0, 1]
        for i in range(k - 2):
            U.append(U[-1] + U[-2])
        score = calculate_score(U)
        if score > best_score:
            best_score = score
            best_set = U
            
    return best_set
