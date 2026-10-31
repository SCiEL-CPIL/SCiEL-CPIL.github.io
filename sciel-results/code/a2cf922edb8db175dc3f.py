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
    
    # Generate power of 2 sets
    for n in range(2, 15):
        U = [2**i for i in range(n)]
        score = calculate_score(U)
        if score > best_score:
            best_score = score
            best_set = U
            
    return best_set
