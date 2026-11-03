import math

def construct():
    best_set = [0, 1, 3]
    best_score = 0.0
    
    for n in range(1, 15):
        U = [3**i for i in range(n)]
        if 0 not in U:
            U.append(0)
        U.sort()
        
        # Calculate U+U
        sumset = set()
        for u in U:
            for v in U:
                sumset.add(u + v)
        size_sum = len(sumset)
        
        # Calculate U-U
        diffset = set()
        for u in U:
            for v in U:
                diffset.add(u - v)
        size_diff = len(diffset)
        
        max_u = max(U)
        if max_u == 0:
            continue
            
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            continue
            
        ratio = size_diff / size_sum
        score = 1 + math.log(ratio) / denom
        
        if score > best_score:
            best_score = score
            best_set = U
            
    return best_set
