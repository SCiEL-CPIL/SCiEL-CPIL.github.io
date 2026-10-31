import math

def construct():
    candidates = [
        [0],
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25]
    ]
    
    best_set = [0]
    best_score = -1.0
    
    for u in candidates:
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
            
        max_u = max(u)
        m = 2 * max_u + 1
        
        if m <= 1:
            score = 1.0
        else:
            sumset = set()
            for a in u:
                for b in u:
                    sumset.add(a + b)
            
            diffset = set()
            for a in u:
                for b in u:
                    diffset.add(a - b)
            
            r = len(diffset) / len(sumset)
            score = 1 + math.log(r) / math.log(m)
            
        if score > best_score:
            best_score = score
            best_set = u
            
    return best_set
