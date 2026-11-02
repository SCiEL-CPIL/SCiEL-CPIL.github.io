import math

def construct():
    best_score = 0
    best_set = [0]
    
    # Candidate 1: Geometric progressions U = {0, 1, b, b^2, ..., b^(k-1)}
    for b in range(2, 11):
        for k in range(2, 16):
            u = [0]
            val = 1
            for _ in range(k - 1):
                u.append(val)
                val *= b
            u.sort()
            
            # Calculate score
            sums = set()
            diffs = set()
            max_u = u[-1]
            
            for i in u:
                for j in u:
                    sums.add(i + j)
                    diffs.add(i - j)
            
            if not sums or not diffs:
                continue
                
            ratio = len(diffs) / len(sums)
            if ratio <= 1:
                continue
                
            denom = math.log2(2 * max_u + 1)
            if denom == 0:
                continue
                
            score = 1 + math.log2(ratio) / denom
            
            if score > best_score:
                best_score = score
                best_set = u

    # Candidate 2: Mersenne-like sets U = {0, 1, 3, 7, ..., 2^k-1}
    for k in range(2, 16):
        u = [0]
        val = 1
        for _ in range(k - 1):
            u.append(val)
            val = val * 2 + 1
        u.sort()
        
        sums = set()
        diffs = set()
        max_u = u[-1]
        
        for i in u:
            for j in u:
                sums.add(i + j)
                diffs.add(i - j)
        
        if not sums or not diffs:
            continue
            
        ratio = len(diffs) / len(sums)
        if ratio <= 1:
            continue
            
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            continue
            
        score = 1 + math.log2(ratio) / denom
        
        if score > best_score:
            best_score = score
            best_set = u

    return best_set
