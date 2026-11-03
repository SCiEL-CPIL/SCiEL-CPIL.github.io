import math

def construct():
    best_score = 0
    best_set = [0, 1, 4, 16]
    
    def calc_score(u):
        if not u:
            return 0
        max_u = max(u)
        if max_u == 0:
            return 0
        
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

    # Search over geometric progressions {0, b, b^2, ..., b^(k-1)}
    # and variants {0, 1, b, b^2, ...}
    
    for base in range(2, 15):
        for k in range(3, 12):
            # Variant 1: 0, base, base^2, ...
            u1 = [0]
            val = base
            for i in range(1, k):
                u1.append(val)
                val *= base
            
            s1 = calc_score(u1)
            if s1 > best_score:
                best_score = s1
                best_set = u1
            
            # Variant 2: 0, 1, base, base^2, ...
            if base > 1:
                u2 = [0, 1]
                val = base
                for i in range(2, k):
                    u2.append(val)
                    val *= base
                
                s2 = calc_score(u2)
                if s2 > best_score:
                    best_score = s2
                    best_set = u2
                    
    # Also try small brute force for M < 30 to catch non-geometric outliers
    # The current seed did this, but maybe we can be smarter?
    # For now, stick to the geometric family as per directive.
    
    return best_set
