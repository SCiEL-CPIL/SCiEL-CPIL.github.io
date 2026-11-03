import math

def construct():
    best_score = 0
    best_set = [0]
    
    # Search over parameters for the two-AP construction
    # U = {0, d, ..., (k1-1)d} U {D, D+d, ..., D+(k2-1)d}
    # D = 2*(k1-1)*d + 1 (minimal to avoid sum overlap)
    # We can also vary D to see if it helps (it shouldn't for score, as it increases M)
    
    max_k = 20
    for k1 in range(1, max_k + 1):
        for k2 in range(1, max_k + 1):
            # Try d=1
            d = 1
            D = 2 * (k1 - 1) * d + 1
            u = list(range(0, k1 * d, d))
            u.extend(list(range(D, D + k2 * d, d)))
            u = sorted(set(u))
            
            # Calculate score
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
                continue
                
            ratio = num_diffs / num_sums
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
