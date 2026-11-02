def get_best_set():
    best_set = [0, 1, 3]
    best_score = 0
    
    for n in range(3, 16):
        seq = generate_mian_chowla(n)
        # Calculate score
        u = set(seq)
        sum_set = {a + b for a in u for b in u}
        diff_set = {a - b for a in u for b in u}
        
        if not sum_set or not diff_set:
            continue
            
        ratio = len(diff_set) / len(sum_set)
        max_u = max(u)
        
        if ratio <= 1:
            score = 1.0
        else:
            import math
            score = 1 + math.log2(ratio) / math.log2(2 * max_u + 1)
            
        if score > best_score:
            best_score = score
            best_set = seq
            
    return best_set
