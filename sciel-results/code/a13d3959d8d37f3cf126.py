import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    def calc_score(U):
        if not U:
            return 0.0
        U = sorted(set(U))
        if 0 not in U:
            U.append(0)
            U = sorted(set(U))
        
        max_u = max(U)
        if max_u == 0:
            return 0.0
            
        # Calculate U+U and U-U
        # For efficiency, use sets
        sums = set()
        diffs = set()
        n = len(U)
        for i in range(n):
            for j in range(n):
                sums.add(U[i] + U[j])
                diffs.add(U[i] - U[j])
        
        num_sums = len(sums)
        num_diffs = len(diffs)
        
        if num_sums == 0:
            return 0.0
            
        ratio = num_diffs / num_sums
        if ratio <= 1:
            return 1.0
            
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        score = 1 + math.log(ratio) / denom
        return score

    # Family 1: Powers of 2
    # U = {0, 1, 2, 4, ..., 2^(m-1)}
    for m in range(1, 50):
        if time.time() > deadline:
            break
        U = [0] + [2**i for i in range(m)]
        s = calc_score(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    # Family 2: Powers of 3
    for m in range(1, 30):
        if time.time() > deadline:
            break
        U = [0] + [3**i for i in range(m)]
        s = calc_score(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    # Family 3: Beatty Sequences
    # U = { floor(i * alpha) for i in 0..n-1 }
    # Try some alpha
    alphas = [1.5, 1.6, 1.7, 1.8, 1.9, 2.1, 2.5, 3.0, 3.5]
    for alpha in alphas:
        for n in range(5, 50):
            if time.time() > deadline:
                break
            U = [int(i * alpha) for i in range(n)]
            if 0 not in U:
                U.append(0)
            U = sorted(set(U))
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best_set = U
                
    return best_set
