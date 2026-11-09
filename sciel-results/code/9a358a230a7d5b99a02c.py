import random
import time

def construct():
    start_time = time.time()
    limit = 512
    max_size = 48
    best_set = []
    
    def _try_build(order):
        S = []
        bits_S = 1
        sum_S = 0
        
        for x in order:
            if len(S) >= max_size:
                break
            if x in S:
                continue
            
            n = len(S)
            valid = True
            
            # Check 1: x is not the mean of any subset of S (size >= 2)
            if n >= 2:
                for k in range(2, n + 1):
                    target = k * x
                    if target > sum_S:
                        break
                    if (bits_S >> target) & 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check 2: no a in S becomes the mean of a subset of S U {x} containing x
            if n >= 1:
                for idx_a, a in enumerate(S):
                    bits_without_a = 1
                    sum_without_a = 0
                    for idx_y, y in enumerate(S):
                        if idx_y == idx_a:
                            continue
                        bits_without_a |= bits_without_a << y
                        sum_without_a += y
                    
                    for k in range(2, n + 1):
                        target = k * a - x
                        if target < 1:
                            continue
                        if target > sum_without_a:
                            continue
                        if (bits_without_a >> target) & 1:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                S.append(x)
                bits_S |= bits_S << x
                sum_S += x
        
        return S
    
    attempts = 0
    while time.time() - start_time < 55:
        attempts += 1
        order = list(range(1, limit + 1))
        random.shuffle(order)
        
        S = _try_build(order)
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == max_size:
                break
    
    return best_set
