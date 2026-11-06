import random
import time

def count_3aps_involving(x, S):
    count = 0
    for d in range(1, 257):
        if (x + d) in S and (x + 2 * d) in S:
            count += 1
        if (x - d) in S and (x + d) in S:
            count += 1
        if (x - 2 * d) in S and (x - d) in S:
            count += 1
    return count

def creates_4ap(x, S):
    for d in range(1, 257):
        if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
            return True
        if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
            return True
        if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
            return True
        if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
            return True
    return False

def swap_creates_4ap(x_out, x_in, S):
    for d in range(1, 257):
        # 1. x_in is 1st
        t1, t2, t3 = x_in + d, x_in + 2 * d, x_in + 3 * d
        if t1 in S and t2 in S and t3 in S and t1 != x_out and t2 != x_out and t3 != x_out:
            return True
        # 2. x_in is 2nd
        t1, t2, t3 = x_in - d, x_in + d, x_in + 2 * d
        if t1 in S and t2 in S and t3 in S and t1 != x_out and t2 != x_out and t3 != x_out:
            return True
        # 3. x_in is 3rd
        t1, t2, t3 = x_in - 2 * d, x_in - d, x_in + d
        if t1 in S and t2 in S and t3 in S and t1 != x_out and t2 != x_out and t3 != x_out:
            return True
        # 4. x_in is 4th
        t1, t2, t3 = x_in - 3 * d, x_in - 2 * d, x_in - d
        if t1 in S and t2 in S and t3 in S and t1 != x_out and t2 != x_out and t3 != x_out:
            return True
    return False

def initial_greedy_construct():
    n = 256
    A = set(range(1, n + 1))
    while True:
        ap_found = None
        for a in range(1, n + 1):
            if a not in A: continue
            max_d = (n - a) // 3
            for d in range(1, max_d + 1):
                if (a+d) in A and (a+2*d) in A and (a+3*d) in A:
                    ap_found = (a, a+d, a+2*d, a+3*d)
                    break
            if ap_found: break
        
        if not ap_found:
            break
        
        min_count = float('inf')
        min_elem = None
        for elem in ap_found:
            c = count_3aps_involving(elem, A)
            if c < min_count:
                min_count = c
                min_elem = elem
        A.discard(min_elem)
        
    return A

def construct():
    n = 256
    random.seed(42)
    start_time = time.time()
    time_limit = 55
    
    A = initial_greedy_construct()
    current_score = sum(count_3aps_involving(x, A) for x in A) // 3
    best_score = current_score
    best_A = set(A)
    
    T = 10.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        move_type = random.random()
        if move_type < 0.3:
            candidates = [x for x in range(1, n + 1) if x not in A]
            if not candidates: continue
            x = random.choice(candidates)
            if creates_4ap(x, A):
                continue
            delta = count_3aps_involving(x, A)
            if delta >= 0 or random.random() < pow(2.718, -delta / T):
                A.add(x)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_A = set(A)
        elif move_type < 0.6:
            if not A: continue
            x = random.choice(list(A))
            delta = -count_3aps_involving(x, A)
            if delta >= 0 or random.random() < pow(2.718, -delta / T):
                A.discard(x)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_A = set(A)
        else:
            if not A: continue
            candidates_in = [x for x in range(1, n + 1) if x not in A]
            if not candidates_in: continue
            x_out = random.choice(list(A))
            x_in = random.choice(candidates_in)
            
            if swap_creates_4ap(x_out, x_in, A):
                continue
            
            delta_remove = -count_3aps_involving(x_out, A)
            c_in_A = count_3aps_involving(x_in, A)
            
            zs = set()
            d1 = x_out - x_in
            if d1 > 0: zs.add(x_out + d1)
            d2 = x_in - x_out
            if d2 > 0: zs.add(x_in + d2)
            if (x_out - x_in) % 2 == 0 and (x_out - x_in) != 0:
                zs.add(x_in + (x_out - x_in)//2)
            if (x_in - x_out) % 2 == 0 and (x_in - x_out) != 0:
                zs.add(x_out + (x_in - x_out)//2)
            zs.add(2 * x_in - x_out)
            zs.add(2 * x_out - x_in)
            
            valid_zs = [z for z in zs if z in A and z != x_out]
            overlap = len(valid_zs)
            
            c_in_A_minus_xout = c_in_A - overlap
            delta = delta_remove + c_in_A_minus_xout
            
            if delta >= 0 or random.random() < pow(2.718, -delta / T):
                A.discard(x_out)
                A.add(x_in)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_A = set(A)
            
        T *= cooling_rate
        
    return sorted(best_A)
