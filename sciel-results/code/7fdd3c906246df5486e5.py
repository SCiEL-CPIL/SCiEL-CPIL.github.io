import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    N = 256
    
    # Greedy Initialization
    in_set = [False] * (N + 1)
    for x in range(1, N + 1):
        creates_4ap = False
        # Check 4-APs containing x
        # x as 1st: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if in_set[x+d] and in_set[x+2*d] and in_set[x+3*d]:
                creates_4ap = True
                break
        if not creates_4ap:
            # x as 2nd: x-d, x, x+d, x+2d
            max_d = min(x - 1, (N - x) // 2)
            for d in range(1, max_d + 1):
                if in_set[x-d] and in_set[x+d] and in_set[x+2*d]:
                    creates_4ap = True
                    break
        if not creates_4ap:
            # x as 3rd: x-2d, x-d, x, x+d
            max_d = min((x - 1) // 2, N - x)
            for d in range(1, max_d + 1):
                if in_set[x-2*d] and in_set[x-d] and in_set[x+d]:
                    creates_4ap = True
                    break
        if not creates_4ap:
            # x as 4th: x-3d, x-2d, x-d, x
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                if in_set[x-3*d] and in_set[x-2*d] and in_set[x-d]:
                    creates_4ap = True
                    break
        if not creates_4ap:
            in_set[x] = True

    # Calculate initial scores
    def calc_scores(in_set):
        s3 = 0
        s4 = 0
        # 3-APs
        for d in range(1, N // 2 + 1):
            for a in range(1, N - 2 * d + 1):
                if in_set[a] and in_set[a+d] and in_set[a+2*d]:
                    s3 += 1
        # 4-APs
        for d in range(1, N // 3 + 1):
            for a in range(1, N - 3 * d + 1):
                if in_set[a] and in_set[a+d] and in_set[a+2*d] and in_set[a+3*d]:
                    s4 += 1
        return s3, s4

    cur_s3, cur_s4 = calc_scores(in_set)
    P = 10.0
    cur_S = cur_s3 - P * cur_s4
    
    best_valid_set = in_set.copy()
    best_valid_score = cur_s3
    
    T = 5.0
    alpha = 0.999
    
    while time.time() < deadline:
        x = random.randint(1, N)
        
        if in_set[x]:
            # Remove x
            delta_s3 = 0
            delta_s4 = 0
            
            # 3-APs lost
            # x middle
            max_d = min(x - 1, N - x)
            for d in range(1, max_d + 1):
                if in_set[x-d] and in_set[x+d]:
                    delta_s3 -= 1
            # x start
            max_d = (N - x) // 2
            for d in range(1, max_d + 1):
                if in_set[x+d] and in_set[x+2*d]:
                    delta_s3 -= 1
            # x end
            max_d = (x - 1) // 2
            for d in range(1, max_d + 1):
                if in_set[x-2*d] and in_set[x-d]:
                    delta_s3 -= 1
            
            # 4-APs lost
            # x 1st
            max_d = (N - x) // 3
            for d in range(1, max_d + 1):
                if in_set[x+d] and in_set[x+2*d] and in_set[x+3*d]:
                    delta_s4 -= 1
            # x 2nd
            max_d = min(x - 1, (N - x) // 2)
            for d in range(1, max_d + 1):
                if in_set[x-d] and in_set[x+d] and in_set[x+2*d]:
                    delta_s4 -= 1
            # x 3rd
            max_d = min((x - 1) // 2, N - x)
            for d in range(1, max_d + 1):
                if in_set[x-2*d] and in_set[x-d] and in_set[x+d]:
                    delta_s4 -= 1
            # x 4th
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                if in_set[x-3*d] and in_set[x-2*d] and in_set[x-d]:
                    delta_s4 -= 1
            
            in_set[x] = False
        else:
            # Add x
            delta_s3 = 0
            delta_s4 = 0
            
            # 3-APs gained
            max_d = min(x - 1, N - x)
            for d in range(1, max_d + 1):
                if in_set[x-d] and in_set[x+d]:
                    delta_s3 += 1
            max_d = (N - x) // 2
            for d in range(1, max_d + 1):
                if in_set[x+d] and in_set[x+2*d]:
                    delta_s3 += 1
            max_d = (x - 1) // 2
            for d in range(1, max_d + 1):
                if in_set[x-2*d] and in_set[x-d]:
                    delta_s3 += 1
            
            # 4-APs gained
            max_d = (N - x) // 3
            for d in range(1, max_d + 1):
                if in_set[x+d] and in_set[x+2*d] and in_set[x+3*d]:
                    delta_s4 += 1
            max_d = min(x - 1, (N - x) // 2)
            for d in range(1, max_d + 1):
                if in_set[x-d] and in_set[x+d] and in_set[x+2*d]:
                    delta_s4 += 1
            max_d = min((x - 1) // 2, N - x)
            for d in range(1, max_d + 1):
                if in_set[x-2*d] and in_set[x-d] and in_set[x+d]:
                    delta_s4 += 1
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                if in_set[x-3*d] and in_set[x-2*d] and in_set[x-d]:
                    delta_s4 += 1
                    
            in_set[x] = True
            
        new_s3 = cur_s3 + delta_s3
        new_s4 = cur_s4 + delta_s4
        new_S = new_s3 - P * new_s4
        
        if new_S > cur_S or random.random() < math.exp((new_S - cur_S) / T):
            cur_s3 = new_s3
            cur_s4 = new_s4
            cur_S = new_S
            
            if cur_s4 == 0 and cur_s3 > best_valid_score:
                best_valid_score = cur_s3
                best_valid_set = in_set.copy()
        else:
            in_set[x] = not in_set[x]
            
        T *= alpha
        if T < 0.01:
            T = 0.01

    return [i for i in range(1, N + 1) if best_valid_set[i]]
