import time
import random
import math

def construct():
    deadline = time.time() + 55
    
    # Precompute conflict triples: (a,b,c,ab,ac,bc) with a<b<c, all sums <=128
    conflict_triples = []
    for b in range(2, 64):
        for c in range(b+1, 129-b):
            for a in range(1, b):
                ab, ac, bc = a+b, a+c, b+c
                if ab <= 128 and ac <= 128 and bc <= 128:
                    conflict_triples.append((a, b, c, ab, ac, bc))
    
    def count_violations(S):
        v = 0
        for t in conflict_triples:
            if t[0] in S and t[1] in S and t[2] in S and t[3] in S and t[4] in S and t[5] in S:
                v += 1
        return v
    
    # Start with 65..128
    S = set(range(65, 129))
    
    # Greedy: try adding 1..64
    for x in range(1, 65):
        if time.time() > deadline - 15:
            break
        S_test = S | {x}
        if count_violations(S_test) == 0:
            S = S_test
    
    # SA
    best_S = set(S)
    best_size = len(S)
    current_S = set(S)
    
    T = 5.0
    cooling = 0.99995
    min_T = 0.001
    
    while time.time() < deadline - 2:
        if T < min_T:
            T = min_T
        
        x = random.randint(1, 128)
        
        if x in current_S:
            new_S = set(current_S)
            new_S.discard(x)
        else:
            new_S = set(current_S)
            new_S.add(x)
        
        new_viol = count_violations(new_S)
        cur_viol = count_violations(current_S)
        
        cur_e = cur_viol * 1000 - len(current_S)
        new_e = new_viol * 1000 - len(new_S)
        delta = new_e - cur_e
        
        accept = False
        if delta <= 0:
            accept = True
        elif delta < 5000 and T > 0:
            if random.random() < math.exp(-delta / T):
                accept = True
        
        if accept:
            current_S = new_S
            if new_viol == 0 and len(current_S) > best_size:
                best_S = set(current_S)
                best_size = len(current_S)
        
        T *= cooling
    
    if count_violations(best_S) == 0:
        return list(best_S)
    return list(range(65, 129))
