import random
import time

def construct():
    deadline = time.time() + 50
    N = 128
    best_size = 0
    best_set = None
    
    all_vals = list(range(1, N+1))
    
    seed_set = list(range(N//2+1, N+1))
    best_size = len(seed_set)
    best_set = seed_set[:]
    
    def can_add_fast(bs, S, v):
        for i, a in enumerate(S):
            s_av = a + v
            if s_av > N:
                continue
            if not ((bs >> (s_av - 1)) & 1):
                continue
            for b in S:
                if b == a:
                    continue
                s_bv = b + v
                if s_bv > N:
                    continue
                if not ((bs >> (s_bv - 1)) & 1):
                    continue
                s_ab = a + b
                if s_ab > N:
                    continue
                if (bs >> (s_ab - 1)) & 1:
                    return False
        return True
    
    for trial in range(500):
        if time.time() > deadline - 5:
            break
        
        if trial % 3 == 0:
            order = all_vals[:]
            random.shuffle(order)
        elif trial % 3 == 1:
            order = sorted(all_vals, reverse=True)
        else:
            order = sorted(all_vals)
        
        S = []
        bs = 0
        
        for v in order:
            if (bs >> (v-1)) & 1:
                continue
            if can_add_fast(bs, S, v):
                S.append(v)
                bs |= (1 << (v-1))
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    if best_set:
        S = best_set[:]
        bs = 0
        for x in S:
            bs |= (1 << (x-1))
        for v in all_vals:
            if time.time() > deadline - 5:
                break
            if not ((bs >> (v-1)) & 1) and can_add_fast(bs, S, v):
                S.append(v)
                bs |= (1 << (v-1))
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    return [int(x) for x in best_set]

