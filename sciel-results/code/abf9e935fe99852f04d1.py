import time
import random
import math

def _score(S_list):
    S_set = set(S_list)
    S_sorted = sorted(S_list)
    diffs = set()
    n = len(S_sorted)
    for i in range(n):
        for j in range(i + 1, n):
            d = S_sorted[j] - S_sorted[i]
            if S_sorted[j] + d in S_set:
                diffs.add(d)
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    best_set = list(range(1, 33))
    best_score = _score(best_set)
    
    random.seed(42)
    
    def run_sa(S_init, T_init=10.0, cooling=0.999):
        S = list(S_init)
        S_set = set(S)
        cur_score = _score(S)
        T = T_init
        
        while time.time() < deadline:
            idx = random.randint(0, 31)
            u = S[idx]
            
            while True:
                v = random.randint(1, 512)
                if v not in S_set:
                    break
            
            S[idx] = v
            S_set.remove(u)
            S_set.add(v)
            
            new_score = _score(S)
            
            if new_score >= cur_score:
                cur_score = new_score
            else:
                delta = cur_score - new_score
                # T might be 0
                if T > 1e-6:
                    prob = math.exp(-delta / T)
                else:
                    prob = 0.0
                if random.random() < prob:
                    cur_score = new_score
                else:
                    S[idx] = u
                    S_set.remove(v)
                    S_set.add(u)
            
            T *= cooling
            if T < 0.01:
                T = 0.01
                
        return S, cur_score

    # Initializers
    initializers = [
        list(range(1, 33)),
        list(range(1, 17)) + list(range(200, 216)),
        list(range(1, 11)) + list(range(50, 61)) + list(range(100, 111)) + list(range(150, 161)),
        random.sample(range(1, 513), 32)
    ]
    
    for init in initializers:
        if time.time() >= deadline:
            break
        S_new, sc_new = run_sa(init)
        if sc_new > best_score:
            best_score = sc_new
            best_set = S_new
            
    # Random restarts
    while time.time() < deadline:
        S_rand = random.sample(range(1, 513), 32)
        S_new, sc_new = run_sa(S_rand)
        if sc_new > best_score:
            best_score = sc_new
            best_set = S_new

    return best_set
