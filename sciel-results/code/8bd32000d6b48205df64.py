import time
import random

def score_set(S):
    S_set = set(S)
    valid_d = set()
    for i in range(len(S)):
        for j in range(len(S)):
            if i == j: continue
            a = S[i]
            b = S[j]
            if a > b:
                a, b = b, a
            d = b - a
            if d == 0: continue
            # Check a, b, b+d
            if b + d in S_set:
                valid_d.add(d)
            # Check a-d, a, b
            if a - d in S_set:
                valid_d.add(d)
    return len(valid_d)

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    # Initial set
    # Try a few random starts or heuristic
    best_S = list(range(1, 33))
    best_score = score_set(best_S)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    S = best_S[:]
    current_score = best_score
    
    while time.time() < deadline and T > T_min:
        # Generate neighbor
        # 80% replace, 20% swap
        if random.random() < 0.8:
            i = random.randint(0, 31)
            old_val = S[i]
            # Find a new value
            # To avoid infinite loop, just pick random until unique
            while True:
                new_val = random.randint(1, 512)
                if new_val not in S:
                    break
            S[i] = new_val
            new_score = score_set(S)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_S = S[:]
            else:
                S[i] = old_val
        else:
            i, j = random.sample(range(32), 2)
            old_i, old_j = S[i], S[j]
            while True:
                new_i = random.randint(1, 512)
                if new_i not in S:
                    break
            S[i] = new_i
            while True:
                new_j = random.randint(1, 512)
                if new_j not in S:
                    break
            S[j] = new_j
            new_score = score_set(S)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_S = S[:]
            else:
                S[i] = old_i
                S[j] = old_j
                
        T *= cooling
        
    # Polish phase
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(32):
            old_val = best_S[i]
            best_local_val = old_val
            best_local_score = best_score
            # Try all possible replacements? 512 options. 32*512 = 16k evals. Fast.
            for new_val in range(1, 513):
                if new_val in best_S:
                    continue
                best_S[i] = new_val
                s = score_set(best_S)
                if s > best_local_score:
                    best_local_score = s
                    best_local_val = new_val
            best_S[i] = best_local_val
            if best_local_val != old_val:
                improved = True
                best_score = best_local_score
                
    return best_S
