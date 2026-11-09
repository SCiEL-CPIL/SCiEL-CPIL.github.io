import random
import math
import time

def construct():
    def score(S):
        present = [False] * 513
        for x in S:
            present[x] = True
            
        diffs = set()
        list_S = list(S)
        n = len(list_S)
        
        for i in range(n):
            a = list_S[i]
            for j in range(i+1, n):
                c = list_S[j]
                s = a + c
                if s % 2 == 0:
                    b = s // 2
                    if present[b]:
                        d = c - a
                        diffs.add(d)
        return len(diffs)

    def initial_candidates():
        candidates = []
        candidates.append(set(range(1, 33)))
        candidates.append(set(range(1, 17)) | set(range(33, 49)))
        candidates.append(set(range(1, 12)) | set(range(24, 35)) | set(range(47, 58)))
        candidates.append(set(random.sample(range(1, 100), 32)))
        candidates.append(set(random.sample(range(1, 513), 32)))
        return candidates

    def move(S):
        items = list(S)
        x = random.choice(items)
        y = random.randint(1, 512)
        if y == x:
            return S
        new_S = set(S)
        new_S.remove(x)
        new_S.add(y)
        return new_S

    def run_sa(start_S, duration):
        start_time = time.time()
        current_S = set(start_S)
        current_score = score(current_S)
        best_S = set(current_S)
        best_score = current_score
        
        T = 5.0
        cooling_rate = 0.999
        min_T = 0.01
        
        while time.time() - start_time < duration and T > min_T:
            new_S = move(current_S)
            new_score = score(new_S)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_S = new_S
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(current_S)
            
            T *= cooling_rate
            
        return best_S, best_score

    total_time = 110
    num_runs = 5
    time_per_run = total_time / num_runs
    
    global_best_S = set()
    global_best_score = -1
    
    for i in range(num_runs):
        candidates = initial_candidates()
        # Pick a random candidate to start this run
        start_S = random.choice(candidates)
        best_S, best_score = run_sa(start_S, time_per_run)
        
        if best_score > global_best_score:
            global_best_score = best_score
            global_best_S = best_S
            
    return list(global_best_S)
