import time
import random

def _score(S):
    s_list = list(S)
    diffs = set()
    n = len(s_list)
    for i in range(n):
        for j in range(i+1, n):
            d = s_list[j] - s_list[i]
            if d <= 255:
                diffs.add(d)
    
    count = 0
    for d in diffs:
        S_d = {x - d for x in S}
        S_2d = {x - 2*d for x in S}
        if S & S_d & S_2d:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with seed
    best_set = set(range(1, 33))
    best_score = _score(best_set)
    
    current_set = set(best_set)
    current_score = best_score
    
    # Initial random sampling to find a better starting point
    for _ in range(20):
        if time.time() > deadline: break
        rand_set = set(random.sample(range(1, 513), 32))
        rand_score = _score(rand_set)
        if rand_score > best_score:
            best_score = rand_score
            best_set = rand_set
            current_set = rand_set
            current_score = rand_score

    while time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            x = random.choice(list(current_set))
            while True:
                y = random.randint(1, 512)
                if y not in current_set:
                    break
            
            new_set = set(current_set)
            new_set.remove(x)
            new_set.add(y)
            
            new_score = _score(new_set)
            
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            elif new_score < current_score and random.random() < 0.05:
                current_set = new_set
                current_score = new_score
        
        if time.time() < deadline:
             if random.random() < 0.7:
                 current_set = set(best_set)
                 current_score = best_score
             else:
                 current_set = set(random.sample(range(1, 513), 32))
                 current_score = _score(current_set)
                 if current_score > best_score:
                     best_score = current_score
                     best_set = set(current_set)

    return list(best_set)
