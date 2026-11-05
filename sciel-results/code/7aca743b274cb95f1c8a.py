import random
import time

def score(S):
    S_set = set(S)
    diffs = set()
    for m in S_set:
        max_d = min(m - 1, 512 - m)
        for d in range(1, max_d + 1):
            if (m - d) in S_set and (m + d) in S_set:
                diffs.add(d)
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_score = 0
    best_set = set()
    
    while time.time() < deadline:
        # Choose an initialization strategy
        strategy = random.randint(0, 2)
        if strategy == 0:
            current_set = set(random.sample(range(1, 513), 32))
        elif strategy == 1:
            # Geometric-like
            current_set = set()
            x = 1
            while len(current_set) < 32 and x <= 512:
                current_set.add(x)
                x = int(x * 1.5) + random.randint(0, 2)
            # Fill remaining randomly
            while len(current_set) < 32:
                current_set.add(random.randint(1, 512))
        else:
            # Clustered
            center = random.randint(100, 400)
            current_set = set()
            while len(current_set) < 32:
                current_set.add(random.randint(max(1, center - 50), min(512, center + 50)))
        
        current_score = score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
            
        no_improve_count = 0
        max_no_improve = 500 # Restart sooner to explore more
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            elem_to_remove = random.choice(list(current_set))
            elem_to_add = random.randint(1, 512)
            
            if elem_to_add in current_set:
                no_improve_count += 1
                continue
                
            new_set = set(current_set)
            new_set.remove(elem_to_remove)
            new_set.add(elem_to_add)
            
            new_score = score(new_set)
            
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            else:
                no_improve_count += 1
                
    return sorted(best_set)
