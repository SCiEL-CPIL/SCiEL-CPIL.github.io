import time
import random
import itertools

def is_sidon_4(combo):
    a, b, c, d = combo
    sums = [
        a+a, b+b, c+c, d+d,
        a+b, a+c, a+d,
        b+c, b+d,
        c+d
    ]
    return len(set(sums)) == 10

def count_sidon_4(ambient):
    count = 0
    # 12 elements, 495 combos
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                for l in range(k+1, 12):
                    if is_sidon_4((ambient[i], ambient[j], ambient[k], ambient[l])):
                        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    rng = random.Random(42)
    
    best_set = None
    best_cost = float('inf')
    
    # Try multiple restarts
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Generate initial set
        # Mix of random and structured
        if restart % 2 == 0:
            # Two clusters
            c1 = sorted(rng.sample(range(0, 40), 6))
            c2 = sorted(rng.sample(range(90, 128), 6))
            current_set = sorted(c1 + c2)
        else:
            # Random
            current_set = sorted(rng.sample(range(129), 12))
            
        if len(set(current_set)) != 12:
            continue
            
        current_cost = count_sidon_4(current_set)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_set = current_set[:]
            
        if best_cost == 0:
            return best_set

        # Hill Climbing
        for i in range(2000):
            if time.time() > deadline:
                break
                
            idx = rng.randint(0, 11)
            old_val = current_set[idx]
            
            # Try 5 new values
            for _ in range(5):
                new_val = rng.randint(0, 128)
                if new_val == old_val or new_val in current_set:
                    continue
                    
                new_set = current_set[:]
                new_set[idx] = new_val
                new_set = sorted(new_set)
                
                new_cost = count_sidon_4(new_set)
                
                if new_cost < current_cost:
                    current_set = new_set
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_set = new_set[:]
                    if best_cost == 0:
                        return best_set
                elif new_cost == current_cost:
                    if rng.random() < 0.05:
                        current_set = new_set

    return best_set
