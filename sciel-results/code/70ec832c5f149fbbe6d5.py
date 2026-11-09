import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 78.0  # safety margin
    UNIVERSE = set(range(1, 25))
    
    def is_valid(S):
        S_list = list(S)
        n = len(S_list)
        if n <= 1:
            return True
        for i in range(n):
            target = 1.0 / S_list[i]
            others = [1.0 / S_list[j] for j in range(n) if j != i]
            m = len(others)
            achievable = {0.0}
            for val in others:
                new_sums = set()
                for s in achievable:
                    ns = s + val
                    if abs(ns - target) < 1e-9:
                        return False
                    new_sums.add(ns)
                achievable |= new_sums
        return True
    
    best_set = set()
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
    
    # Try multiple random restarts with local search
    for restart in range(200):
        if time.time() - start_time > DEADLINE:
            break
        
        # Initialize
        if restart == 0:
            current = set(UNIVERSE)
        elif restart % 3 == 1:
            current = set(random.sample(range(1, 25), random.randint(8, 18)))
        else:
            current = set(random.sample(range(1, 25), random.randint(5, 15)))
        
        # Remove elements until valid
        while not is_valid(current) and current:
            current.remove(random.choice(list(current)))
        
        update_best(current)
        
        # Try to add elements
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            candidates = list(UNIVERSE - current)
            random.shuffle(candidates)
            for c in candidates:
                if time.time() - start_time > DEADLINE:
                    break
                new_set = set(current)
                new_set.add(c)
                if is_valid(new_set):
                    current = new_set
                    improved = True
                    update_best(current)
                    break
        
        # Try swaps to potentially find larger sets
        for _ in range(50):
            if time.time() - start_time > DEADLINE:
                break
            if not current:
                break
            to_remove = random.choice(list(current))
            to_add_candidates = list(UNIVERSE - current)
            if not to_add_candidates:
                break
            to_add = random.choice(to_add_candidates)
            new_set = set(current)
            new_set.discard(to_remove)
            new_set.add(to_add)
            if is_valid(new_set):
                current = new_set
                update_best(current)
    
    if not best_set:
        best_set = {24}
    
    return list(best_set)
