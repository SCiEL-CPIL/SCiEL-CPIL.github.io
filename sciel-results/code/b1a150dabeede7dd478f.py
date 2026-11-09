import random
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 115  # Safety margin
    N = 284
    PRESCRIBED = [1, 2, 5, 11]
    PRESCRIBED_SET = set(PRESCRIBED)

    def get_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def collisions(S):
        # Returns number of duplicate sums
        sums = []
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.append(S[i] + S[j])
        return len(sums) - len(set(sums))

    def score(S):
        c = collisions(S)
        if c > 0:
            return len(S) - 100 * c
        return len(S)

    # 1. Greedy Initialization
    # We try multiple random greedy initializations to find a good starting point
    best_initial_S = []
    best_initial_score = -1
    
    for _ in range(10):
        S = list(PRESCRIBED)
        S_set = set(S)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if x in S_set:
                continue
            S_new = S + [x]
            if collisions(S_new) == 0:
                S = S_new
                S_set.add(x)
        
        s = score(S)
        if s > best_initial_score:
            best_initial_score = s
            best_initial_S = list(S)

    current_S = list(best_initial_S)
    current_score = score(current_S)
    best_S = list(current_S)
    best_score = current_score
    
    # 2. Simulated Annealing
    T = 10.0
    T_min = 0.01
    alpha = 0.99995
    
    current_S_set = set(current_S)
    
    while time.time() - start_time < DEADLINE:
        # Generate neighbor
        move_type = random.random()
        new_S = list(current_S)
        new_S_set = set(new_S)
        
        if move_type < 0.6:
            # Swap: Remove one non-prescribed, add one random
            non_prescribed = [y for y in new_S if y not in PRESCRIBED_SET]
            if not non_prescribed:
                continue
            y = random.choice(non_prescribed)
            new_S.remove(y)
            new_S_set.discard(y)
            
            # Pick random x not in new_S
            while True:
                x = random.randint(1, N)
                if x not in new_S_set:
                    break
            new_S.append(x)
            new_S_set.add(x)
            
        elif move_type < 0.8:
            # Add
            while True:
                x = random.randint(1, N)
                if x not in new_S_set:
                    break
            new_S.append(x)
            new_S_set.add(x)
            
        else:
            # Remove
            non_prescribed = [y for y in new_S if y not in PRESCRIBED_SET]
            if not non_prescribed:
                continue
            y = random.choice(non_prescribed)
            new_S.remove(y)
            new_S_set.discard(y)
            
        new_score = score(new_S)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_S = new_S
            current_S_set = new_S_set
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_S = list(current_S)
                
        T *= alpha
        if T < T_min:
            T = T_min

    return best_S
