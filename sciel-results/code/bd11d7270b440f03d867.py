import random
import math
import time

def construct():
    N = 256
    # Precompute reciprocals
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i
    
    def is_valid_addition(S, x):
        # Check if x can be added to S without forming a 3-AP
        # S is a set
        for a in S:
            for b in S:
                if a == b: continue
                # Check if x, a, b form AP
                # Cases:
                # 1. x is middle: 2*x == a + b
                # 2. a is middle: 2*a == x + b
                # 3. b is middle: 2*b == x + a
                if 2 * x == a + b: return False
                if 2 * a == x + b: return False
                if 2 * b == x + a: return False
        return True

    def get_score(S):
        return sum(recip[x] for x in S)

    # 1. Greedy Initialization
    S = set()
    # Sort candidates by reciprocal value (descending)
    candidates = sorted(range(1, N + 1), key=lambda x: recip[x], reverse=True)
    for x in candidates:
        if is_valid_addition(S, x):
            S.add(x)
    
    best_S = set(S)
    best_score = get_score(best_S)
    
    # 2. Simulated Annealing
    start_time = time.time()
    time_limit = 80.0 # Leave 10s margin
    
    T = 0.1
    T_min = 0.001
    cooling_rate = 0.999
    
    # We need to be careful with the number of iterations
    # Estimate cost per iteration: O(|S|^2) ~ 2500 ops.
    # 100k iterations ~ 2.5e8 ops. Might be slow in Python.
    # Let's limit iterations based on time.
    
    while time.time() - start_time < time_limit and T > T_min:
        # Generate a neighbor
        # Randomly choose a move type
        move_type = random.randint(0, 2)
        new_S = set(S)
        
        if move_type == 0: # Add
            # Pick random x not in S
            x = random.randint(1, N)
            if x not in S:
                if is_valid_addition(new_S, x):
                    new_S.add(x)
                else:
                    new_S = None # Invalid move
        elif move_type == 1: # Remove
            if S:
                x = random.choice(list(S))
                new_S.remove(x)
            else:
                new_S = None
        else: # Swap
            if S:
                x = random.choice(list(S))
                y = random.randint(1, N)
                if y not in S:
                    # Check if S - {x} + {y} is valid
                    temp_S = set(S)
                    temp_S.remove(x)
                    if is_valid_addition(temp_S, y):
                        temp_S.add(y)
                        new_S = temp_S
                    else:
                        new_S = None
                else:
                    new_S = None
            else:
                new_S = None
        
        if new_S is None:
            T *= cooling_rate
            continue
            
        score_S = get_score(S)
        score_new = get_score(new_S)
        delta = score_new - score_S
        
        if delta > 0 or random.random() < math.exp(delta / T):
            S = new_S
            if score_new > best_score:
                best_score = score_new
                best_S = set(S)
        
        T *= cooling_rate
        
    return list(best_S)
