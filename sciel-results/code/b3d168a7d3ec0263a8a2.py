import random
import math

def construct():
    # Initial construction: 4x8 grid with large spacing to avoid overlap
    # This yields ~18 distinct differences
    M = 100
    S_init = set()
    for i in range(4):
        for j in range(8):
            S_init.add(i * M + j + 1)
            
    # Ensure within 1..512
    # Max val: 3*100 + 8 + 1 = 309. OK.
    
    def score(S):
        S = set(S)
        diffs = set()
        # Optimization: for each pair (a, c), check if (a+c)/2 in S
        # d = (c-a)/2
        # This is O(N^2)
        list_S = list(S)
        for i in range(len(list_S)):
            for j in range(i+1, len(list_S)):
                a = list_S[i]
                c = list_S[j]
                if (a + c) % 2 == 0:
                    b = (a + c) // 2
                    if b in S:
                        d = c - a
                        if d > 0:
                            diffs.add(d)
        return len(diffs)
    
    current_S = list(S_init)
    current_score = score(current_S)
    best_S = current_S.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.99
    iterations = 5000
    
    for _ in range(iterations):
        # Pick a random element to move
        idx = random.randint(0, 31)
        old_val = current_S[idx]
        
        # Pick a new random value
        new_val = random.randint(1, 512)
        
        if new_val in current_S and new_val != old_val:
            continue
            
        current_S[idx] = new_val
        new_score = score(current_S)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_S = current_S.copy()
        else:
            current_S[idx] = old_val
            
        T *= cooling
        if T < 0.1:
            break
            
    return best_S
