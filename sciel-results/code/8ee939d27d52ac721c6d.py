import time
import random

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    def evaluate(S):
        S_set = set(S)
        valid_diffs = set()
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                d = b - a
                c = 2 * b - a
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    # Initial candidates
    candidates = []
    candidates.append(list(range(1, 33)))
    
    # Generate some random initial sets
    for _ in range(10):
        candidates.append(random.sample(range(1, 513), 32))
        
    # Find best initial
    best_S = max(candidates, key=evaluate)
    best_score = evaluate(best_S)
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    current_S = best_S[:]
    current_score = best_score
    
    while time.time() - start_time < timeout:
        if T < 0.1:
            break
            
        # Random move: pick a random index and assign a new random value
        i = random.randint(0, 31)
        old_val = current_S[i]
        
        # Generate new value not in current_S
        # To avoid infinite loop if 512 is small, we can just pick from a precomputed list of available values
        # But 512-32 = 480 available.
        # We can maintain a set of available values for faster sampling?
        # Or just pick randomly and check.
        
        # Optimization: Precompute available values?
        # Since N is small, just picking randomly is fine.
        
        new_val = random.randint(1, 512)
        if new_val in current_S:
            continue
            
        current_S[i] = new_val
        new_score = evaluate(current_S)
        
        if new_score >= current_score or random.random() < pow(2.718281828, (new_score - current_score) / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_S = current_S[:]
        else:
            current_S[i] = old_val
            
        T *= cooling
        
    return best_S
