import time
import random

def construct():
    start_time = time.time()
    timeout = 55.0
    
    # Initial candidate: Arithmetic Progression
    A = list(range(1, 21))
    
    def evaluate(A):
        sums = set()
        prods = set()
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))
    
    best_score = evaluate(A)
    current_score = best_score
    best_A = A[:]
    current_A = A[:]
    
    T = 5.0
    cooling = 0.999
    
    # To escape local minima, we can occasionally do a larger jump
    # Or just run for a long time
    
    while time.time() - start_time < timeout:
        # Pick a random index to perturb
        idx = random.randint(0, 19)
        old_val = current_A[idx]
        
        # Generate new value
        # Bias towards values that might reduce product set size?
        # Small numbers have more collisions in products?
        # Actually, small numbers in GP create many products.
        # But AP uses small numbers.
        
        # Let's try a random value in [1, 10000]
        new_val = random.randint(1, 10000)
        
        # Check distinctness
        # Using a set for current_A for O(1) lookup?
        # We maintain a set of current values for fast check
        # But updating the set on every step is O(1) if we remove old and add new.
        
        # Let's maintain a set `current_set`
        # Initialize once
        # ...
        
        # For simplicity in this snippet, I'll just check list membership O(n) which is 20.
        if new_val in current_A:
            continue
            
        current_A[idx] = new_val
        
        new_score = evaluate(current_A)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = current_A[:]
        else:
            delta = new_score - current_score
            if delta > 0:
                # Accept with probability
                # Use log to avoid underflow? delta is small (integer)
                # exp(-delta/T)
                if random.random() < (2.71828 ** (-delta / T)):
                    current_score = new_score
                else:
                    current_A[idx] = old_val
            # If delta == 0, accept (neutral move)
            # This helps explore the plateau
            
        T *= cooling
        
    return best_A
