import random
import math
import time

def construct():
    # Generate 7-smooth numbers up to 10000
    # 7-smooth numbers are numbers of the form 2^a * 3^b * 5^c * 7^d
    smooth_numbers = set([1])
    max_val = 10000
    
    # Generate all 7-smooth numbers
    # We can do this by iterating through exponents or using a priority queue.
    # Given the limit 10000, the number of such numbers is small (~hundreds).
    # Let's just generate them by brute force checking factors or generation.
    # Generation:
    candidates = [1]
    for a in range(15): # 2^14 > 10000
        for b in range(10): # 3^9 > 10000
            for c in range(7): # 5^6 > 10000
                for d in range(5): # 7^4 > 10000
                    val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if val <= max_val:
                        candidates.append(val)
    
    candidates = sorted(list(set(candidates)))
    
    # Initial set: 1 to 20
    # This set has very small sumset (39) and moderate productset.
    initial_set = list(range(1, 21))
    
    def evaluate(A):
        A_set = set(A)
        sum_set = set()
        prod_set = set()
        n = len(A)
        for i in range(n):
            for j in range(i, n):
                sum_set.add(A[i] + A[j])
                prod_set.add(A[i] * A[j])
        return max(len(sum_set), len(prod_set))
        
    current_A = initial_set[:]
    current_score = evaluate(current_A)
    
    best_A = current_A[:]
    best_score = current_score
    
    # Simulated Annealing
    start_time = time.time()
    timeout = 55 # 5s buffer
    
    # Candidate pool: 7-smooth numbers + small integers 1..50
    # We want to explore values that might create collisions.
    # 7-smooth numbers are good for product collisions.
    # Small integers are good for sum collisions.
    candidate_pool = list(set(candidates + list(range(1, 51))))
    
    T = 10.0
    cooling_rate = 0.999
    
    while time.time() - start_time < timeout:
        # Pick a random index
        idx = random.randint(0, 19)
        old_val = current_A[idx]
        
        # Pick a new value
        new_val = random.choice(candidate_pool)
        
        if new_val == old_val:
            continue
        if new_val in current_A:
            continue
            
        new_A = current_A[:]
        new_A[idx] = new_val
        
        new_score = evaluate(new_A)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_A = new_A
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
                
        T *= cooling_rate
        if T < 0.1:
            T = 1.0 # Restart cooling if it gets too low, or just keep going
            # Actually, standard SA cools down. Let's just let it cool.
            # But we have a time limit, so we can just run until timeout.
            
    return best_A
