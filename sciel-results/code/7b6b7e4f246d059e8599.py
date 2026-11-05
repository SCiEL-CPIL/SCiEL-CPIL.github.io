import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    best_set = []
    best_size = 0
    
    def can_add(x, S):
        n = len(S)
        # Check 1: x does not divide sum of any 3 distinct elements in S
        if n >= 3:
            # Unroll loops for speed? No, Python is slow.
            # But n is small.
            for i in range(n):
                a = S[i]
                for j in range(i+1, n):
                    b = S[j]
                    for k in range(j+1, n):
                        c = S[k]
                        if (a + b + c) % x == 0:
                            return False
        
        # Check 2: No y in S divides (x + a + b) for any distinct a,b in S\{y}
        if n >= 2:
            for y in S:
                # Construct list of others
                # Optimization: If y is small, (x+a+b) % y == 0 is likely.
                # We can check residues.
                others = []
                for v in S:
                    if v != y:
                        others.append(v)
                
                # If we have at least 2 others
                if len(others) >= 2:
                    # Check all pairs
                    m = len(others)
                    for i in range(m):
                        a = others[i]
                        for j in range(i+1, m):
                            b = others[j]
                            if (x + a + b) % y == 0:
                                return False
        return True

    def run_greedy(order):
        S = []
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.append(x)
        return S

    # 1. Deterministic Descending
    S1 = run_greedy(range(192, 0, -1))
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1

    # 2. Deterministic Ascending (unlikely to be good, but fast to check)
    S2 = run_greedy(range(1, 193))
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = S2

    # 3. Randomized Restart with Bias
    # We want to explore different structures.
    # Bias towards larger numbers helps validity.
    
    random.seed(42) # For reproducibility in dev, remove in prod? No, evaluator uses time/random.
    
    iterations = 0
    max_iterations = 200 # Limit by time
    
    while time.time() < deadline - 5.0:
        # Create biased random order
        # Larger numbers should appear earlier.
        # Key: -x + noise
        noise_scale = 50
        order = list(range(1, 193))
        random.shuffle(order)
        # Sort by a score that favors larger x
        # If we just shuffle, we lose the bias.
        # Let's generate a random weight for each number, but subtract x * weight_factor
        # Or simpler: Sort by (random.random() - x/200)
        # This makes larger x have smaller key (earlier)
        
        # Let's try a few different biases
        if iterations % 2 == 0:
            # Strong bias to large
            order.sort(key=lambda x: random.random() * 10 - x)
        else:
            # Weak bias to large
            order.sort(key=lambda x: random.random() * 50 - x)
            
        S_rand = run_greedy(order)
        if len(S_rand) > best_size:
            best_size = len(S_rand)
            best_set = S_rand
            
        # Early stop if we hit the max possible
        if best_size >= 48:
            break
            
        iterations += 1
        
    return best_set
