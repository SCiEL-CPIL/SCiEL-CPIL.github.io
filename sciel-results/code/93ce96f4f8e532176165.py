import random
import math
import time

def construct():
    N = 256
    # Precompute factors and compatibility
    factors = [set() for _ in range(N+1)]
    for i in range(2, N+1):
        x = i
        p = 2
        while p * p <= x:
            if x % p == 0:
                factors[i].add(p)
                while x % p == 0:
                    x //= p
            p += 1
        if x > 1:
            factors[i].add(x)
    factors[1] = set()
    
    # Compatibility matrix: 1 if coprime, 0 otherwise
    # Using a list of lists of bools
    compat = [[True]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            if i == j:
                compat[i][j] = False # Can't have duplicates
            else:
                # Check intersection
                if factors[i] & factors[j]:
                    compat[i][j] = False
                    compat[j][i] = False
                else:
                    compat[i][j] = True
                    compat[j][i] = True
                    
    def is_valid_swap(S, a, b):
        # S is a set
        # Check if b is coprime to all x in S \ {a}
        for x in S:
            if x == a:
                continue
            if not compat[x][b]:
                return False
        return True

    def run_sa(initial_S, steps=5000):
        S = set(initial_S)
        # Ensure 1 is in S? 1 is always good.
        # If 1 not in S, add it if valid (it always is).
        if 1 not in S:
            S.add(1)
            
        current_sum = sum(S)
        best_S = set(S)
        best_sum = current_sum
        
        T = 10.0
        cooling = 0.99
        
        for i in range(steps):
            if len(S) == 0:
                S.add(1)
                current_sum = 1
                continue
                
            # Pick random a in S
            a = random.choice(list(S))
            # Pick random b not in S
            # To speed up, pick from a pre-generated list of candidates?
            # Or just random 1..N and retry if in S.
            b = random.randint(1, N)
            while b in S:
                b = random.randint(1, N)
                
            if not is_valid_swap(S, a, b):
                continue
                
            delta = b - a
            if delta > 0:
                # Always accept
                S.remove(a)
                S.add(b)
                current_sum += delta
            else:
                # Accept with probability
                if random.random() < math.exp(delta / T):
                    S.remove(a)
                    S.add(b)
                    current_sum += delta
                    
            if current_sum > best_sum:
                best_sum = current_sum
                best_S = set(S)
                
            T *= cooling
            if T < 0.1:
                break
                
        return best_S, best_sum

    best_overall_S = None
    best_overall_sum = 0
    
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    # Initialization strategies
    # 1. Greedy Descending
    S_init = {1}
    for n in range(N, 1, -1):
        valid = True
        for x in S_init:
            if not compat[x][n]:
                valid = False
                break
        if valid:
            S_init.add(n)
            
    # Run SA from this init
    S_res, sum_res = run_sa(S_init, steps=10000)
    if sum_res > best_overall_sum:
        best_overall_sum = sum_res
        best_overall_S = S_res
        
    # 2. Random Restarts
    while time.time() < deadline:
        # Random Init
        S_init = {1}
        # Add random numbers
        candidates = list(range(2, N+1))
        random.shuffle(candidates)
        for n in candidates:
            if random.random() < 0.1: # Sparse init?
                valid = True
                for x in S_init:
                    if not compat[x][n]:
                        valid = False
                        break
                if valid:
                    S_init.add(n)
                    
        S_res, sum_res = run_sa(S_init, steps=5000)
        if sum_res > best_overall_sum:
            best_overall_sum = sum_res
            best_overall_S = S_res
            
    if best_overall_S is None:
        return [1]
        
    return list(best_overall_S)
