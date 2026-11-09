import time
import random
from math import gcd
from functools import reduce
from fractions import Fraction

def _lcm_range(n):
    result = 1
    for i in range(2, n + 1):
        result = result * i // gcd(result, i)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s safety margin before 90s kill
    
    n = 30
    lcm_val = _lcm_range(n)
    
    # Precompute coefficients: c[i] corresponds to 1/(i+1), so c[i-1]/i
    # We want sum_{i=1}^{30} c[i-1]/i where c[i-1] in {-1,0,1}
    # Equivalently, let a[i] = c[i-1] for i=1..30, a[i] in {-1,0,1}
    # Sum = sum_{i=1}^{30} a[i]/i
    
    # The seed solution: [0]*28 + [1,-1] means a[1..28]=0, a[29]=1, a[30]=-1
    # Sum = 1/29 - 1/30 = 1/870
    
    # Convert to integer arithmetic: multiply by lcm_val
    # S = sum_{i=1}^{30} a[i] * (lcm_val // i)
    # We want S != 0, minimize |S| (equivalently minimize |S|/lcm_val)
    
    weights = [lcm_val // i for i in range(1, n + 1)]
    
    # Seed solution
    best_a = [0] * n
    best_a[28] = 1  # a[29] = 1 (0-indexed: index 28)
    best_a[29] = -1  # a[30] = -1 (0-indexed: index 29)
    best_S = sum(best_a[i] * weights[i] for i in range(n))
    # |S| = lcm_val/29 - lcm_val/30 = lcm_val * (30-29)/(29*30) = lcm_val/870
    
    # Local search: flip one coefficient at a time
    def evaluate(a):
        S = 0
        for i in range(n):
            S += a[i] * weights[i]
        return S
    
    # Start from seed
    a = list(best_a)
    S = best_S
    
    # Try single flips
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        # Randomly try flipping a position
        pos = random.randint(0, n - 1)
        old_val = a[pos]
        
        if old_val == 0:
            # Try setting to 1 or -1
            for new_val in (1, -1):
                new_S = S + new_val * weights[pos]
                if new_S != 0 and abs(new_S) < abs(best_S):
                    a[pos] = new_val
                    S = new_S
                    best_S = S
                    best_a = list(a)
                    break
        elif old_val == 1:
            # Try setting to 0 or -1
            for new_val in (0, -1):
                new_S = S - weights[pos] + new_val * weights[pos]
                new_S = S + (new_val - 1) * weights[pos]
                if new_S != 0 and abs(new_S) < abs(best_S):
                    a[pos] = new_val
                    S = new_S
                    best_S = S
                    best_a = list(a)
                    break
        elif old_val == -1:
            # Try setting to 0 or 1
            for new_val in (0, 1):
                new_S = S + (new_val - (-1)) * weights[pos]
                if new_S != 0 and abs(new_S) < abs(best_S):
                    a[pos] = new_val
                    S = new_S
                    best_S = S
                    best_a = list(a)
                    break
    
    # Multi-start local search
    if time.time() < deadline - 10:
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random restart: start with a random sparse configuration
            a = [0] * n
            # Pick 2-4 random positions and assign random signs
            num_nonzero = random.randint(2, 4)
            positions = random.sample(range(n), num_nonzero)
            for p in positions:
                a[p] = random.choice([-1, 1])
            
            S = sum(a[i] * weights[i] for i in range(n))
            if S == 0:
                continue
            
            if abs(S) < abs(best_S):
                best_S = S
                best_a = list(a)
            
            # Local search from this starting point
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for pos in range(n):
                    if time.time() > deadline - 5:
                        break
                    old_val = a[pos]
                    candidates = [v for v in (-1, 0, 1) if v != old_val]
                    random.shuffle(candidates)
                    for new_val in candidates:
                        new_S = S + (new_val - old_val) * weights[pos]
                        if new_S != 0 and abs(new_S) < abs(S):
                            a[pos] = new_val
                            S = new_S
                            improved = True
                            break
            
            if abs(S) < abs(best_S):
                best_S = S
                best_a = list(a)
    
    # Final polish: try all single flips from best
    a = list(best_a)
    S = best_S
    for pos in range(n):
        if time.time() > deadline:
            break
        old_val = a[pos]
        for new_val in (-1, 0, 1):
            if new_val == old_val:
                continue
            new_S = S + (new_val - old_val) * weights[pos]
            if new_S != 0 and abs(new_S) < abs(S):
                a[pos] = new_val
                S = new_S
                best_S = S
                best_a = list(a)
    
    # Return the coefficients
    return best_a
