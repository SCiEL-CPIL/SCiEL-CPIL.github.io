import math
import time
from itertools import product

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
    
    best_span = float('inf')
    best_values = None
    
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin
    
    # First pass: find best span with natural values
    candidate_ws = []
    for W in range(L):
        vals = [W + ((-W) % d) for d in divisors]
        if len(set(vals)) == n:
            span = max(vals) - min(vals)
            candidate_ws.append((span, W, vals))
        
        if time.time() - start_time > deadline:
            break
    
    # Sort by span, take top candidates for local polish
    candidate_ws.sort(key=lambda x: x[0])
    
    # Local polish: for top K candidates, try all 3^n shift combinations
    # 3^10 = 59049, which is fast. Do it for top 100 W values.
    K = 100
    for span0, W, base_vals in candidate_ws[:K]:
        if span0 >= best_span:
            break
        
        # Try all combinations of shifts {-1, 0, 1} for each element
        # shift -1: v_i - d_i, shift 0: v_i, shift +1: v_i + d_i
        for shifts in product([-1, 0, 1], repeat=n):
            test_vals = [base_vals[i] + shifts[i] * divisors[i] for i in range(n)]
            
            # Check positivity and bounds
            if any(v <= 0 or v > 1000000 for v in test_vals):
                continue
            
            # Check distinctness
            if len(set(test_vals)) != n:
                continue
            
            span = max(test_vals) - min(test_vals)
            if span < best_span:
                best_span = span
                best_values = test_vals[:]
        
        if time.time() - start_time > deadline:
            break
    
    # If no valid solution found (shouldn't happen), fallback
    if best_values is None:
        # Try a simple constructive approach
        # Use the divisors themselves if they're distinct
        if len(set(divisors)) == n:
            best_values = divisors[:]
        else:
            # Fallback: use multiples that are distinct
            best_values = [d * (i + 1) for i, d in enumerate(divisors)]
    
    return [int(v) for v in best_values]

if __name__ == '__main__':
    res = construct()
    print(res)
    print("Span:", max(res) - min(res))
