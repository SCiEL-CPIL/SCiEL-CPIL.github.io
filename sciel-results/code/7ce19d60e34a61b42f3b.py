import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    def has_conflict(S, a):
        """Check if adding 'a' to S creates a reciprocal Schur triple with three distinct elements."""
        # 1/a = 1/u + 1/v where u,v in S, u != v, u,v != a, and a < u, a < v
        for u in S:
            if u <= a:
                continue
            numerator = u * a
            denominator = u - a
            if numerator % denominator == 0:
                v = numerator // denominator
                if v in S and v != u and v != a:
                    return True
        return False
    
    def greedy_with_order(order):
        """Run greedy with a given order of elements to consider."""
        S = set()
        for a in order:
            if time.time() - start_time > DEADLINE:
                break
            if not has_conflict(S, a):
                S.add(a)
        return S
    
    # Try deterministic descending first
    best_S = greedy_with_order(range(180, 0, -1))
    best_size = len(best_S)
    
    # Try multiple randomized restarts
    rng = random.Random(42)
    num_restarts = 500
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Randomize the order
        order = list(range(1, 181))
        rng.shuffle(order)
        
        S = greedy_with_order(order)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
    
    return list(best_S)
