import sys
import random
import math

def construct():
    n = 256
    # 1. Generate initial greedy solution
    S = set()
    for x in range(1, n + 1):
        valid = True
        # Check if x forms a 3-AP with any pair in S
        # x is the largest, so x = 2b - a => a = 2b - x
        # Or x is middle: a, x, b => b = 2x - a
        # Or x is smallest: x, a, b => b = 2a - x
        # Since we add x to S, we must ensure no AP in S U {x}.
        # Existing S is AP-free. So we only check APs involving x.
        
        # Case 1: x is 3rd term. a, b, x. 2b = a + x => a = 2b - x.
        # b in S. a must be in S.
        for b in S:
            a = 2 * b - x
            if a in S and a < b:
                valid = False
                break
        
        if valid:
            # Case 2: x is 2nd term. a, x, b. 2x = a + b => b = 2x - a.
            # a in S. b must be in S.
            for a in S:
                b = 2 * x - a
                if b in S and b > x:
                    valid = False
                    break
        
        if valid:
            # Case 3: x is 1st term. x, a, b. 2a = x + b => b = 2a - x.
            # a in S. b must be in S.
            for a in S:
                b = 2 * a - x
                if b in S and b > a:
                    valid = False
                    break
        
        if valid:
            S.add(x)
    
    S_list = sorted(S)
    
    def get_weight(S):
        return sum(1.0 / x for x in S)
    
    def is_valid_addition(S, x):
        # Check if S U {x} is 3-AP free.
        # S is assumed 3-AP free.
        # Check APs involving x.
        # 1. x, a, b => b = 2a - x. a in S, b in S, b > a.
        for a in S:
            b = 2 * a - x
            if b in S and b > a:
                return False
        # 2. a, x, b => b = 2x - a. a in S, b in S, b > x.
        for a in S:
            b = 2 * x - a
            if b in S and b > x:
                return False
        # 3. a, b, x => a = 2b - x. b in S, a in S, a < b.
        for b in S:
            a = 2 * b - x
            if a in S and a < b:
                return False
        return True

    def is_valid_swap(S, x, y):
        # S' = S - {y} + {x}
        # Check if S' is 3-AP free.
        # S - {y} is 3-AP free.
        # We only need to check APs involving x in S'.
        # Elements in S' are S \ {y} U {x}.
        
        # 1. x, a, b => b = 2a - x. a in S', b in S', b > a.
        for a in S:
            if a == y: continue
            b = 2 * a - x
            # b must be in S'. b != x (since b > a >= 1, x < y, but b could be x? No, b=2a-x. If b=x, 2a=2x, a=x. But a != x in S? x not in S. So a != x. So b != x unless a=x which is not in S. Wait, if a is in S, a != x. So b != x.
            # So b must be in S and b != y.
            if b in S and b != y and b > a:
                return False
            # Also check if b == x? Impossible as shown.
            
        # 2. a, x, b => b = 2x - a. a in S', b in S', b > x.
        for a in S:
            if a == y: continue
            b = 2 * x - a
            if b in S and b != y and b > x:
                return False
            
        # 3. a, b, x => a = 2b - x. b in S', a in S', a < b.
        for b in S:
            if b == y: continue
            a = 2 * b - x
            if a in S and a != y and a < b:
                return False
                
        return True

    current_weight = get_weight(S)
    
    # Local Search: Best 1-Swap
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_x = None
        best_y = None
        
        rejected = [x for x in range(1, n + 1) if x not in S]
        
        for x in rejected:
            # Only swap if x < y to potentially increase weight?
            # 1/x - 1/y > 0 iff x < y.
            for y in S:
                if y <= x:
                    continue
                if is_valid_swap(S, x, y):
                    gain = (1.0 / x) - (1.0 / y)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_y = y
        
        if best_x is not None:
            S.remove(best_y)
            S.add(best_x)
            current_weight += best_gain
            improved = True
            
    # Simulated Annealing to escape local minima
    # We will try random swaps
    T = 0.05
    cooling = 0.99
    min_T = 0.001
    iterations = 0
    max_iterations = 50000 # Limit to ensure time safety
    
    while T > min_T and iterations < max_iterations:
        # Pick random x not in S
        rejected = [x for x in range(1, n + 1) if x not in S]
        if not rejected:
            break
        x = random.choice(rejected)
        
        # Pick random y in S such that y > x (to have potential gain)
        # If we pick y < x, gain is negative, which is a "downhill" move.
        # Let's allow any y in S to explore more.
        y = random.choice(list(S))
        
        if x == y:
            continue
            
        if is_valid_swap(S, x, y):
            gain = (1.0 / x) - (1.0 / y)
            if gain > 0 or random.random() < math.exp(gain / T):
                S.remove(y)
                S.add(x)
                current_weight += gain
                T *= cooling
        iterations += 1
        
    # Final local search cleanup
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_x = None
        best_y = None
        
        rejected = [x for x in range(1, n + 1) if x not in S]
        
        for x in rejected:
            for y in S:
                if y <= x:
                    continue
                if is_valid_swap(S, x, y):
                    gain = (1.0 / x) - (1.0 / y)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_y = y
        
        if best_x is not None:
            S.remove(best_y)
            S.add(best_x)
            current_weight += best_gain
            improved = True
            
    return sorted(S)
