import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    
    N = 256
    # Weights
    weights = [0] + [1.0/i for i in range(1, N+1)]
    
    # Greedy Initialization
    greedy_set = set()
    for x in range(1, N+1):
        valid = True
        for a in greedy_set:
            for b in greedy_set:
                if a == b: continue
                # Check if x, a, b form 3-AP
                # x is 1st: 2a - b == x
                # x is 2nd: a + b == 2x
                # x is 3rd: 2b - a == x
                # Since we iterate x increasing, and a,b < x? No, greedy_set contains < x.
                # So a, b < x.
                # Possible APs involving x and two smaller numbers a, b:
                # 1. a, b, x -> 2b - a = x
                # 2. a, x, b -> impossible since x > a,b? No, x is the largest candidate.
                #    Wait, if a, b < x, then x cannot be the middle term of a,b,x unless a,b > x? No.
                #    If a, b < x, the only way x is in a 3-AP with them is if x is the largest term.
                #    So only check 2b - a == x.
                if 2*b - a == x:
                    valid = False
                    break
            if not valid: break
        if valid:
            greedy_set.add(x)
            
    best_set = greedy_set
    best_score = sum(weights[x] for x in greedy_set)
    
    # Branch and Bound
    # We will try to improve upon the greedy solution.
    # The greedy solution includes small numbers.
    # We can try to exclude some small numbers to include more medium numbers?
    # Unlikely to be better due to convexity, but possible.
    
    # To make B&B efficient, we need a good upper bound.
    # Upper bound at step i (considering numbers i..N):
    # current_sum + sum(weights[i..N])
    # This is very loose.
    
    # Better approach: Local Search on the Greedy Solution.
    # Try to swap elements.
    
    def is_valid_set(S):
        S_list = list(S)
        S_set = set(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                # Check 3rd term
                c1 = 2*b - a
                if c1 in S_set and c1 != a and c1 != b: return False
                c2 = 2*a - b
                if c2 in S_set and c2 != a and c2 != b: return False
                if (a+b) % 2 == 0:
                    c3 = (a+b)//2
                    if c3 in S_set and c3 != a and c3 != b: return False
        return True
        
    def score(S):
        return sum(weights[x] for x in S)

    current_set = set(greedy_set)
    current_score = best_score
    
    # Simple Hill Climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all possible single additions
        for x in range(1, N+1):
            if x in current_set: continue
            new_set = current_set | {x}
            if is_valid_set(new_set):
                new_score = current_score + weights[x]
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    improved = True
        # If no addition, try swaps
        if not improved:
            for y in list(current_set):
                for x in range(1, N+1):
                    if x in current_set or x == y: continue
                    if weights[x] <= weights[y]: continue # Only swap if gain
                    new_set = (current_set - {y}) | {x}
                    if is_valid_set(new_set):
                        new_score = current_score - weights[y] + weights[x]
                        if new_score > current_score:
                            current_set = new_set
                            current_score = new_score
                            improved = True
                            break
                if improved: break
                if time.time() > deadline: break
            if time.time() > deadline: break
            
    if current_score > best_score:
        best_set = current_set
        best_score = current_score
        
    return list(best_set)
