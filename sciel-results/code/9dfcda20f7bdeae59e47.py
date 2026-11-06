import random
import time

def construct():
    N = 256
    
    def can_add(x, S):
        """
        Check if x can be added to S without creating a 3-AP.
        S is a set of integers.
        """
        for d in range(1, N + 1):
            # x is middle: x-d, x, x+d
            a = x - d
            b = x + d
            if a >= 1 and b <= N:
                if a in S and b in S:
                    return False
            
            # x is start: x, x+d, x+2d
            b = x + d
            c = x + 2 * d
            if c <= N:
                if b in S and c in S:
                    return False
            
            # x is end: x-2d, x-d, x
            a = x - 2 * d
            b = x - d
            if a >= 1:
                if a in S and b in S:
                    return False
            
            # Optimization: break if d is too large
            if d > N:
                break
        return True

    def is_3ap_free(S):
        S_set = set(S)
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                # Case 1: a, b, c (b is middle) -> c = 2b - a
                c = 2 * b - a
                if c in S_set:
                    return False
                # Case 2: b, a, c (a is middle) -> c = 2a - b
                c = 2 * a - b
                if c in S_set:
                    return False
                # Case 3: a, c, b (c is middle) -> c = (a+b)/2
                if (a + b) % 2 == 0:
                    c = (a + b) // 2
                    if c in S_set:
                        return False
        return True

    # 1. Greedy Initialization
    S = set()
    for x in range(1, N + 1):
        if can_add(x, S):
            S.add(x)
    
    best_S = set(S)
    best_score = sum(1.0 / x for x in best_S)
    
    start_time = time.time()
    time_limit = 85.0  # Safe margin from 90s
    
    # 2. Local Search
    current_S = set(best_S)
    current_score = best_score
    
    while time.time() - start_time < time_limit:
        move_type = random.random()
        
        if move_type < 0.3:
            # Add
            x = random.randint(1, N)
            if x not in current_S:
                if can_add(x, current_S):
                    current_S.add(x)
                    current_score += 1.0 / x
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(current_S)
                        
        elif move_type < 0.6:
            # Remove
            if current_S:
                x = random.choice(list(current_S))
                current_S.remove(x)
                current_score -= 1.0 / x
                
        else:
            # Swap
            if current_S:
                x = random.choice(list(current_S))
                y = random.randint(1, N)
                if y not in current_S and y != x:
                    current_S.remove(x)
                    if can_add(y, current_S):
                        current_S.add(y)
                        delta = 1.0 / y - 1.0 / x
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_S = set(current_S)
                    else:
                        current_S.add(x)
                        # current_score unchanged
                        pass
                        
    # Final Verification
    if not is_3ap_free(best_S):
         # Fallback: Greedy set
         S = set()
         for x in range(1, N + 1):
             if can_add(x, S):
                 S.add(x)
         best_S = S
         
    return list(best_S)
