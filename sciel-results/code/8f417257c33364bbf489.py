import random
import time
import math

def construct():
    max_n = 512
    max_size = 48
    
    def is_valid_subset_sum(values, target, min_size, max_size):
        M = len(values)
        if min_size > M:
            return False
        if max_size > M:
            max_size = M
            
        max_abs_sum = max_size * max([abs(v) for v in values] + [1])
        offset = max_abs_sum + 1
        target_bit_idx = offset + target
        
        if target_bit_idx < 0 or target_bit_idx > 2 * offset:
            return False

        dp = [0] * (M + 1)
        dp[0] = 1 << offset
        
        for v in values:
            for k in range(M - 1, -1, -1):
                if dp[k]:
                    if v >= 0:
                        dp[k+1] |= (dp[k] << v)
                    else:
                        dp[k+1] |= (dp[k] >> (-v))
                        
        for k in range(min_size, max_size + 1):
            if dp[k] & (1 << target_bit_idx):
                return True
        return False

    def is_valid_set(S):
        # Check if S is a valid set
        if len(S) < 2:
            return True
        S_set = set(S)
        for x in S:
            # Check if x is mean of subset of S \ {x}
            others = [t for t in S if t != x]
            if len(others) >= 2:
                values = [t - x for t in others]
                if is_valid_subset_sum(values, 0, 2, len(others)):
                    return False
        return True

    def score(S):
        if is_valid_set(S):
            return len(S)
        else:
            return -1000 + len(S)

    # Simulated Annealing
    best_S = []
    best_score = -1000
    
    start_time = time.time()
    time_limit = 55
    
    # Initialize with a random valid set or empty
    # Start with greedy to get a good start
    def greedy_init():
        C = []
        rand_order = list(range(1, max_n + 1))
        random.shuffle(rand_order)
        for x in rand_order:
            if len(C) >= max_size:
                break
            # Fast check
            # We can reuse is_valid_set logic but incremental?
            # For init, just use full check on C + [x]
            # To speed up, we can just check if x is valid with C
            # and if C remains valid (it does, as x is new)
            # Wait, adding x might break validity of existing elements?
            # Yes. So we must check full validity of C U {x}
            # Or use the incremental check from before.
            # Let's use the incremental check for speed.
            
            # Incremental Check for x in C
            # 1. x mean of subset of C?
            if len(C) >= 2:
                vals = [t - x for t in C]
                if is_valid_subset_sum(vals, 0, 2, len(C)):
                    continue
            # 2. a in C mean of subset involving x?
            valid = True
            for a in C:
                others = [t for t in C if t != a]
                if not others: continue
                vals = [t - a for t in others]
                target = a - x
                if is_valid_subset_sum(vals, target, 1, len(others)):
                    valid = False
                    break
            if valid:
                C.append(x)
        return C

    initial_S = greedy_init()
    best_S = initial_S
    best_score = score(best_S)
    
    current_S = list(initial_S)
    current_score = best_score
    T = 100.0
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        if T < 0.1:
            break
            
        # Generate a move
        move_type = random.random()
        new_S = list(current_S)
        
        if move_type < 0.5:
            # Add
            candidates = [x for x in range(1, max_n + 1) if x not in current_S]
            if candidates:
                x = random.choice(candidates)
                new_S.append(x)
        elif move_type < 0.75:
            # Remove
            if current_S:
                x = random.choice(current_S)
                new_S.remove(x)
        else:
            # Swap
            if current_S:
                x = random.choice(current_S)
                candidates = [y for y in range(1, max_n + 1) if y not in current_S]
                if candidates:
                    y = random.choice(candidates)
                    new_S.remove(x)
                    new_S.append(y)
                    
        new_score = score(new_S)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_S = new_S
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_S = list(current_S)
                
        T *= cooling
        
    return best_S
