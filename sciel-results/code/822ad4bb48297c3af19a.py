import random
import time

def construct():
    N = 192
    # 85 seconds to be safe within the 90s limit
    deadline = time.time() + 85.0
    
    def can_add(S, x):
        """
        Returns True if S U {x} is a valid set, assuming S is valid.
        x must not be in S.
        """
        if x in S:
            return False
            
        S_list = list(S)
        n = len(S_list)
        
        # 1. Check if x divides the sum of any 3 distinct elements in S
        # If so, x (as divisor) violates the condition.
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                for k in range(j + 1, n):
                    c = S_list[k]
                    if (a + b + c) % x == 0:
                        return False

        # 2. Check if any a in S divides the sum of x and 2 distinct elements in S \ {a}
        # If so, a (as divisor) violates the condition with the new element x.
        for i in range(n):
            a = S_list[i]
            # We need distinct b, c from S \ {a}
            # such that (x + b + c) % a == 0
            for j in range(n):
                if j == i:
                    continue
                b = S_list[j]
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    c = S_list[k]
                    # b and c are distinct and not a
                    if (x + b + c) % a == 0:
                        return False
                            
        return True

    def generate_greedy_set(seed):
        random.seed(seed)
        S = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(S, x):
                S.add(x)
        return S

    # 1. Generate a good initial set
    best_init = set()
    
    # Try a few random greedy starts
    for i in range(20):
        S = generate_greedy_set(i)
        if len(S) > len(best_init):
            best_init = S
            
    # Try descending order (similar to seed but using fast check)
    S_desc = set()
    for x in range(N, 0, -1):
        if can_add(S_desc, x):
            S_desc.add(x)
    if len(S_desc) > len(best_init):
        best_init = S_desc
        
    # Try ascending order
    S_asc = set()
    for x in range(1, N + 1):
        if can_add(S_asc, x):
            S_asc.add(x)
    if len(S_asc) > len(best_init):
        best_init = S_asc

    # 2. Simulated Annealing
    current_set = set(best_init)
    best_set = set(best_init)
    
    T = 5.0
    cooling = 0.995
    
    # Precompute list of all numbers for random choice
    all_nums = list(range(1, N + 1))
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to Add
            # Pick a random number not in current_set
            # Optimization: keep a list of available numbers?
            # For N=192, just filtering is fast enough.
            available = [x for x in all_nums if x not in current_set]
            if not available:
                continue
            x = random.choice(available)
            
            if can_add(current_set, x):
                new_set = current_set | {x}
                # Size increased, always accept
                current_set = new_set
                if len(current_set) > len(best_set):
                    best_set = set(current_set)
                    
        elif move_type < 0.7:
            # Try to Remove
            if not current_set:
                continue
            x = random.choice(list(current_set))
            new_set = current_set - {x}
            
            # Accept removal with some probability to allow exploration
            # Energy delta is +1 (size decreases by 1)
            # prob = exp(-1/T)
            if random.random() < (2.71828 ** (-1.0 / T)):
                current_set = new_set
                
        else:
            # Try to Swap
            if not current_set:
                continue
            x = random.choice(list(current_set))
            available = [y for y in all_nums if y not in current_set]
            if not available:
                continue
            y = random.choice(available)
            
            # Check if (current_set - {x}) U {y} is valid
            S_reduced = current_set - {x}
            if can_add(S_reduced, y):
                new_set = S_reduced | {y}
                # Size same, accept to explore
                current_set = new_set
                if len(current_set) > len(best_set):
                    best_set = set(current_set)
                
        T *= cooling
        if T < 0.01:
            T = 0.01

    return list(best_set)
