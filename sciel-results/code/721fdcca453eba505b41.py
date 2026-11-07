import random
import time

def construct():
    N = 128
    
    def is_valid_addition_bit(S_bit, x):
        # Condition 1: x in {a,b,c}
        # b, c in S_old, b != c
        # x+b in S_old, x+c in S_old
        # b+c in S_old OR b+c == x
        
        B = []
        for b in range(1, N - x + 1):
            if (S_bit >> b) & 1 and (S_bit >> (x + b)) & 1:
                B.append(b)
                
        if len(B) >= 2:
            for i in range(len(B)):
                b = B[i]
                for j in range(i+1, len(B)):
                    c = B[j]
                    # Check b+c in S_old or b+c == x
                    if (S_bit >> (b + c)) & 1 or (b + c == x):
                        return False
        
        # Condition 2: x is a sum a+b
        # a, b, c in S_old, distinct
        # a+b == x
        # a+c in S_old OR a+c == x
        # b+c in S_old OR b+c == x
        
        A = []
        for a in range(1, x):
            b = x - a
            if b > 0 and b != a and (S_bit >> a) & 1 and (S_bit >> b) & 1:
                A.append((a, b))
        
        if A:
            for a, b in A:
                # Check if exists c in S_old \ {a,b}
                max_c = N - max(a, b)
                for c in range(1, max_c + 1):
                    if c == a or c == b:
                        continue
                    if not ((S_bit >> c) & 1):
                        continue
                    
                    # Check a+c in S_old or a+c == x
                    cond_ac = ((S_bit >> (a + c)) & 1) or (a + c == x)
                    # Check b+c in S_old or b+c == x
                    cond_bc = ((S_bit >> (b + c)) & 1) or (b + c == x)
                    
                    if cond_ac and cond_bc:
                        return False
        return True

    def greedy_construct_bit(order):
        S_bit = 0
        for x in order:
            if not (S_bit >> x) & 1:
                if is_valid_addition_bit(S_bit, x):
                    S_bit |= (1 << x)
        return S_bit

    def bitset_to_list(S_bit):
        return [i for i in range(1, N + 1) if (S_bit >> i) & 1]

    # Initial best
    S_init = greedy_construct_bit(range(N, 0, -1))
    best_list = bitset_to_list(S_init)
    best_size = len(best_list)
    
    start_time = time.time()
    timeout = 55.0
    
    while time.time() - start_time < timeout:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        S_bit = greedy_construct_bit(random_order)
        
        # Local Search: Add if possible
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if not (S_bit >> x) & 1:
                    if is_valid_addition_bit(S_bit, x):
                        S_bit |= (1 << x)
                        improved = True
                        break
        
        current_list = bitset_to_list(S_bit)
        if len(current_list) > best_size:
            best_list = current_list
            best_size = len(current_list)
            
    return best_list

