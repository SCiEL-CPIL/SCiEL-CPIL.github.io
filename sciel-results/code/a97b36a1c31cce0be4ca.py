import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 256
    best_size = 0
    best_A = []
    best_B = []
    
    def is_valid_addition(x, S, sums_S, D_S, D_other):
        # Check Sidon condition for S + {x}
        for s in S:
            if (x + s) in sums_S:
                return False
        if (2 * x) in sums_S:
            return False
            
        # Check difference collisions
        for s in S:
            d = x - s
            if d in D_S or d in D_other:
                return False
            
        return True

    def greedy_construct(order):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        D_A = set()
        D_B = set()
        
        for x in order:
            valid_A = is_valid_addition(x, A, sums_A, D_A, D_B)
            valid_B = is_valid_addition(x, B, sums_B, D_B, D_A)
            
            if valid_A and valid_B:
                if len(A) < len(B):
                    choice = 'A'
                elif len(B) < len(A):
                    choice = 'B'
                else:
                    choice = random.choice(['A', 'B'])
            elif valid_A:
                choice = 'A'
            elif valid_B:
                choice = 'B'
            else:
                continue
                
            if choice == 'A':
                for s in A:
                    sums_A.add(x + s)
                    D_A.add(x - s)
                    D_A.add(s - x)
                sums_A.add(2 * x)
                A.append(x)
            else:
                for s in B:
                    sums_B.add(x + s)
                    D_B.add(x - s)
                    D_B.add(s - x)
                sums_B.add(2 * x)
                B.append(x)
        
        return A, B

    # Initial deterministic run
    base_order = list(range(1, n + 1))
    A0, B0 = greedy_construct(base_order)
    best_size = len(A0) + len(B0)
    best_A, best_B = A0, B0
    
    # Randomized restarts
    while time.time() < deadline:
        random_order = list(range(1, n + 1))
        random.shuffle(random_order)
        A, B = greedy_construct(random_order)
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A, best_B = A, B
            
    return [best_A, best_B]
