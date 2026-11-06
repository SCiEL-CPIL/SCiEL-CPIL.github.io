import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5 second safety margin
    
    n = 20
    max_val = 10000
    
    def get_score(A):
        sums = set()
        prods = set()
        for i in range(n):
            ai = A[i]
            for j in range(n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))

    # Initial candidate: Arithmetic Progression 1..20
    best_A = list(range(1, n + 1))
    best_score = get_score(best_A)
    
    # Try a few different initializations if time permits
    inits = [
        list(range(1, n + 1)),
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 9000, 9200, 9400, 9600, 9800, 9999], # GP + random tail
        [random.randint(1, max_val) for _ in range(n)]
    ]
    # Ensure distinctness for random init
    for k, init in enumerate(inits):
        if time.time() > deadline:
            break
        # Clean init
        unique_init = list(set(init))
        while len(unique_init) < n:
            unique_init.append(random.randint(1, max_val))
        unique_init = unique_init[:n]
        if not all(1 <= x <= max_val for x in unique_init):
            continue
            
        curr_A = unique_init.copy()
        curr_score = get_score(curr_A)
        
        if curr_score < best_score:
            best_score = curr_score
            best_A = curr_A.copy()
            
        # Simulated Annealing for this start
        T = 5.0
        cool_rate = 0.999
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < deadline:
            i = random.randint(0, n - 1)
            old_val = curr_A[i]
            
            # Strategy: 80% local move, 20% global move
            if random.random() < 0.8:
                # Local move: shift by small amount
                step = random.choice([-10, -5, -2, -1, 1, 2, 5, 10, 20, 50])
                new_val = old_val + step
            else:
                # Global move
                new_val = random.randint(1, max_val)
                
            if new_val < 1 or new_val > max_val:
                continue
            if new_val in curr_A:
                continue
                
            curr_A[i] = new_val
            new_score = get_score(curr_A)
            delta = new_score - curr_score
            
            if delta < 0 or (T > 0.01 and random.random() < math.exp(-delta / T)):
                curr_score = new_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_A = curr_A.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                curr_A[i] = old_val
                no_improve_count += 1
                
            T *= cool_rate
            
            if no_improve_count > max_no_improve:
                break

    return best_A
