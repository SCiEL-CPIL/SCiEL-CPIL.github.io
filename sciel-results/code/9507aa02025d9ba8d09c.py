import time
import random
import math

def compute_metrics(A):
    """
    Compute sizes of A+A and A*A.
    A is a list of distinct positive integers.
    """
    n = len(A)
    sums = set()
    prods = set()
    # Pre-sort to potentially help with early termination? No, sets don't care.
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return len(sums), len(prods)

def objective(A):
    s, p = compute_metrics(A)
    return max(s, p)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Validate and optimize a candidate
    def try_candidate(A):
        # Check constraints
        if len(A) != 20:
            return False
        if len(set(A)) != 20:
            return False
        if any(x < 1 or x > 10000 for x in A):
            return False
        return True

    # Generate initial candidates
    candidates = []
    
    # 1. Arithmetic Progression
    candidates.append(list(range(1, 21)))
    
    # 2. Geometric Progression (base 1.5)
    gp = []
    v = 1
    while len(gp) < 20 and v <= 10000:
        if v not in gp:
            gp.append(v)
        v = int(v * 1.5) + 1 # +1 to ensure growth if integer rounding stalls
        # Actually, better to use a float and floor
    # Let's do a proper GP generation
    gp = []
    r = 1.5
    v = 1.0
    while len(gp) < 20:
        iv = int(v)
        if iv > 10000:
            break
        if iv not in gp and iv >= 1:
            gp.append(iv)
        v = v * r
    # If we didn't get 20, fill with random
    while len(gp) < 20:
        x = random.randint(1, 10000)
        if x not in gp:
            gp.append(x)
    candidates.append(gp)
    
    # 3. Another GP base 1.2
    gp2 = []
    r = 1.2
    v = 1.0
    while len(gp2) < 20:
        iv = int(v)
        if iv > 10000:
            break
        if iv not in gp2 and iv >= 1:
            gp2.append(iv)
        v = v * r
    while len(gp2) < 20:
        x = random.randint(1, 10000)
        if x not in gp2:
            gp2.append(x)
    candidates.append(gp2)
    
    # 4. Random sets
    for _ in range(10):
        rs = random.sample(range(1, 10001), 20)
        candidates.append(rs)

    best_A = None
    best_obj = float('inf')
    
    # Evaluate initial candidates
    for A in candidates:
        if try_candidate(A):
            obj = objective(A)
            if obj < best_obj:
                best_obj = obj
                best_A = A[:]
                
    # Simulated Annealing
    # Start from the best initial candidate
    A = best_A[:]
    cur_obj = best_obj
    
    # Temperature parameters
    T_start = 100.0
    T_end = 1.0
    cooling_rate = 0.999
    
    # We will run SA for a fixed number of iterations or until time limit
    # Estimate iterations: objective takes ~0.1ms? 
    # 20*20/2 = 200 ops. Python is slow. ~0.5ms per eval.
    # 50s / 0.0005s = 100,000 iterations.
    
    T = T_start
    it = 0
    while time.time() < deadline:
        if T < T_end:
            break
            
        it += 1
        # Generate neighbor
        A_new = A[:]
        op = random.random()
        if op < 0.7:
            # Replace one element
            i = random.randrange(20)
            # Propose new value
            # Mix of local and global moves
            if random.random() < 0.5:
                # Local move: perturb current value
                delta = random.randint(-50, 50)
                new_val = A_new[i] + delta
            else:
                # Global move: random value
                new_val = random.randint(1, 10000)
            
            if 1 <= new_val <= 10000 and new_val not in A_new:
                A_new[i] = new_val
            else:
                # If invalid, skip this iteration or try a simple random replacement
                new_val = random.randint(1, 10000)
                if new_val not in A_new:
                    A_new[i] = new_val
                else:
                    continue
        else:
            # Swap two elements
            i, j = random.sample(range(20), 2)
            A_new[i], A_new[j] = A_new[j], A_new[i]
        
        if not try_candidate(A_new):
            continue
            
        new_obj = objective(A_new)
        delta_obj = new_obj - cur_obj
        
        if delta_obj <= 0:
            # Accept improvement
            A = A_new
            cur_obj = new_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_A = A[:]
        else:
            # Accept worse with probability
            prob = math.exp(-delta_obj / T)
            if random.random() < prob:
                A = A_new
                cur_obj = new_obj
                
        T *= cooling_rate
        
    return best_A
