import time
import random
import sys

def compute_metrics(A):
    """
    Computes the maximum of the size of the sumset A+A and product set A*A.
    A is a list of distinct positive integers.
    """
    sum_set = set()
    prod_set = set()
    
    # Optimization: Use local variables for speed
    A_list = list(A)
    n = len(A_list)
    
    for i in range(n):
        ai = A_list[i]
        for j in range(n):
            aj = A_list[j]
            sum_set.add(ai + aj)
            prod_set.add(ai * aj)
            
    return max(len(sum_set), len(prod_set))

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Initial candidates
    candidates = []
    
    # 1. Arithmetic Progression
    ap = list(range(1, 21))
    candidates.append(ap)
    
    # 2. Geometric-like (limited by 10000)
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 (13 terms)
    # Need 7 more. 
    # Maybe extend with larger steps?
    # Let's just add random distinct numbers > 4096
    gp = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    while len(gp) < 20:
        new_val = random.randint(4097, 10000)
        if new_val not in gp:
            gp.append(new_val)
    candidates.append(gp)
    
    # 3. Random set
    random_set = random.sample(range(1, 10001), 20)
    candidates.append(random_set)
    
    best_A = None
    best_score = float('inf')
    
    for cand in candidates:
        score = compute_metrics(cand)
        if score < best_score:
            best_score = score
            best_A = cand[:]
            
    # Local Search (Hill Climbing)
    # We will try to improve best_A
    current_A = best_A[:]
    current_score = best_score
    
    # To make random restarts possible, we might keep a list of local optima
    # But simple hill climbing on the best found so far is a good start.
    # Let's implement a loop that runs until deadline.
    
    # To escape local minima, we can occasionally restart from a random point
    # or perturb heavily.
    
    while time.time() < deadline:
        # Randomly pick an index to change
        i = random.randint(0, 19)
        old_val = current_A[i]
        
        # Generate a new value
        # Strategy: Sometimes small change, sometimes large
        if random.random() < 0.5:
            # Small change: +/- random small amount
            delta = random.randint(-10, 10)
            new_val = old_val + delta
        else:
            # Large change: random anywhere
            new_val = random.randint(1, 10000)
        
        # Ensure positive and distinct
        if new_val < 1 or new_val > 10000:
            continue
        if new_val in current_A and new_val != old_val:
            continue
            
        # Update A
        current_A[i] = new_val
        
        # Evaluate
        new_score = compute_metrics(current_A)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
        else:
            # Revert
            current_A[i] = old_val
            
            # Simulated Annealing: Accept worse with small probability?
            # Or just restart if stuck?
            # For now, let's add a restart mechanism if we are stuck for a while.
            # Simple approach: If no improvement in last 1000 steps, restart from best_A or random.
            # We need a counter for this.
            
    return best_A
