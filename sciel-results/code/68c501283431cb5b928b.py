import time
import random
import math

def _score(A):
    """Compute the objective value: max(|A+A|, |A*A|)"""
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def _sa_search(A, start_temp=10.0, end_temp=0.1, steps=50000, seed=None):
    """Simulated Annealing search."""
    if seed is not None:
        random.seed(seed)
    
    current_A = A[:]
    current_score = _score(current_A)
    best_A = current_A[:]
    best_score = current_score
    
    n = len(current_A)
    deadline = time.time() + 45.0  # Safety margin for 60s limit
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Cooling schedule
        T = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # Generate neighbor
        k = random.randint(0, n - 1)
        old_val = current_A[k]
        
        # Choose new value
        # Strategy: mix of random jumps and local perturbations
        if random.random() < 0.5:
            # Local perturbation
            delta = random.randint(-50, 50)
            new_val = old_val + delta
        else:
            # Global jump
            new_val = random.randint(1, 10000)
            
        # Ensure bounds and distinctness
        if new_val < 1 or new_val > 10000:
            continue
            
        # Check distinctness
        is_distinct = True
        for i in range(n):
            if i != k and current_A[i] == new_val:
                is_distinct = False
                break
        
        if not is_distinct:
            continue
            
        # Apply move
        current_A[k] = new_val
        new_score = _score(current_A)
        
        # Acceptance probability
        delta_score = new_score - current_score
        if delta_score < 0 or random.random() < math.exp(-delta_score / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = current_A[:]
        else:
            # Reject move
            current_A[k] = old_val
            
    return best_A, best_score

def construct():
    n = 20
    max_val = 10000
    
    # Deadline guard
    start_time = time.time()
    time_limit = 50.0  # Leave 10s buffer for overhead/return
    
    best_A = None
    best_score = float('inf')
    
    # Initial candidates
    # 1. Arithmetic Progression
    A_ap = list(range(1, n + 1))
    # 2. Geometric-like (powers of 2, scaled)
    A_gp = [2**i for i in range(14)]
    # Pad GP to 20 elements with small distinct numbers? 
    # Actually, let's just use a mix.
    # 3. Random
    A_rand = random.sample(range(1, max_val + 1), n)
    
    candidates = [A_ap, A_gp + list(range(1, 7)), A_rand]
    
    # Ensure all candidates are valid (distinct, positive, <= max)
    valid_candidates = []
    for c in candidates:
        if len(c) != n:
            # Fix length if needed
            if len(c) < n:
                # Add random distinct numbers
                existing = set(c)
                while len(c) < n:
                    v = random.randint(1, max_val)
                    if v not in existing:
                        c.append(v)
                        existing.add(v)
            else:
                c = c[:n]
        if len(set(c)) == n and all(1 <= x <= max_val for x in c):
            valid_candidates.append(c[:])
            
    if not valid_candidates:
        # Fallback
        valid_candidates = [list(range(1, n+1))]
        
    # Run SA from each candidate
    for i, init_A in enumerate(valid_candidates):
        if time.time() - start_time > time_limit:
            break
            
        # Use different parameters for diversity
        # Higher temp for early restarts, lower for later?
        # Or just run standard SA.
        # We can run multiple SA instances in parallel? No, single thread.
        # We can run SA with different seeds.
        
        # Limit steps per restart to ensure we try multiple restarts
        # Total time is the constraint.
        # Let's allocate time proportionally.
        
        A_res, score_res = _sa_search(init_A, start_temp=5.0, end_temp=0.01, steps=20000, seed=i)
        
        if score_res < best_score:
            best_score = score_res
            best_A = A_res
            
    # If we have time left, do more refinements on the best solution found
    if best_A is not None and time.time() - start_time < time_limit - 5:
        # Refine with lower temperature and more steps
        A_refined, score_refined = _sa_search(best_A, start_temp=1.0, end_temp=0.001, steps=50000, seed=123)
        if score_refined < best_score:
            best_score = score_refined
            best_A = A_refined
            
    if best_A is None:
        best_A = list(range(1, n+1))
        
    return best_A
