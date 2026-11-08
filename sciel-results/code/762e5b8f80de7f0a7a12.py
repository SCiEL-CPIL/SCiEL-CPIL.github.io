import time
import math
import random
from itertools import combinations

def compute_score(U):
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0
    max_val = max(U_set)
    sumset = set()
    diffset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    denom = math.log(2 * max_val + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def local_search(U, deadline, max_no_improvement=30):
    """Local search with dynamic step size"""
    U = sorted(set(U))
    best_score = compute_score(U)
    best_set = U[:]
    no_improve = 0
    step_size = max(1, int(0.1 * max(U)) if U else 1)
    
    while time.time() < deadline and no_improve < max_no_improvement:
        improved = False
        # Try moving each element by various amounts
        for i in range(len(U)):
            if time.time() > deadline:
                break
            original = U[i]
            # Try several step sizes
            for delta in range(-step_size, step_size + 1):
                if delta == 0:
                    continue
                new_val = original + delta
                if new_val < 0:
                    continue
                if new_val in U and new_val != original:
                    continue
                # Create new set
                new_U = U[:]
                new_U[i] = new_val
                new_U = sorted(set(new_U))
                if len(new_U) != len(U):
                    continue
                s = compute_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    no_improve = 0
                    improved = True
                    break
            if improved:
                break
        if not improved:
            no_improve += 1
            # Reduce step size
            step_size = max(1, step_size // 2)
    
    return best_set, best_score

def simulated_annealing(U, deadline, max_iter=5000):
    """Simulated annealing to explore larger moves"""
    U = sorted(set(U))
    current_score = compute_score(U)
    best_set = U[:]
    best_score = current_score
    
    T = max(1.0, 0.1 * max(U)) if U else 1.0
    T_min = 0.1
    cooling = 0.999
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if T <= T_min:
            break
        
        # Generate neighbor: move one element
        i = random.randint(0, len(U) - 1)
        delta = random.randint(-int(T), int(T))
        new_val = U[i] + delta
        if new_val < 0 or new_val in U:
            continue
        
        new_U = U[:]
        new_U[i] = new_val
        new_U = sorted(set(new_U))
        if len(new_U) != len(U):
            continue
        
        new_score = compute_score(new_U)
        delta_score = new_score - current_score
        
        if delta_score > 0 or random.random() < math.exp(delta_score / T):
            U = new_U
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = U[:]
        
        T *= cooling
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 150s limit
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Phase 1: Geometric sets
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        for k in range(1, 40):
            if time.time() > deadline - 10:
                break
            U = [0] + [base**i for i in range(k)]
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Phase 2: Local search from best geometric set
    if time.time() < deadline - 5:
        best_set, best_score = local_search(best_set, deadline - 5)
    
    # Phase 3: Simulated annealing from multiple starting points
    if time.time() < deadline - 10:
        # Try SA from geometric sets
        for base in [2, 3, 4, 5]:
            if time.time() > deadline - 10:
                break
            k = 8
            U = [0] + [base**i for i in range(k)]
            sa_set, sa_score = simulated_annealing(U, deadline - 10, max_iter=2000)
            if sa_score > best_score:
                best_score = sa_score
                best_set = sa_set
    
    # Phase 4: Final polish with local search
    if time.time() < deadline - 2:
        best_set, best_score = local_search(best_set, deadline - 2, max_no_improvement=50)
    
    return best_set
