import time
import math
import random
import numpy as np
from itertools import combinations

def fast_score(U):
    """Fast scoring using numpy for vectorized set size computation."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    if 0 not in U or len(U) < 2:
        return -1.0
    max_u = int(U.max())
    if max_u == 0:
        return -1.0
    
    n = len(U)
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    
    if n_sums == 0:
        return -1.0
        
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def local_search(U, deadline, max_steps=200, step_range=5):
    """
    Local search to improve the score of U.
    Tries to move each element, add elements, or remove elements.
    """
    U = sorted(set(U))
    if 0 not in U:
        U.insert(0, 0)
    U = sorted(U)
    
    current_score = fast_score(U)
    if current_score <= 0:
        return U, current_score

    n = len(U)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        improved = False
        indices = list(range(1, n))
        random.shuffle(indices)
        
        # Move existing elements
        for i in indices:
            if time.time() > deadline:
                break
                
            old_val = U[i]
            low = max(1, old_val - step_range)
            high = old_val + step_range
            
            best_new_val = old_val
            best_new_score = current_score
            
            candidates = set()
            candidates.add(old_val - 1)
            candidates.add(old_val + 1)
            if step_range > 1:
                for _ in range(8):
                    candidates.add(random.randint(low, high))
            
            for new_val in candidates:
                if new_val == old_val or new_val < 1:
                    continue
                
                new_U = U[:i] + [new_val] + U[i+1:]
                if len(set(new_U)) != len(new_U):
                    continue
                
                s = fast_score(new_U)
                if s > best_new_score:
                    best_new_score = s
                    best_new_val = new_val
            
            if best_new_val != old_val:
                U[i] = best_new_val
                U = sorted(U)
                current_score = best_new_score
                improved = True
                n = len(U)
        
        # Try adding an element
        if len(U) < 10 and time.time() < deadline:
            max_u = max(U)
            for new_val in range(1, max_u + step_range + 2):
                if new_val in U:
                    continue
                new_U = sorted(U + [new_val])
                s = fast_score(new_U)
                if s > current_score:
                    U = new_U
                    current_score = s
                    improved = True
                    n = len(U)
                    break
        
        # Try removing an element (not 0)
        if len(U) > 3 and time.time() < deadline:
            for i in range(1, len(U)):
                new_U = U[:i] + U[i+1:]
                s = fast_score(new_U)
                if s > current_score:
                    U = new_U
                    current_score = s
                    improved = True
                    n = len(U)
                    break
        
        if not improved:
            step_range += 2
            if step_range > 30:
                break
                
    return U, current_score

def construct():
    start = time.time()
    deadline = start + 110
    
    best_set = [0, 1, 2, 4, 8]
    best_score = fast_score(best_set)
    
    random.seed(42)
    
    # Phase 1: Exhaustive search over small sets
    # Search all subsets of {1, ..., 25} of size 1-6 (with 0)
    max_val = 25
    max_size = 6
    
    candidates = list(range(1, max_val + 1))
    
    for size in range(1, max_size + 1):
        if time.time() > deadline - 20:
            break
        for combo in combinations(candidates, size):
            if time.time() > deadline - 20:
                break
            U = [0] + list(combo)
            U.sort()
            s = fast_score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Phase 2: Local search from multiple starting points
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        strategy = random.random()
        
        if strategy < 0.4:
            # Random small set
            n = random.randint(3, 8)
            max_v = random.randint(15, 80)
            S = {0}
            while len(S) < n:
                x = random.randint(1, max_v)
                S.add(x)
            U = sorted(S)
        elif strategy < 0.7:
            # Geometric-ish
            n = random.randint(3, 7)
            b = random.uniform(1.5, 3.0)
            U = [0, 1]
            cur = 1
            for i in range(n-1):
                cur = int(cur * b)
                while cur in U:
                    cur += 1
                U.append(cur)
            # Perturb
            for i in range(1, len(U)):
                if random.random() < 0.3:
                    U[i] += random.randint(-2, 2)
                    if U[i] < 0:
                        U[i] = 1
            U = sorted(set(U))
        else:
            # Start from current best and perturb
            U = list(best_set)
            for i in range(1, len(U)):
                if random.random() < 0.5:
                    U[i] += random.randint(-3, 3)
                    if U[i] < 1:
                        U[i] = 1
            U = sorted(set(U))
            if 0 not in U:
                U.insert(0, 0)
        
        if len(U) < 2 or 0 not in U:
            continue
            
        improved_U, score = local_search(U, deadline, max_steps=80, step_range=5)
        
        if score > best_score:
            best_score = score
            best_set = improved_U
    
    # Phase 3: Final deep polish on the best set
    if time.time() < deadline - 5:
        improved_U, score = local_search(best_set, deadline, max_steps=300, step_range=15)
        if score > best_score:
            best_score = score
            best_set = improved_U
            
    if time.time() < deadline - 2:
        improved_U, score = local_search(best_set, deadline, max_steps=100, step_range=25)
        if score > best_score:
            best_score = score
            best_set = improved_U

    return best_set

if __name__ == "__main__":
    U = construct()
    print(f"Best Set: {U}")
    print(f"Score: {fast_score(U)}")
