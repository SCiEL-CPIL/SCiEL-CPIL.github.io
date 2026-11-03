import time
import math
import random
import numpy as np

def compute_score(U):
    U = tuple(sorted(set(U)))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def fast_score(U):
    """Fast scoring using numpy for vectorized set size computation."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    if 0 not in U or len(U) < 2:
        return -1.0
    max_u = int(U.max())
    if max_u == 0:
        return -1.0
    
    # Compute all sums and differences using broadcasting
    # For small n, this is fast
    n = len(U)
    if n > 30:
        # Fall back to set-based for large n
        sums_set = set()
        diffs_set = set()
        for a in U:
            for b in U:
                sums_set.add(a + b)
                diffs_set.add(a - b)
        ratio = len(diffs_set) / len(sums_set)
        if ratio <= 1:
            return 1.0
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    
    # Vectorized: outer sum and difference
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def construct():
    start = time.time()
    deadline = start + 135  # Safety margin before 150s kill
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    random.seed(42)
    
    # Phase 1: Systematic exploration of superincreasing-like sequences
    # Powers of 2
    for n in range(2, 25):
        if time.time() > deadline:
            return best_set
        U = [0] + [2**i for i in range(n-1)]
        if max(U) > 100000:
            break
        s = fast_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Other bases
    for base in range(2, 10):
        if time.time() > deadline:
            return best_set
        for n in range(2, 20):
            U = [0] + [base**i for i in range(n-1)]
            if max(U) > 200000:
                break
            s = fast_score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Phase 2: Random search with guided generation
    # Strategy: generate sets that are "spread out" to maximize differences
    for n in range(3, 18):
        for M in [10, 20, 50, 100, 200, 500, 1000, 2000]:
            if time.time() > deadline:
                return best_set
            
            for trial in range(50):
                if time.time() > deadline:
                    return best_set
                
                # Generate a random set with 0
                U = {0}
                attempts = 0
                while len(U) < n and attempts < 5000:
                    # Bias toward larger values to maximize spread
                    if random.random() < 0.3:
                        U.add(random.randint(1, M))
                    else:
                        # Geometric-like distribution
                        U.add(int(random.random() ** 0.5 * M))
                    attempts += 1
                if len(U) < n:
                    continue
                U = sorted(U)
                
                s = fast_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Phase 3: Simulated annealing on the best set
    U = np.array(best_set, dtype=np.int64)
    current_score = best_score
    
    T = 0.1  # Initial temperature
    T_min = 0.001
    cooling = 0.995
    
    for iteration in range(50000):
        if time.time() > deadline:
            return best_set
        
        if T < T_min:
            T = T_min
        
        n = len(U)
        
        # Random move: modify one element, add, or remove
        move_type = random.random()
        
        if move_type < 0.6 and n > 2:
            # Modify a random element (not 0)
            idx = random.randint(1, n-1)
            old_val = U[idx]
            # Propose new value near old or random
            if random.random() < 0.5:
                new_val = max(0, old_val + random.randint(-5, 5))
            else:
                new_val = random.randint(0, max(U) + 10)
            
            if new_val == old_val:
                continue
            
            new_U = U.copy()
            new_U[idx] = new_val
            # Ensure uniqueness
            if len(set(new_U)) != len(new_U):
                continue
            
            s = fast_score(new_U)
            delta = s - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                U = new_U
                current_score = s
                if s > best_score:
                    best_score = s
                    best_set = sorted(U.tolist())
        
        elif move_type < 0.85:
            # Add a new element
            if n < 25:
                new_val = random.randint(1, max(U) + 20)
                if new_val not in U:
                    new_U = np.append(U, new_val)
                    new_U = np.sort(new_U)
                    s = fast_score(new_U)
                    delta = s - current_score
                    
                    if delta > 0 or random.random() < math.exp(delta / T):
                        U = new_U
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = sorted(U.tolist())
        
        else:
            # Remove an element (not 0)
            if n > 3:
                idx = random.randint(1, n-1)
                new_U = np.delete(U, idx)
                s = fast_score(new_U)
                delta = s - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    U = new_U
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_set = sorted(U.tolist())
        
        T *= cooling
    
    # Phase 4: Final local polish
    U = sorted(best_set)
    for iteration in range(2000):
        if time.time() > deadline:
            return best_set
        
        improved = False
        n = len(U)
        
        # Try modifying each element
        for i in range(1, n):
            if time.time() > deadline:
                return best_set
            old_val = U[i]
            for new_val in range(max(0, old_val - 3), min(max(U) + 3, old_val + 4)):
                if new_val == old_val:
                    continue
                new_U = U[:i] + [new_val] + U[i+1:]
                if len(set(new_U)) != len(new_U):
                    continue
                s = fast_score(new_U)
                if s > best_score:
                    best_score = s
                    best_set = new_U[:]
                    U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding
            for new_val in range(1, max(U) + 10):
                if new_val in U:
                    continue
                new_U = U + [new_val]
                s = fast_score(new_U)
                if s > best_score:
                    best_score = s
                    best_set = new_U[:]
                    U = new_U[:]
                    improved = True
                    break
        
        if not improved:
            # Try removing
            for i in range(1, len(U)):
                new_U = U[:i] + U[i+1:]
                if len(new_U) < 2:
                    continue
                s = fast_score(new_U)
                if s > best_score:
                    best_score = s
                    best_set = new_U[:]
                    U = new_U[:]
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_set
