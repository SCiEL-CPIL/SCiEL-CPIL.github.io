import time
import itertools
from math import log

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin
    
    def score_set(U):
        U = sorted(set(U))
        if 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 1.0
        # Compute sumset and difference set
        sums = set()
        diffs = set()
        n = len(U)
        for i in range(n):
            for j in range(i, n):
                sums.add(U[i] + U[j])
            for j in range(n):
                diffs.add(U[i] - U[j])
        ratio = len(diffs) / len(sums)
        if ratio <= 0:
            return -1
        return 1.0 + log(ratio) / log(2 * max_u + 1)
    
    best_score = score_set([0, 1, 3])
    best_set = [0, 1, 3]
    
    # Try various small sets
    # First, try known good structures
    
    # Geometric-like sequences
    for base in [2, 3, 4, 5]:
        for n in range(3, 8):
            if time.time() > deadline:
                break
            U = [0]
            val = 1
            for i in range(n - 1):
                U.append(val)
                val *= base
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Try power-of-2 based with some variation
    for n in range(3, 10):
        if time.time() > deadline:
            break
        U = [0]
        val = 1
        for i in range(n - 1):
            U.append(val)
            val = val * 2 + 1  # Mersenne-like
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Greedy Sidon sequence construction
    def greedy_sidon(max_elements=15):
        U = [0]
        used_sums = {0}
        candidate = 1
        while len(U) < max_elements:
            # Check if candidate can be added
            valid = True
            for u in U:
                if u + candidate in used_sums:
                    valid = False
                    break
            if valid:
                for u in U:
                    used_sums.add(u + candidate)
                U.append(candidate)
            candidate += 1
        return U
    
    U_sidon = greedy_sidon(12)
    s = score_set(U_sidon)
    if s > best_score:
        best_score = s
        best_set = U_sidon[:]
    
    # Try random search with local improvement
    import random
    random.seed(42)
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        n = random.randint(3, 8)
        max_val = random.randint(10, 200)
        
        # Generate random set containing 0
        U = [0] + sorted(random.sample(range(1, max_val + 1), n - 1))
        
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Local search: try to improve best_set by swapping elements
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        U = best_set[:]
        n = len(U)
        
        # Try replacing one element
        idx = random.randint(0, n - 1)
        old_val = U[idx]
        
        # Try nearby values
        for delta in [-3, -2, -1, 1, 2, 3]:
            new_val = old_val + delta
            if new_val < 0:
                continue
            if new_val == 0 and idx != 0:
                continue
            if new_val in U:
                continue
            
            U[idx] = new_val
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
                break
            U[idx] = old_val
    
    return best_set
