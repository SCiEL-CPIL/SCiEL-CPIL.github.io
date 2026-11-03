import time
import random
import math

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

def construct():
    start = time.time()
    deadline = start + 140
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    random.seed(42)
    
    # Phase 1: Try superincreasing sequences
    # Powers of 2
    for n in range(2, 20):
        U = [0] + [2**i for i in range(n-1)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Phase 2: Random search
    for n in range(2, 12):
        for M in range(2, 50):
            if time.time() > deadline:
                return best_set
            
            for trial in range(100):
                if time.time() > deadline:
                    return best_set
                
                U = {0}
                attempts = 0
                while len(U) < n and attempts < 1000:
                    U.add(random.randint(1, M))
                    attempts += 1
                if len(U) < n:
                    continue
                U = sorted(U)
                
                s = compute_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Phase 3: Local search on best set
    U = best_set[:]
    for iteration in range(10000):
        if time.time() > deadline:
            return best_set
        
        # Try to improve by modifying one element
        improved = False
        for i in range(len(U)):
            old_val = U[i]
            for new_val in range(0, max(U) + 10):
                if new_val == old_val:
                    continue
                new_U = U[:i] + [new_val] + U[i+1:]
                if len(set(new_U)) != len(new_U):
                    continue
                s = compute_score(new_U)
                if s > best_score:
                    best_score = s
                    best_set = new_U[:]
                    U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding a new element
            for new_val in range(1, max(U) + 20):
                if new_val in U:
                    continue
                new_U = U + [new_val]
                s = compute_score(new_U)
                if s > best_score:
                    best_score = s
                    best_set = new_U[:]
                    U = new_U[:]
                    improved = True
                    break
            
            if not improved:
                # Try removing an element (except 0)
                for i in range(1, len(U)):
                    new_U = U[:i] + U[i+1:]
                    if len(new_U) < 2:
                        continue
                    s = compute_score(new_U)
                    if s > best_score:
                        best_score = s
                        best_set = new_U[:]
                        U = new_U[:]
                        improved = True
                        break
            
            if not improved:
                break
    
    return best_set
