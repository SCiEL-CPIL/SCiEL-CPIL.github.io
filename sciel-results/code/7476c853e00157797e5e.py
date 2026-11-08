import time
import math
import random
import numpy as np
from itertools import combinations

def compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # Determine if we can use numpy int64 safely
    MAX_INT64 = 2**63 - 1
    
    if n <= 150 and maxU <= MAX_INT64:
        arr = np.array(U, dtype=np.int64)
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[:, None]
        sum_count = len(np.unique(sums))
        diff_count = len(np.unique(diffs))
    else:
        sum_set = set()
        diff_set = set()
        for i in range(n):
            ui = U[i]
            for j in range(n):
                sum_set.add(ui + U[j])
                diff_set.add(ui - U[j])
        sum_count = len(sum_set)
        diff_count = len(diff_set)
    
    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / math.log(2 * maxU + 1)
    return score

def grid_search_small(deadline):
    """Grid search for small sets (size 4-6) with bounded values."""
    best_score = 0.0
    best_set = [0, 1, 3]
    
    def try_set(U):
        nonlocal best_score, best_set
        if time.time() > deadline:
            return False
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
        return time.time() <= deadline
    
    # Size 4: [0, a, b, c]
    for a in range(1, 25):
        if time.time() > deadline: break
        for b in range(a + 1, 50):
            if time.time() > deadline: break
            for c in range(b + 1, 80):
                if not try_set([0, a, b, c]):
                    break
    
    # Size 5: [0, a, b, c, d]
    for a in range(1, 15):
        if time.time() > deadline: break
        for b in range(a + 1, 30):
            if time.time() > deadline: break
            for c in range(b + 1, 50):
                if time.time() > deadline: break
                for d in range(c + 1, 70):
                    if not try_set([0, a, b, c, d]):
                        break
    
    # Size 6: [0, a, b, c, d, e] (coarser)
    for a in range(1, 10):
        if time.time() > deadline: break
        for b in range(a + 1, 20):
            if time.time() > deadline: break
            for c in range(b + 1, 35):
                if time.time() > deadline: break
                for d in range(c + 1, 50):
                    if time.time() > deadline: break
                    for e in range(d + 1, 70):
                        if not try_set([0, a, b, c, d, e]):
                            break
    
    return best_score, best_set

def parametric_families(deadline):
    """Try various parametric families"""
    best_score = 0.0
    best_set = None
    
    def try_set(U):
        nonlocal best_score, best_set
        if not U or 0 not in U:
            return
        U = sorted(set(U))
        if len(U) < 2:
            return
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Family 1: Arithmetic progression
    for n in range(3, 50):
        if time.time() > deadline: break
        try_set(list(range(n)))
    
    # Family 2: Mersenne-like
    for n in range(3, 20):
        if time.time() > deadline: break
        try_set([0] + [2**k - 1 for k in range(1, n)])
    
    # Family 3: Powers of 2
    for n in range(3, 20):
        if time.time() > deadline: break
        try_set([0] + [2**k for k in range(1, n)])
    
    # Family 4: Triangular
    for n in range(3, 40):
        if time.time() > deadline: break
        try_set([0] + [k*(k+1)//2 for k in range(1, n)])
    
    # Family 5: Quadratic
    for n in range(3, 40):
        if time.time() > deadline: break
        try_set([0] + [k*k//2 for k in range(1, n)])
    
    # Family 6: Mixed dense/sparse
    for base_n in range(3, 15):
        if time.time() > deadline: break
        for gap in range(2, 30):
            if time.time() > deadline: break
            for extra in range(0, 8):
                if time.time() > deadline: break
                U = list(range(base_n))
                for i in range(extra):
                    U.append(base_n * gap * (i + 1))
                try_set(U)
    
    # Family 7: Geometric
    for b in range(2, 15):
        if time.time() > deadline: break
        for n in range(3, 15):
            if time.time() > deadline: break
            try_set([0] + [b**k for k in range(1, n)])
    
    # Family 8: Two-block
    for a in range(2, 15):
        if time.time() > deadline: break
        for b in range(a + 2, a * 5 + 5):
            if time.time() > deadline: break
            for c in range(1, 10):
                if time.time() > deadline: break
                U = list(range(a + 1)) + list(range(b, b + c + 1))
                try_set(U)
    
    return best_score, best_set

def hill_climb(initial_set, deadline, max_iter=5000, search_range=100):
    """Local search via element replacement with smarter moves."""
    U = sorted(set(initial_set))
    if len(U) < 2:
        return 0.0, U
    best_score = compute_score(U)
    best_set = U[:]
    
    max_val = max(U)
    stagnant = 0
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        n = len(U)
        
        # Try replacing each element with smart candidates
        for i in range(n):
            if time.time() > deadline:
                break
            # Generate diverse replacement candidates
            candidates = set()
            # Random in range
            for _ in range(5):
                candidates.add(random.randint(0, search_range))
            # Nearby values
            candidates.add(max(0, U[i] - 1))
            candidates.add(U[i] + 1)
            candidates.add(max(0, U[i] // 2))
            candidates.add(U[i] * 2)
            # Values that might improve structure
            if i > 0 and i < n - 1:
                candidates.add((U[i-1] + U[i+1]) // 2)
            
            for new_val in candidates:
                if time.time() > deadline:
                    break
                if new_val in U and new_val != U[i]:
                    continue
                new_U = sorted(set(U[:i] + [new_val] + U[i+1:]))
                if 0 not in new_U:
                    new_U = sorted([0] + [x for x in new_U if x > 0])
                if len(new_U) < 2:
                    continue
                s = compute_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    max_val = max(U)
                    improved = True
                    stagnant = 0
                    break
        
        if not improved:
            stagnant += 1
            
            # Try adding a new element
            if len(U) < 60 and stagnant < 10:
                for _ in range(8):
                    if time.time() > deadline:
                        break
                    new_val = random.randint(1, search_range)
                    if new_val in U:
                        continue
                    new_U = sorted(U + [new_val])
                    s = compute_score(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_set = new_U
                        U = new_U
                        max_val = max(U)
                        improved = True
                        stagnant = 0
                        break
            
            if not improved:
                # Try removing an element
                if len(U) > 3 and stagnant < 10:
                    for _ in range(5):
                        if time.time() > deadline:
                            break
                        idx = random.choice([i for i in range(len(U)) if U[i] != 0])
                        new_U = U[:idx] + U[idx+1:]
                        s = compute_score(new_U)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_set = new_U
                            U = new_U
                            max_val = max(U)
                            improved = True
                            stagnant = 0
                            break
            
            if not improved and stagnant > 30:
                break
    
    return best_score, best_set

def simulated_annealing(initial_set, deadline, max_iter=10000):
    """Simulated annealing to escape local optima"""
    U = sorted(set(initial_set))
    if len(U) < 2:
        return 0.0, U
    
    current_score = compute_score(U)
    best_score = current_score
    best_set = U[:]
    
    T0 = 0.5
    T_min = 0.001
    cooling_rate = 0.9995
    
    T = T0
    stagnant = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(U)
        max_val = max(U)
        
        move_type = random.random()
        
        if move_type < 0.6:
            i = random.randint(1, n - 1)
            if random.random() < 0.3:
                new_val = random.randint(0, max_val * 2)
            elif random.random() < 0.5:
                new_val = random.randint(0, max_val)
            else:
                new_val = max(0, U[i] + random.randint(-max_val // 2, max_val // 2))
            
            if new_val in U and new_val != U[i]:
                continue
            
            new_U = sorted(set(U[:i] + [new_val] + U[i+1:]))
            if 0 not in new_U:
                new_U = sorted([0] + [x for x in new_U if x > 0])
                
        elif move_type < 0.8:
            if len(U) >= 50:
                continue
            if random.random() < 0.3:
                new_val = max_val + random.randint(1, max(1, max_val // 3))
            else:
                new_val = random.randint(1, max_val * 2)
            
            if new_val in U:
                continue
            
            new_U = sorted(U + [new_val])
            
        elif move_type < 0.9:
            if len(U) <= 3:
                continue
            i = random.randint(1, n - 1)
            new_U = U[:i] + U[i+1:]
            
        else:
            if n < 4:
                continue
            i, j = random.sample(range(1, n), 2)
            new_i = random.randint(0, max_val * 2)
            new_j = random.randint(0, max_val * 2)
            
            new_U = sorted(set(U[:i] + [new_i] + U[i+1:j] + [new_j] + U[j+1:]))
            if 0 not in new_U:
                new_U = sorted([0] + [x for x in new_U if x > 0])
        
        if len(new_U) < 2:
            continue
        
        new_score = compute_score(new_U)
        
        delta = new_score - current_score
        if delta > 0 or (T > T_min and random.random() < math.exp(delta / T)):
            current_score = new_score
            U = new_U
            
            if current_score > best_score + 1e-12:
                best_score = current_score
                best_set = U[:]
                stagnant = 0
            else:
                stagnant += 1
        else:
            stagnant += 1
        
        T *= cooling_rate
        
        if stagnant > 200 and T < 0.01:
            break
    
    return best_score, best_set

def final_polish(initial_set, deadline):
    """High-precision polish pass with small deltas."""
    U = sorted(set(initial_set))
    if len(U) < 2:
        return 0.0, U
    
    best_score = compute_score(U)
    best_set = U[:]
    U_set = set(U)
    
    for _ in range(500):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(1, len(U)):
            old = U[i]
            # Try small deltas
            for delta in range(-10, 11):
                new_val = old + delta
                if new_val <= 0 or new_val in U_set:
                    continue
                candidate = U[:]
                candidate[i] = new_val
                candidate = sorted(candidate)
                if len(set(candidate)) < len(candidate):
                    continue
                s = compute_score(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = candidate
                    U = candidate
                    U_set = set(U)
                    improved = True
                    break
                if time.time() > deadline:
                    break
            if improved:
                break
            if time.time() > deadline:
                break
        
        if not improved:
            # Try adding
            for new_val in range(1, 100):
                if new_val in U_set:
                    continue
                candidate = sorted(list(U) + [new_val])
                s = compute_score(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = candidate
                    U = candidate
                    U_set = set(U)
                    improved = True
                    break
                if time.time() > deadline:
                    break
            
            if not improved:
                # Try removing
                for i in range(1, len(U)):
                    candidate = U[:i] + U[i+1:]
                    if len(candidate) < 2:
                        continue
                    s = compute_score(candidate)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_set = candidate
                        U = candidate
                        U_set = set(U)
                        improved = True
                        break
                    if time.time() > deadline:
                        break
                
                if not improved:
                    break
    
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # 1. Grid search for small sets
    if time.time() < deadline - 30:
        s, U = grid_search_small(deadline - 25)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 2. Parametric families
    if time.time() < deadline - 25:
        s, U = parametric_families(deadline - 20)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 3. Hill climb from best found so far
    if time.time() < deadline - 20:
        s, U = hill_climb(best_set, deadline - 15, max_iter=3000, search_range=100)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 4. Simulated annealing from best found so far
    if time.time() < deadline - 15:
        s, U = simulated_annealing(best_set, deadline - 10, max_iter=8000)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 5. Try more searches from different starting points
    if time.time() < deadline - 10:
        random.seed(42)
        starts = [
            [0, 1, 2, 4, 8],
            [0, 1, 3, 7, 15],
            [0, 1, 2, 3, 5, 8, 13],
            [0, 1, 4, 9, 16, 25],
            [0, 2, 4, 6, 10, 15, 21],
            [0, 1, 3, 9],
        ]
        for seed in range(6):
            if time.time() > deadline - 5:
                break
            random.seed(seed)
            n = random.randint(4, 15)
            U = sorted(random.sample(range(1, 200), n - 1) + [0])
            starts.append(U)
        
        for U0 in starts:
            if time.time() > deadline - 5:
                break
            
            if random.random() < 0.5:
                s, U2 = hill_climb(U0, deadline - 2, max_iter=1500, search_range=100)
            else:
                s, U2 = simulated_annealing(U0, deadline - 2, max_iter=5000)
            
            if s > best_score:
                best_score = s
                best_set = U2
    
    # 6. Final polish
    if time.time() < deadline - 2:
        s, U = final_polish(best_set, deadline - 1)
        if s > best_score:
            best_score = s
            best_set = U
    
    return best_set
