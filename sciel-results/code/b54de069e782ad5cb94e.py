import time
import math
import random
import numpy as np
from itertools import combinations
from collections import Counter

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
    
    if n <= 200 and maxU <= MAX_INT64:
        # Use numpy for speed if values fit
        arr = np.array(U, dtype=np.int64)
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        # Use a faster unique count: sort and count transitions
        sums_flat = sums.flatten()
        diffs_flat = diffs.flatten()
        sums_sorted = np.sort(sums_flat)
        diffs_sorted = np.sort(diffs_flat)
        sum_count = 1 + int(np.sum(sums_sorted[1:] != sums_sorted[:-1]))
        diff_count = 1 + int(np.sum(diffs_sorted[1:] != diffs_sorted[:-1]))
    else:
        # Fall back to Python sets for safety (handles arbitrary large ints)
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


def simulated_annealing(initial_set, deadline, max_iter=10000):
    """Simulated annealing to escape local optima"""
    U = sorted(set(initial_set))
    if len(U) < 2:
        return 0.0, U
    
    current_score = compute_score(U)
    best_score = current_score
    best_set = U[:]
    
    # Temperature schedule
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
        
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.6:
            # Replace one element
            i = random.randint(1, n - 1)  # Don't replace 0
            # Generate candidate in a wider range
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
            # Add an element
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
            # Remove an element
            if len(U) <= 3:
                continue
            i = random.randint(1, n - 1)  # Don't remove 0
            new_U = U[:i] + U[i+1:]
            
        else:
            # Swap two elements
            if n < 4:
                continue
            i, j = random.sample(range(1, n), 2)
            # Generate new values
            new_i = random.randint(0, max_val * 2)
            new_j = random.randint(0, max_val * 2)
            
            new_U = sorted(set(U[:i] + [new_i] + U[i+1:j] + [new_j] + U[j+1:]))
            if 0 not in new_U:
                new_U = sorted([0] + [x for x in new_U if x > 0])
        
        if len(new_U) < 2:
            continue
        
        new_score = compute_score(new_U)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or (T > T_min and random.random() < math.exp(delta / T)):
            # Accept the move
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
        
        # Cool down
        T *= cooling_rate
        
        # Early termination if stuck
        if stagnant > 200 and T < 0.01:
            break
    
    return best_score, best_set

def exhaustive_search_small(n, max_val, deadline):
    """Exhaustive search for small sets of size n"""
    best_score = 0.0
    best_set = None
    # Try all subsets of {0, 1, ..., max_val} of size n that contain 0
    candidates = list(range(1, max_val + 1))
    for combo in combinations(candidates, n - 1):
        if time.time() > deadline:
            break
        U = sorted([0] + list(combo))
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
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
    
    # Family 1: {0, 1, 2, ..., n-1} - arithmetic progression
    for n in range(3, 60):
        if time.time() > deadline:
            break
        try_set(list(range(n)))
    
    # Family 2: {0, 1, 3, 7, 15, ...} - Mersenne-like (2^k - 1)
    # Limit n to avoid extremely large numbers that slow down Python set fallback
    for n in range(3, 20):
        if time.time() > deadline:
            break
        try_set([0] + [2**k - 1 for k in range(1, n)])
    
    # Family 3: {0, 1, 2, 4, 8, ...} - powers of 2
    for n in range(3, 20):
        if time.time() > deadline:
            break
        try_set([0] + [2**k for k in range(1, n)])
    
    # Family 4: {0, 1, 3, 6, 10, ...} - triangular numbers
    for n in range(3, 40):
        if time.time() > deadline:
            break
        try_set([0] + [k*(k+1)//2 for k in range(1, n)])
    
    # Family 5: {0, 1, 2, 4, 7, 11, ...} - quadratic
    for n in range(3, 40):
        if time.time() > deadline:
            break
        try_set([0] + [k*k//2 for k in range(1, n)])
    
    # Family 6: Mixed - {0} + small dense set + sparse large elements
    for base_n in range(3, 15):
        if time.time() > deadline:
            break
        for gap in range(2, 30):
            if time.time() > deadline:
                break
            for extra in range(0, 8):
                if time.time() > deadline:
                    break
                U = list(range(base_n))
                for i in range(extra):
                    U.append(base_n * gap * (i + 1))
                try_set(U)
    
    # Family 7: Geometric progression with base b
    for b in range(2, 15):
        if time.time() > deadline:
            break
        for n in range(3, 15):
            if time.time() > deadline:
                break
            try_set([0] + [b**k for k in range(1, n)])
    
    # Family 8: {0, a, 2a, ..., (m-1)a, m*a*b, (m+1)*a*b, ...}
    for m in range(3, 12):
        if time.time() > deadline:
            break
        for b in range(2, 10):
            if time.time() > deadline:
                break
            for extra in range(0, 6):
                if time.time() > deadline:
                    break
                U = list(range(m))
                for i in range(extra):
                    U.append(m * b * (i + 1))
                try_set(U)
    
    # Family 9: Sparse sets with controlled growth
    for start in range(1, 10):
        if time.time() > deadline:
            break
        for ratio in [2, 3, 4, 5]:
            if time.time() > deadline:
                break
            for n in range(3, 15):
                if time.time() > deadline:
                    break
                U = [0]
                val = start
                for _ in range(n - 1):
                    U.append(val)
                    val *= ratio
                try_set(U)
    
    # Family 10: Two-block structure {0, ..., a} ∪ {b, b+1, ..., b+c}
    for a in range(2, 15):
        if time.time() > deadline:
            break
        for b in range(a + 2, a * 5 + 5):
            if time.time() > deadline:
                break
            for c in range(1, 10):
                if time.time() > deadline:
                    break
                U = list(range(a + 1)) + list(range(b, b + c + 1))
                try_set(U)
    
    return best_score, best_set

def hill_climb(initial_set, deadline, max_iter=5000):
    """Local search via element replacement with smarter moves"""
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
                candidates.add(random.randint(0, max_val * 2))
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
                    # Smart addition: try values that extend the pattern
                    if random.random() < 0.3:
                        new_val = max_val + random.randint(1, max(1, max_val // 5))
                    elif random.random() < 0.6:
                        new_val = random.randint(1, max_val * 3)
                    else:
                        new_val = random.randint(1, max_val)
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
                    # Remove random non-zero element
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
            
            if not improved:
                # Try a bigger jump: replace 2 elements
                if len(U) > 4 and stagnant >= 5:
                    for _ in range(3):
                        if time.time() > deadline:
                            break
                        i, j = random.sample(range(1, len(U)), 2)
                        new_i = random.randint(0, max_val * 2)
                        new_j = random.randint(0, max_val * 2)
                        new_U = sorted(set(U[:i] + [new_i] + U[i+1:j] + [new_j] + U[j+1:]))
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
            
            if not improved and stagnant > 30:
                break
    
    return best_score, best_set

def final_polish(U, deadline):
    """High-precision polish: try all single-element replacements and small perturbations"""
    U = sorted(set(U))
    if len(U) < 2:
        return U
    best_score = compute_score(U)
    best_set = U[:]
    max_val = max(U)
    n = len(U)
    
    # Try replacing each element with a range of nearby values
    for i in range(n):
        if time.time() > deadline:
            break
        # Generate candidates: neighbors, scaled versions
        candidates = set()
        base = U[i]
        for delta in range(-5, 6):
            candidates.add(max(0, base + delta))
        candidates.add(base * 2)
        candidates.add(base // 2 if base > 0 else 0)
        candidates.add(base + max_val // 10)
        candidates.add(max(0, base - max_val // 10))
        
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
                n = len(U)
    
    # Try adding one element
    if len(U) < 80:
        for _ in range(50):
            if time.time() > deadline:
                break
            new_val = random.randint(1, max_val * 3)
            if new_val in U:
                continue
            new_U = sorted(U + [new_val])
            s = compute_score(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_set = new_U
                U = new_U
                max_val = max(U)
    
    # Try removing one element
    if len(U) > 3:
        for i in range(1, len(U)):
            if time.time() > deadline:
                break
            new_U = U[:i] + U[i+1:]
            s = compute_score(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_set = new_U
                U = new_U
                max_val = max(U)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # 1. Try small exhaustive search (limited to keep time reasonable)
    for n in range(3, 6):
        if time.time() > deadline - 50:
            break
        max_val = 25 if n <= 4 else 18
        s, U = exhaustive_search_small(n, max_val, deadline)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 2. Try parametric families
    if time.time() < deadline - 40:
        s, U = parametric_families(deadline)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 3. Hill climb from best found so far
    if time.time() < deadline - 30:
        s, U = hill_climb(best_set, deadline, max_iter=4000)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 4. Simulated annealing from best found so far
    if time.time() < deadline - 20:
        s, U = simulated_annealing(best_set, deadline, max_iter=10000)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 5. Try more searches from different starting points
    if time.time() < deadline - 15:
        # Start from structured and random sets
        random.seed(42)
        # Structured starts
        starts = [
            [0, 1, 2, 4, 8],
            [0, 1, 3, 7, 15],
            [0, 1, 2, 3, 5, 8, 13],
            [0, 1, 4, 9, 16, 25],
            [0, 2, 4, 6, 10, 15, 21],
            [0, 1, 3, 9],  # Known good from inspirations
            [0, 1, 2, 5, 10, 20],
            [0, 1, 3, 6, 12, 24],
        ]
        # Random starts
        for seed in range(8):
            if time.time() > deadline - 10:
                break
            random.seed(seed)
            n = random.randint(4, 18)
            U = sorted(random.sample(range(1, 300), n - 1) + [0])
            starts.append(U)
        
        for U0 in starts:
            if time.time() > deadline - 10:
                break
            
            # Alternate between hill climbing and simulated annealing
            if random.random() < 0.5:
                s, U2 = hill_climb(U0, deadline, max_iter=2000)
            else:
                s, U2 = simulated_annealing(U0, deadline, max_iter=6000)
            
            if s > best_score:
                best_score = s
                best_set = U2
    
    # 6. Final polish pass
    if time.time() < deadline - 2:
        best_set = final_polish(best_set, deadline)
    
    return best_set

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
