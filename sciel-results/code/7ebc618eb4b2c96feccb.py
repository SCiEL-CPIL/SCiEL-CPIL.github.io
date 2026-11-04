"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def _compute_score(U):
    """Compute the lower bound score for a set U."""
    if not U or 0 not in U:
        return 0.0
    U_sorted = sorted(U)
    n = len(U_sorted)
    max_u = U_sorted[-1]
    
    # Compute sumset U+U
    sumset = set()
    for i in range(n):
        for j in range(i, n):
            sumset.add(U_sorted[i] + U_sorted[j])
    
    # Compute difference set U-U
    diffset = set()
    for i in range(n):
        for j in range(n):
            diffset.add(U_sorted[i] - U_sorted[j])
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    
    denom = math.log2(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    
    return 1.0 + math.log(ratio, 2) / denom

def _geometric_seed(ratio, n):
    """Create a geometric sequence seed."""
    U = [0]
    current = 1
    for _ in range(n - 1):
        current = int(current * ratio + 0.5)
        if current in U:
            current += 1
        U.append(current)
    return U

def _random_seed(n, max_val):
    """Create a random seed."""
    U = {0}
    while len(U) < n:
        U.add(random.randint(1, max_val))
    return sorted(U)

def _local_search(U, max_steps=500, timeout=10.0):
    """Perform local search to improve the score."""
    start_time = time.time()
    current_U = sorted(U)
    current_score = _compute_score(current_U)
    best_U = current_U[:]
    best_score = current_score
    
    n = len(current_U)
    
    for step in range(max_steps):
        if time.time() - start_time > timeout:
            break
        
        improved = False
        
        # Try adding a new element
        if random.random() < 0.3 and n < 15:
            # Try adding a new element
            max_u = current_U[-1]
            candidates = []
            for delta in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
                candidates.append(max_u + delta)
            candidates.extend([random.randint(1, max_u * 2) for _ in range(5)])
            
            for cand in candidates:
                if cand not in current_U and cand > 0:
                    new_U = sorted(current_U + [cand])
                    new_score = _compute_score(new_U)
                    if new_score > current_score:
                        current_U = new_U
                        current_score = new_score
                        if current_score > best_score:
                            best_U = current_U[:]
                            best_score = current_score
                        improved = True
                        break
        
        # Try replacing an element
        if not improved:
            idx = random.randint(0, n - 1)
            old_val = current_U[idx]
            candidates = []
            for delta in [-3, -2, -1, 1, 2, 3, 5, 8, 13]:
                new_val = old_val + delta
                if new_val > 0 and new_val not in current_U:
                    candidates.append(new_val)
            candidates.extend([random.randint(1, current_U[-1] * 2) for _ in range(3)])
            
            for cand in candidates:
                new_U = current_U[:]
                new_U[idx] = cand
                new_U = sorted(set(new_U))
                if len(new_U) != n:
                    continue
                new_score = _compute_score(new_U)
                if new_score > current_score:
                    current_U = new_U
                    current_score = new_score
                    if current_score > best_score:
                        best_U = current_U[:]
                        best_score = current_score
                    improved = True
                    break
        
        # Try removing an element
        if not improved and n > 3:
            idx = random.randint(1, n - 1)  # Don't remove 0
            new_U = current_U[:]
            del new_U[idx]
            new_score = _compute_score(new_U)
            if new_score > current_score:
                current_U = new_U
                current_score = new_score
                if current_score > best_score:
                    best_U = current_U[:]
                    best_score = current_score
                improved = True
    
    return best_U, best_score

def construct():
    """Construct a finite set U to maximize the sumset/difference-set exponent lower bound."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    random.seed(42)
    
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # Try various geometric ratios and sizes
    ratios = [1.5, 1.7, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0]
    sizes = [5, 7, 9, 11, 13, 15]
    
    for ratio in ratios:
        for size in sizes:
            if time.time() > deadline:
                return best_U
            
            U = _geometric_seed(ratio, size)
            U, score = _local_search(U, max_steps=300, timeout=5.0)
            if score > best_score:
                best_U = U[:]
                best_score = score
    
    # Try random seeds
    for _ in range(10):
        if time.time() > deadline:
            return best_U
        
        size = random.randint(5, 12)
        max_val = random.randint(50, 500)
        U = _random_seed(size, max_val)
        U, score = _local_search(U, max_steps=400, timeout=8.0)
        if score > best_score:
            best_U = U[:]
            best_score = score
    
    # Final polish with more steps on the best solution
    if time.time() < deadline - 10:
        U, score = _local_search(best_U, max_steps=1000, timeout=20.0)
        if score > best_score:
            best_U = U[:]
            best_score = score
    
    return best_U
# EVOLVE-BLOCK-END
