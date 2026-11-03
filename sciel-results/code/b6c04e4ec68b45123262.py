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

def fast_score_set(U):
    """Score using Python sets - faster for very small sets."""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1.0
    max_u = max(U)
    if max_u == 0:
        return -1.0
    
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    n_sums = len(sums)
    n_diffs = len(diffs)
    
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
    Uses dynamic candidate generation to escape local minima.
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
        
        # Move existing elements with dynamic candidate generation
        for i in indices:
            if time.time() > deadline:
                break
                
            old_val = U[i]
            low = max(1, old_val - step_range)
            high = old_val + step_range
            
            best_new_val = old_val
            best_new_score = current_score
            
            # Generate diverse candidates
            candidates = set()
            # Immediate neighbors
            candidates.add(old_val - 1)
            candidates.add(old_val + 1)
            if step_range > 1:
                candidates.add(old_val - 2)
                candidates.add(old_val + 2)
            
            # Dynamic candidates based on sumset/diffset structure
            # Try values that create new sums or reduce duplicate sums
            max_u = max(U)
            for target in [1, 2, 3, max_u // 2, max_u - 1, max_u]:
                for offset in range(-step_range, step_range + 1):
                    c = target + offset
                    if 1 <= c <= max_u + step_range:
                        candidates.add(c)
            
            # Random candidates
            for _ in range(10):
                candidates.add(random.randint(low, high))
            
            # Sort by distance to old_val for early termination potential
            candidate_list = sorted(candidates, key=lambda x: abs(x - old_val))
            
            for new_val in candidate_list:
                if new_val == old_val or new_val < 1:
                    continue
                if new_val in U:
                    continue
                
                new_U = U[:i] + [new_val] + U[i+1:]
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
        
        # Try swapping two elements
        if n >= 3 and time.time() < deadline:
            for i in range(1, n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    new_U = U[:]
                    new_U[i], new_U[j] = new_U[j], new_U[i]
                    if len(set(new_U)) != len(new_U):
                        continue
                    s = fast_score(new_U)
                    if s > current_score:
                        U = sorted(new_U)
                        current_score = s
                        improved = True
                        break
                if improved:
                    break
        
        # Try adding an element
        if len(U) < 12 and time.time() < deadline:
            max_u = max(U)
            # Generate candidate positions
            add_candidates = set()
            for v in range(1, max_u + step_range + 3):
                add_candidates.add(v)
            # Also try gaps in the set
            for a, b in zip(U, U[1:]):
                if b - a > 1:
                    add_candidates.add(a + (b - a) // 2)
            
            for new_val in sorted(add_candidates):
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
            step_range += 3
            if step_range > 50:
                break
                
    return U, current_score

def construct():
    start = time.time()
    deadline = start + 110
    
    best_set = [0, 1, 2, 4, 8]
    best_score = fast_score(best_set)
    
    random.seed(42)
    
    # Phase 1: Exhaustive search over small sets
    # Search all subsets of {1, ..., 30} of size 1-6 (with 0)
    max_val = 30
    max_size = 6
    
    candidates = list(range(1, max_val + 1))
    
    for size in range(1, max_size + 1):
        if time.time() > deadline - 25:
            break
        for combo in combinations(candidates, size):
            if time.time() > deadline - 25:
                break
            U = [0] + list(combo)
            U.sort()
            s = fast_score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Phase 2: Local search from multiple starting points
    num_restarts = 150
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        strategy = random.random()
        
        if strategy < 0.3:
            # Random small set
            n = random.randint(3, 9)
            max_v = random.randint(15, 100)
            S = {0}
            while len(S) < n:
                x = random.randint(1, max_v)
                S.add(x)
            U = sorted(S)
        elif strategy < 0.5:
            # Geometric-ish
            n = random.randint(3, 8)
            b = random.uniform(1.3, 3.5)
            U = [0, 1]
            cur = 1
            for i in range(n-1):
                cur = int(cur * b)
                while cur in U:
                    cur += 1
                U.append(cur)
            # Perturb
            for i in range(1, len(U)):
                if random.random() < 0.4:
                    U[i] += random.randint(-3, 3)
                    if U[i] < 0:
                        U[i] = 1
            U = sorted(set(U))
        elif strategy < 0.7:
            # Arithmetic progression with noise
            n = random.randint(4, 8)
            start_val = random.randint(1, 5)
            step = random.randint(1, 8)
            U = [0]
            for i in range(n - 1):
                val = start_val + i * step
                if random.random() < 0.3:
                    val += random.randint(-1, 1)
                if val >= 1:
                    U.append(val)
            U = sorted(set(U))
        elif strategy < 0.85:
            # Start from current best and perturb
            U = list(best_set)
            for i in range(1, len(U)):
                if random.random() < 0.6:
                    U[i] += random.randint(-4, 4)
                    if U[i] < 1:
                        U[i] = 1
            U = sorted(set(U))
            if 0 not in U:
                U.insert(0, 0)
            # Also try adding/removing elements
            if random.random() < 0.3 and len(U) < 10:
                U.append(random.randint(1, max(U) + 5))
                U = sorted(set(U))
        else:
            # Fibonacci-like or other structured
            n = random.randint(4, 7)
            fib = [0, 1]
            while len(fib) < n:
                fib.append(fib[-1] + fib[-2])
            U = fib[:]
            for i in range(1, len(U)):
                if random.random() < 0.3:
                    U[i] += random.randint(-2, 2)
                    if U[i] < 1:
                        U[i] = 1
            U = sorted(set(U))
        
        if len(U) < 2 or 0 not in U:
            continue
            
        improved_U, score = local_search(U, deadline, max_steps=100, step_range=5)
        
        if score > best_score:
            best_score = score
            best_set = improved_U
    
    # Phase 3: Final deep polish on the best set
    if time.time() < deadline - 5:
        improved_U, score = local_search(best_set, deadline, max_steps=400, step_range=15)
        if score > best_score:
            best_score = score
            best_set = improved_U
            
    if time.time() < deadline - 2:
        improved_U, score = local_search(best_set, deadline, max_steps=200, step_range=30)
        if score > best_score:
            best_score = score
            best_set = improved_U

    # Final micro-polish with very small steps
    if time.time() < deadline - 1:
        improved_U, score = local_search(best_set, deadline, max_steps=500, step_range=2)
        if score > best_score:
            best_score = score
            best_set = improved_U

    return best_set

if __name__ == "__main__":
    U = construct()
    print(f"Best Set: {U}")
    print(f"Score: {fast_score(U)}")
