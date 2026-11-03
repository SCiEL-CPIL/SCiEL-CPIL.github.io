import time
import itertools
import math

def _compute_score(U):
    """Compute the score for a set U."""
    if not U or 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute sumset and difference set
    u_set = set(U)
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    num_sum = len(sumset)
    num_diff = len(diffset)
    if num_sum == 0:
        return 0.0
    log_ratio = math.log(num_diff / num_sum)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 0.0
    return 1.0 + log_ratio / log_denom

def _try_set(U):
    """Try a specific set and return its score."""
    U = list(set(U))  # Ensure unique
    if 0 not in U:
        U.append(0)
    U.sort()
    return _compute_score(U)

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = _compute_score(best_set)
    
    # Strategy: try sets with geometric-like growth
    # For small sets, brute force search over possible element values
    
    # First, try sets of the form {0, a, b, c, ...} with increasing elements
    # Focus on sets where elements grow moderately fast
    
    # Try sets with 3-8 elements
    for k in range(3, 9):
        if time.time() > deadline - 10:
            break
        
        # Try various growth patterns
        # Pattern 1: powers of 2
        U = [0] + [2**i for i in range(k-1)]
        s = _try_set(U)
        if s > best_score:
            best_score = s
            best_set = list(U)
        
        # Pattern 2: powers of 3
        U = [0] + [3**i for i in range(k-1)]
        s = _try_set(U)
        if s > best_score:
            best_score = s
            best_set = list(U)
        
        # Pattern 3: Fibonacci-like
        fib = [1, 2]
        for i in range(k-3):
            fib.append(fib[-1] + fib[-2])
        U = [0] + fib
        s = _try_set(U)
        if s > best_score:
            best_score = s
            best_set = list(U)
        
        # Pattern 4: Slow growth - arithmetic progression with gap
        for gap in [2, 3, 4, 5]:
            U = [0] + [gap * i for i in range(1, k)]
            s = _try_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
        
        # Pattern 5: Mix - start slow, then jump
        for first_gap in [1, 2, 3]:
            for jump_factor in [3, 4, 5]:
                U = [0]
                current = first_gap
                for i in range(1, k):
                    U.append(current)
                    current = current * jump_factor
                s = _try_set(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
    
    # Try some hand-crafted sets known to have good properties
    test_sets = [
        [0, 1, 3],
        [0, 1, 3, 9],
        [0, 1, 3, 9, 27],
        [0, 1, 4, 10],
        [0, 1, 4, 10, 28],
        [0, 2, 5],
        [0, 2, 5, 14],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 2, 5, 10],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 15, 31],
        [0, 2, 6, 18],
        [0, 1, 3, 6, 10, 15],  # Triangular numbers
        [0, 1, 4, 9, 16, 25],  # Squares
    ]
    
    for U in test_sets:
        if time.time() > deadline - 5:
            break
        s = _try_set(U)
        if s > best_score:
            best_score = s
            best_set = list(U)
    
    # Local search: try adding/removing elements from best_set
    if time.time() < deadline - 20:
        current_set = list(best_set)
        current_score = best_score
        improved = True
        max_iterations = 500
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            if not improved:
                break
            improved = False
            
            # Try adding a new element
            max_elem = max(current_set) if current_set else 0
            for new_elem in range(1, max_elem * 3 + 10):
                if new_elem in current_set:
                    continue
                trial = current_set + [new_elem]
                s = _try_set(trial)
                if s > current_score + 1e-9:
                    current_set = trial
                    current_score = s
                    best_set = list(current_set)
                    best_score = s
                    improved = True
                    break
            
            # Try removing an element (except 0)
            if len(current_set) > 2:
                for i in range(1, len(current_set)):
                    trial = current_set[:i] + current_set[i+1:]
                    s = _try_set(trial)
                    if s > current_score + 1e-9:
                        current_set = trial
                        current_score = s
                        best_set = list(current_set)
                        best_score = s
                        improved = True
                        break
    
    return best_set
