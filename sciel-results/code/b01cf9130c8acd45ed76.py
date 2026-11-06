import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 82  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A_arr = np.array(A, dtype=np.int64)
    sums = A_arr[:, None] + A_arr[None, :]
    num_sums = len(np.unique(sums))
    
    products = A_arr[:, None] * A_arr[None, :]
    num_products = len(np.unique(products))
    
    return max(num_sums, num_products)

def _is_valid(A):
    """Check if A has 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > MAX_VAL:
            return False
    return True

def _rational_geometric_search(deadline):
    """
    Search for sets of the form { round(a * (p/q)^i) } for i=0..19.
    We iterate over small p, q and optimize a.
    """
    best_obj = float('inf')
    best_A = None
    rng = random.Random(42)
    
    max_pq = 30 # Search up to 30/1, 29/1, etc
    
    for p in range(2, max_pq + 1):
        for q in range(1, p):
            if time.time() > deadline:
                return best_A, best_obj
            
            ratio = p / q
            
            # The sequence is a, a*ratio, a*ratio^2, ...
            # a * ratio^19 <= MAX_VAL
            # a <= MAX_VAL / (ratio^19)
            max_a = int(MAX_VAL / (ratio ** 19))
            if max_a < 1:
                continue
            
            candidates_a = set()
            
            # 1. Boundary
            candidates_a.add(max_a)
            
            # 2. Small integers
            for k in range(1, 15):
                if k <= max_a:
                    candidates_a.add(k)
            
            # 3. Multiples of q^19 to ensure integrality of exact geometric progression
            base = q ** 19
            max_k = max_a // base
            if max_k > 0:
                for k in range(1, min(max_k, 30) + 1):
                    candidates_a.add(k * base)
            
            # 4. More random samples
            for _ in range(10):
                candidates_a.add(rng.randint(1, max_a))
            
            # 5. Values near geometric mean positions
            for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
                val = int(max_a * frac)
                if 1 <= val <= max_a:
                    candidates_a.add(val)
            
            # Evaluate candidates
            for a in candidates_a:
                if a < 1:
                    continue
                
                A = []
                valid = True
                current = a
                for i in range(20):
                    if current > MAX_VAL:
                        valid = False
                        break
                    if current in A:
                        valid = False
                        break
                    A.append(current)
                    
                    if i < 19:
                        next_val = (current * p) // q
                        current = next_val
                
                if not valid or len(A) != 20:
                    continue
                
                if len(set(A)) != 20:
                    continue
                
                obj = _objective(A)
                if obj < best_obj:
                    best_obj = obj
                    best_A = list(A)
                
                if time.time() > deadline:
                    return best_A, best_obj

    return best_A, best_obj

def _near_geometric_perturb(A, time_limit, rng):
    """
    Start from a geometric-like set and make small integer perturbations
    to break exact geometric structure while preserving approximate ratios.
    """
    A = list(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    
    # Try perturbing individual elements by small amounts
    while time.time() - start_time < time_limit:
        improved = False
        for idx in range(20):
            if time.time() - start_time > time_limit:
                break
            
            old_val = A[idx]
            
            # Try deltas from -10 to 10
            for delta in range(-10, 11):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 1 or new_val > MAX_VAL:
                    continue
                if new_val in A:
                    continue
                
                A[idx] = new_val
                new_obj = _objective(A)
                A[idx] = old_val
                
                if new_obj < current_obj:
                    A[idx] = new_val
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    improved = True
                    break
        
        if not improved:
            # Try larger perturbations
            for _ in range(20):
                if time.time() - start_time > time_limit:
                    break
                idx = rng.randint(0, 19)
                old_val = A[idx]
                delta = rng.randint(-50, 50)
                new_val = old_val + delta
                if new_val < 1 or new_val > MAX_VAL:
                    continue
                if new_val in A:
                    continue
                
                A[idx] = new_val
                new_obj = _objective(A)
                A[idx] = old_val
                
                if new_obj < current_obj:
                    A[idx] = new_val
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_A, best_obj

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing to minimize the objective."""
    A = list(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 100.0
    cooling_rate = 0.99
    min_temperature = 0.01
    
    while temperature > min_temperature:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to modify
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try several random changes
        for _ in range(30):
            if time.time() - start_time > time_limit:
                break
            
            # Random delta - use smaller steps for better convergence
            r = rng.random()
            if r < 0.3:
                new_val = old_val + rng.randint(-20, 20)
            elif r < 0.6:
                new_val = old_val + rng.randint(-80, 80)
            elif r < 0.85:
                new_val = old_val + rng.randint(-200, 200)
            else:
                new_val = old_val + rng.randint(-500, 500)
            
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in A:
                continue
            
            # Test the change
            A[idx] = new_val
            new_obj = _objective(A)
            A[idx] = old_val
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                break
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _polish_pass(A, time_limit, rng):
    """Final polish with very small steps to find local optima."""
    A = list(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    
    # Use much smaller steps for polishing
    while time.time() - start_time < time_limit:
        improved = False
        for idx in range(20):
            if time.time() - start_time > time_limit:
                break
            
            old_val = A[idx]
            
            # Try small deltas
            for delta in [-7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7]:
                new_val = old_val + delta
                if new_val < 1 or new_val > MAX_VAL:
                    continue
                if new_val in A:
                    continue
                
                A[idx] = new_val
                new_obj = _objective(A)
                A[idx] = old_val
                
                if new_obj < current_obj:
                    A[idx] = new_val
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5  # leave margin
    deadline = start_time + time_budget
    
    best_A = None
    best_obj = float('inf')
    
    # Phase 1: Deterministic Rational Geometric Search
    try:
        A, obj = _rational_geometric_search(deadline)
        if A:
            best_A = A
            best_obj = obj
    except Exception:
        pass
    
    # If we haven't found anything, fallback to simple geometric
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    # Phase 2: Near-geometric perturbation (break exact structure)
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 1.0:
            try:
                perturbed_A, perturbed_obj = _near_geometric_perturb(best_A, remaining * 0.3, random.Random(789))
                if perturbed_obj < best_obj:
                    best_A = perturbed_A
                    best_obj = perturbed_obj
            except Exception:
                pass
    
    # Phase 3: Local Search Refinement
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 1.0:
            try:
                refined_A, refined_obj = _local_search(best_A, remaining * 0.5, random.Random(123))
                if refined_obj < best_obj:
                    best_A = refined_A
                    best_obj = refined_obj
            except Exception:
                pass
    
    # Phase 4: Polish pass
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 0.5:
            try:
                polished_A, polished_obj = _polish_pass(best_A, remaining, random.Random(456))
                if polished_obj < best_obj:
                    best_A = polished_A
                    best_obj = polished_obj
            except Exception:
                pass
    
    # Ensure validity
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return sorted(best_A)
