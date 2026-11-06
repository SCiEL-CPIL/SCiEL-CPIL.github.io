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
    
    max_pq = 20
    
    for p in range(2, max_pq + 1):
        for q in range(1, p):
            if time.time() > deadline:
                return best_A, best_obj
            
            ratio = p / q
            max_a = int(MAX_VAL / (ratio ** 19))
            if max_a < 1:
                continue
            
            candidates_a = set()
            candidates_a.add(max_a)
            
            for k in range(1, 10):
                if k <= max_a:
                    candidates_a.add(k)
            
            base = q ** 19
            max_k = max_a // base
            if max_k > 0:
                for k in range(1, min(max_k, 20) + 1):
                    candidates_a.add(k * base)
            
            for _ in range(5):
                candidates_a.add(rng.randint(1, max_a))
            
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

def _arithmetic_progression_seed(rng, deadline):
    """
    Seed with arithmetic progressions to exploit additive collisions.
    For an AP {a, a+d, ..., a+(n-1)d}, |A+A| = 2n-1 = 39 (exact).
    We try various (a, d) pairs and small perturbations.
    """
    best_obj = float('inf')
    best_A = None
    
    # Try various common differences
    for d in range(1, 500):
        if time.time() > deadline:
            break
        
        # a must satisfy a + 19*d <= 10000
        max_a = MAX_VAL - 19 * d
        if max_a < 1:
            continue
        
        # Try a few values of a
        candidate_as = set()
        candidate_as.add(1)
        candidate_as.add(max_a)
        for _ in range(5):
            candidate_as.add(rng.randint(1, max_a))
        
        for a in candidate_as:
            A = [a + i * d for i in range(20)]
            if len(set(A)) != 20:
                continue
            if any(x < 1 or x > MAX_VAL for x in A):
                continue
            
            obj = _objective(A)
            if obj < best_obj:
                best_obj = obj
                best_A = list(A)
            
            # Also try slight perturbations of the AP
            if obj < 80:  # Only perturb good candidates
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    A_pert = list(A)
                    idx = rng.randint(0, 19)
                    delta = rng.randint(-2*d, 2*d)
                    if delta == 0:
                        continue
                    new_val = A_pert[idx] + delta
                    if new_val < 1 or new_val > MAX_VAL or new_val in A_pert:
                        continue
                    A_pert[idx] = new_val
                    obj_pert = _objective(A_pert)
                    if obj_pert < best_obj:
                        best_obj = obj_pert
                        best_A = list(A_pert)
    
    return best_A, best_obj

def _swap_local_search(A, time_limit, rng):
    """
    Local search that swaps elements to preserve multiplicative structure.
    Swaps are more likely to maintain product set structure than single-element changes.
    """
    A = list(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 50.0
    cooling_rate = 0.997
    min_temperature = 0.01
    
    while temperature > min_temperature:
        if time.time() - start_time > time_limit:
            break
        
        # Try a swap
        i = rng.randint(0, 19)
        j = rng.randint(0, 19)
        if i == j:
            continue
        
        old_i, old_j = A[i], A[j]
        
        # Generate candidate swap values
        found = False
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
            
            # Small perturbations for both elements
            di = rng.randint(-20, 20)
            dj = rng.randint(-20, 20)
            new_i = old_i + di
            new_j = old_j + dj
            
            if new_i < 1 or new_i > MAX_VAL or new_j < 1 or new_j > MAX_VAL:
                continue
            if new_i == new_j:
                continue
            if new_i in A and new_i != old_i:
                continue
            if new_j in A and new_j != old_j:
                continue
            
            A[i] = new_i
            A[j] = new_j
            new_obj = _objective(A)
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                found = True
                break
            else:
                A[i] = old_i
                A[j] = old_j
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing to minimize the objective."""
    A = list(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 100.0
    cooling_rate = 0.995
    min_temperature = 0.01
    
    while temperature > min_temperature:
        if time.time() - start_time > time_limit:
            break
        
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        found_improvement = False
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            
            if rng.random() < 0.4:
                new_val = old_val + rng.randint(-30, 30)
            elif rng.random() < 0.7:
                new_val = old_val + rng.randint(-100, 100)
            else:
                new_val = old_val + rng.randint(-300, 300)
            
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in A:
                continue
            
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
                found_improvement = True
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
    
    while time.time() - start_time < time_limit:
        for idx in range(20):
            if time.time() - start_time > time_limit:
                break
            
            old_val = A[idx]
            
            for delta in [-5, -3, -2, -1, 1, 2, 3, 5]:
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
                    break
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5
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
    
    # Phase 1b: Arithmetic Progression Seeding
    if time.time() < deadline:
        try:
            A, obj = _arithmetic_progression_seed(random.Random(789), deadline)
            if A and obj < best_obj:
                best_A = A
                best_obj = obj
        except Exception:
            pass
    
    # Fallback
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    # Phase 2: Swap-based local search (preserves multiplicative structure)
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 2.0:
            try:
                refined_A, refined_obj = _swap_local_search(best_A, remaining * 0.5, random.Random(123))
                if refined_obj < best_obj:
                    best_A = refined_A
                    best_obj = refined_obj
            except Exception:
                pass
    
    # Phase 3: Standard local search
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 1.0:
            try:
                refined_A, refined_obj = _local_search(best_A, remaining * 0.6, random.Random(456))
                if refined_obj < best_obj:
                    best_A = refined_A
                    best_obj = refined_obj
            except Exception:
                pass
    
    # Phase 4: Polish
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 0.5:
            try:
                polished_A, polished_obj = _polish_pass(best_A, remaining, random.Random(789))
                if polished_obj < best_obj:
                    best_A = polished_A
                    best_obj = polished_obj
            except Exception:
                pass
    
    # Ensure validity
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return sorted(best_A)
