import time
import random
import numpy as np

def _score(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    n = len(A)
    sums = set()
    prods = set()
    # Optimization: Use lists and set conversion, or direct addition.
    # Direct addition is usually fine for n=20.
    for i in range(n):
        ai = A[i]
        for j in range(n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _is_valid(A, n=20, maximum=10000):
    """Check if A has n distinct positive integers <= maximum."""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 for x in A):
        return False
    if any(x > maximum for x in A):
        return False
    return True

def _generate_from_params(a, r, n=20, max_val=10000):
    """
    Generate a candidate set from geometric progression parameters.
    A_k = round(a * r^k)
    Handles duplicates and bounds by nudging.
    """
    A = []
    existing = set()
    for k in range(n):
        val = int(round(a * (r ** k)))
        # Ensure positive and within bounds
        if val < 1:
            val = 1
        if val > max_val:
            val = max_val
            
        # Resolve collisions
        original_val = val
        attempts = 0
        while val in existing and attempts < 20:
            # Try to nudge up or down
            # Simple strategy: increment until unique, or decrement if possible
            val += 1
            if val > max_val:
                val = original_val - 1
            attempts += 1
        
        if val in existing:
            # If still colliding, just pick a random unused number near this value
            # This is a fallback, shouldn't happen often with good r
            while True:
                val = random.randint(max(1, original_val - 10), min(max_val, original_val + 10))
                if val not in existing:
                    break
                if val > max_val:
                    val = max_val - 1
                    if val not in existing:
                        break
                if val < 1:
                    val = 1
                    if val not in existing:
                        break
        
        A.append(val)
        existing.add(val)
        
    # Sort to maintain canonical form for consistency, though order doesn't matter for sets
    A.sort()
    
    # Final validation check
    if not _is_valid(A, n, max_val):
        return None
    return A

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    rng = random.Random(42)
    
    best_A = None
    best_score = float('inf')
    
    # Strategy: Geometric Annealing in Log-Space
    # We represent the set as roughly a * r^k.
    # We optimize a and r.
    # Log(r) is more natural for scaling.
    
    # Initial guess for r:
    # We want 20 terms in [1, 10000].
    # a * r^19 <= 10000. If a=1, r <= 10000^(1/19) ~ 1.85.
    # If a is larger, r must be smaller.
    # Let's sample initial (a, r) pairs.
    
    # Phase 1: Multi-start Geometric Annealing
    # We will run several independent annealing processes on (log_a, log_r).
    
    num_starts = 20
    steps_per_start = 5000
    
    for s in range(num_starts):
        if time.time() > deadline:
            break
            
        # Random initial parameters
        # log_a in [0, log(100)] -> a in [1, 100]
        log_a = rng.uniform(0, np.log(100))
        # log_r in [log(1.05), log(1.85)] -> r in [1.05, 1.85]
        log_r = rng.uniform(np.log(1.05), np.log(1.85))
        
        a = np.exp(log_a)
        r = np.exp(log_r)
        
        A = _generate_from_params(a, r)
        if A is None:
            continue
            
        current_score = _score(A)
        
        # Annealing schedule
        T_start = 100.0
        T_end = 1.0
        decay = (T_end / T_start) ** (1.0 / steps_per_start)
        T = T_start
        
        current_a = a
        current_r = r
        current_A = A
        current_s = current_score
        
        for step in range(steps_per_start):
            if time.time() > deadline - 2:
                break
                
            # Perturb log_a and log_r
            # Step size decreases with T
            scale = T / T_start
            delta_log_a = rng.gauss(0, 0.1 * scale)
            delta_log_r = rng.gauss(0, 0.05 * scale)
            
            new_log_a = np.log(current_a) + delta_log_a
            new_log_r = np.log(current_r) + delta_log_r
            
            # Clamp to reasonable ranges
            # a should be at least 1
            if new_log_a < 0:
                new_log_a = 0
            # r should be > 1 and not too large
            min_log_r = np.log(1.01)
            max_log_r = np.log(2.0) # Allow slightly larger if a is small
            if new_log_r < min_log_r:
                new_log_r = min_log_r
            if new_log_r > max_log_r:
                new_log_r = max_log_r
                
            new_a = np.exp(new_log_a)
            new_r = np.exp(new_log_r)
            
            new_A = _generate_from_params(new_a, new_r)
            if new_A is None:
                continue
                
            new_s = _score(new_A)
            
            delta = new_s - current_s
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_a = new_a
                current_r = new_r
                current_A = new_A
                current_s = new_s
                
                if current_s < best_score:
                    best_score = current_s
                    best_A = current_A[:]
            
            T *= decay
            
        # End of start s
        if current_s < best_score:
            best_score = current_s
            best_A = current_A[:]
            
    # Phase 2: Local Polish on the best candidate found
    # The geometric annealing might leave the set slightly suboptimal due to rounding.
    # We perform a standard local search on the integers themselves, but constrained
    # to stay close to the geometric structure to avoid destroying the product set benefits.
    
    if best_A is not None and time.time() < deadline - 5:
        current_A = best_A[:]
        current_s = best_score
        
        # Simple local search: try changing each element by small deltas
        # But bias the deltas towards the geometric neighbors?
        # For now, just standard small perturbations, but more aggressive steps.
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            indices = list(range(20))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 2:
                    break
                    
                old_val = current_A[i]
                # Try a few random changes
                # Larger steps than before, since we have a good structure
                for _ in range(10):
                    # Perturb by up to 10% of the value or 50
                    delta_range = max(50, int(old_val * 0.1))
                    new_val = old_val + rng.randint(-delta_range, delta_range)
                    
                    if new_val <= 0 or new_val > 10000:
                        continue
                    if new_val in current_A and new_val != old_val:
                        continue
                        
                    new_A = current_A[:]
                    new_A[i] = new_val
                    new_A.sort()
                    
                    s = _score(new_A)
                    
                    if s < current_s:
                        current_A = new_A
                        current_s = s
                        improved = True
                        break # Move to next index
            
            if current_s < best_score:
                best_score = current_s
                best_A = current_A[:]
                
    # Fallback
    if best_A is None:
        # Construct a basic GP
        best_A = _generate_from_params(1, 1.2)
        if best_A is None:
            best_A = list(range(1, 21))
            
    return sorted(best_A)

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    A = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Set: {A}")
    print(f"Score: {_score(A)}")
