"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _compute_cost(A):
    """
    Compute max(|A+A|, |A*A|) for a list A of distinct integers.
    Returns (cost, sum_size, prod_size) for more detailed feedback.
    """
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    s = len(sums)
    p = len(prods)
    return max(s, p), s, p

def _compute_cost_simple(A):
    cost, _, _ = _compute_cost(A)
    return cost

def _generate_ap(start=1, step=1, n=20, max_val=10000):
    A = [start + i * step for i in range(n)]
    if max(A) <= max_val:
        return sorted(A)
    return None

def _generate_gp(n=20, max_val=10000):
    # Generate a geometric progression as long as possible, fill rest with small distinct ints
    A = set()
    val = 1
    while len(A) < n and val <= max_val:
        A.add(val)
        val *= 2
    if len(A) >= n:
        return sorted(list(A)[:n])
    candidate = 1
    while len(A) < n:
        if candidate not in A:
            A.add(candidate)
        candidate += 1
    return sorted(A)

def _generate_random(n=20, max_val=10000):
    return sorted(random.sample(range(1, max_val + 1), n))

def _generate_hybrid(n=20, max_val=10000):
    # Mix of small AP and large GP
    A = set()
    for i in range(1, 11):
        A.add(i)
    base = 100
    i = 1
    while len(A) < n:
        v = base * i
        if v <= max_val and v not in A:
            A.add(v)
        i += 1
    if len(A) >= n:
        return sorted(list(A)[:n])
    candidate = 1
    while len(A) < n:
        if candidate not in A:
            A.add(candidate)
        candidate += 1
    return sorted(A)

def _generate_small_cluster(n=20, max_val=10000):
    """All small values 1..n - minimizes product set, maximizes sum set."""
    return list(range(1, n + 1))

def _generate_log_spaced(n=20, max_val=10000):
    """Log-spaced values to balance sum and product sets."""
    import math
    A = set()
    # Use powers of small bases
    for base in [2, 3, 5]:
        val = base
        while val <= max_val and len(A) < n:
            A.add(val)
            val *= base
    if len(A) >= n:
        return sorted(list(A)[:n])
    # Fill with intermediate values
    candidate = 1
    while len(A) < n:
        if candidate not in A:
            A.add(candidate)
        candidate += 1
    return sorted(A)

def _generate_mixed_scale(n=20, max_val=10000):
    """Mix of very small (1-5) and medium (10-50) values."""
    A = set()
    # Very small
    for v in [1, 2, 3, 4, 5]:
        A.add(v)
    # Medium range
    for v in range(10, 60, 3):
        if len(A) >= n:
            break
        if v not in A:
            A.add(v)
    if len(A) >= n:
        return sorted(list(A)[:n])
    candidate = 6
    while len(A) < n:
        if candidate not in A:
            A.add(candidate)
        candidate += 1
    return sorted(A)

def _generate_prime_cluster(n=20, max_val=10000):
    """Small primes - products are distinct but sums may overlap."""
    import math
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    
    A = set()
    v = 2
    while len(A) < n and v <= max_val:
        if is_prime(v):
            A.add(v)
        v += 1
    if len(A) >= n:
        return sorted(list(A)[:n])
    candidate = 1
    while len(A) < n:
        if candidate not in A:
            A.add(candidate)
        candidate += 1
    return sorted(A)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin for 90s limit
    n = 20
    max_val = 10000
    
    # 1. Generate diverse Initial Candidates
    candidates = []
    
    # Structured candidates
    candidates.append(_generate_ap(1, 1, n, max_val))
    candidates.append(_generate_ap(1, 2, n, max_val))
    candidates.append(_generate_ap(10, 1, n, max_val))
    candidates.append(_generate_gp(n, max_val))
    candidates.append(_generate_hybrid(n, max_val))
    candidates.append(_generate_small_cluster(n, max_val))
    candidates.append(_generate_log_spaced(n, max_val))
    candidates.append(_generate_mixed_scale(n, max_val))
    candidates.append(_generate_prime_cluster(n, max_val))
    
    # Random candidates with different distributions
    for _ in range(15):
        candidates.append(_generate_random(n, max_val))
    
    # Filter out None candidates
    candidates = [c for c in candidates if c is not None and len(c) == n]
    
    if not candidates:
        return list(range(1, n + 1))
    
    # Evaluate all candidates first
    best_A = candidates[0]
    best_cost, _, _ = _compute_cost(best_A)
    
    for init_A in candidates[1:]:
        cost, _, _ = _compute_cost(init_A)
        if cost < best_cost:
            best_cost = cost
            best_A = init_A[:]
    
    # 2. Local Search with multiple restarts
    # Sort candidates by initial cost, search best ones first
    scored_candidates = []
    for init_A in candidates:
        cost, _, _ = _compute_cost(init_A)
        scored_candidates.append((cost, init_A))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Search the top candidates
    max_searches = 8
    searches_done = 0
    
    for init_cost, init_A in scored_candidates:
        if time.time() > deadline:
            break
        if searches_done >= max_searches:
            break
        if init_cost > best_cost:
            break  # No point searching worse initial candidates
        
        current_A = init_A[:]
        current_cost = init_cost
        searches_done += 1
        
        no_improve_count = 0
        max_steps_per_candidate = 8000
        
        while time.time() < deadline and no_improve_count < max_steps_per_candidate:
            # Adaptive perturbation strategy
            idx = random.randint(0, n - 1)
            old_val = current_A[idx]
            
            # Choose perturbation type based on value
            r = random.random()
            
            if r < 0.3:
                # Small local perturbation - good for fine-tuning
                delta = random.randint(-20, 20)
                new_val = old_val + delta
            elif r < 0.6:
                # Medium perturbation - moderate exploration
                if old_val < 100:
                    # For small values, try nearby or slightly larger
                    new_val = random.randint(1, max(2, old_val + 50))
                else:
                    # For larger values, try scaling down or nearby
                    if random.random() < 0.5:
                        new_val = random.randint(1, old_val)
                    else:
                        delta = random.randint(-100, 100)
                        new_val = old_val + delta
            elif r < 0.8:
                # Targeted: try to reduce product set by using smaller values
                if old_val > 10:
                    new_val = random.randint(1, min(old_val, 100))
                else:
                    new_val = random.randint(1, 50)
            else:
                # Random exploration
                new_val = random.randint(1, max_val)
            
            # Clamp
            if new_val < 1:
                new_val = 1
            if new_val > max_val:
                new_val = max_val
            
            if new_val == old_val:
                no_improve_count += 1
                continue
            
            # Check distinctness
            is_distinct = True
            for i, v in enumerate(current_A):
                if i != idx and v == new_val:
                    is_distinct = False
                    break
            
            if not is_distinct:
                no_improve_count += 1
                continue
            
            # Apply and evaluate
            current_A[idx] = new_val
            new_cost, new_s, new_p = _compute_cost(current_A)
            
            if new_cost < current_cost:
                current_cost = new_cost
                no_improve_count = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = current_A[:]
            else:
                # Occasional acceptance of equal-cost moves for diversity
                if new_cost == current_cost and random.random() < 0.1:
                    # Accept but don't reset no_improve
                    pass
                else:
                    # Revert
                    current_A[idx] = old_val
                    no_improve_count += 1
                    
                # If we've been stuck, try a larger jump
                if no_improve_count > 500:
                    idx = random.randint(0, n - 1)
                    old_val = current_A[idx]
                    new_val = random.randint(1, max_val)
                    if new_val != old_val:
                        is_distinct = True
                        for i, v in enumerate(current_A):
                            if i != idx and v == new_val:
                                is_distinct = False
                                break
                        if is_distinct:
                            current_A[idx] = new_val
                            new_cost, _, _ = _compute_cost(current_A)
                            if new_cost < current_cost:
                                current_cost = new_cost
                                no_improve_count = 0
                                if current_cost < best_cost:
                                    best_cost = current_cost
                                    best_A = current_A[:]
                            else:
                                current_A[idx] = old_val
                                no_improve_count += 1
                    else:
                        no_improve_count += 1
    
    # 3. Final polish: targeted refinement of the best solution
    if time.time() < deadline - 5:
        current_A = best_A[:]
        current_cost = best_cost
        polish_steps = 0
        max_polish = 5000
        
        while time.time() < deadline - 5 and polish_steps < max_polish:
            polish_steps += 1
            idx = random.randint(0, n - 1)
            old_val = current_A[idx]
            
            # Very small perturbations for polish
            delta = random.randint(-10, 10)
            new_val = old_val + delta
            if new_val < 1:
                new_val = 1
            if new_val > max_val:
                new_val = max_val
            
            if new_val == old_val:
                continue
            
            is_distinct = True
            for i, v in enumerate(current_A):
                if i != idx and v == new_val:
                    is_distinct = False
                    break
            
            if not is_distinct:
                continue
            
            current_A[idx] = new_val
            new_cost, _, _ = _compute_cost(current_A)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = current_A[:]
            else:
                current_A[idx] = old_val
    
    # Ensure sorted output
    best_A = sorted(best_A)
    return best_A

# EVOLVE-BLOCK-END
