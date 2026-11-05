"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _set_sizes(A):
    """Compute |A+A| and |A*A| for a list A of distinct integers."""
    sums = set()
    prods = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(i, len(A)):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return len(sums), len(prods)

def _objective(A):
    s, p = _set_sizes(A)
    return max(s, p)

def _greedy_construct(n, max_val, deadline, start_elements=None):
    """
    Greedy constructive: build set element by element,
    choosing each new element to minimize incremental max(|A+A|,|A*A|).
    """
    if start_elements is None:
        start_elements = []
    
    A = sorted(start_elements)
    if not A:
        return None
    
    sums = set()
    prods = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(i, len(A)):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    
    used = set(A)
    
    while len(A) < n:
        best_candidate = None
        best_cost = float('inf')
        
        # Candidate pool: small numbers (for sum compactness) and 
        # numbers that create product overlap
        candidates = set()
        
        # Small integers 1..60
        for v in range(1, 61):
            if v not in used:
                candidates.add(v)
        
        # Numbers related to existing elements (products, sums)
        for a in A:
            for b in A:
                p = a * b
                if 1 <= p <= max_val and p not in used:
                    candidates.add(p)
                s = a + b
                if 1 <= s <= max_val and s not in used:
                    candidates.add(s)
            # Multiples
            for m in range(2, 10):
                v = a * m
                if 1 <= v <= max_val and v not in used:
                    candidates.add(v)
        
        # Some random candidates for diversity
        for _ in range(30):
            v = random.randint(1, max_val)
            if v not in used:
                candidates.add(v)
        
        # Log-spaced candidates
        for e in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for mult in range(1, 11):
                v = e * mult
                if 1 <= v <= max_val and v not in used:
                    candidates.add(v)
        
        # Evaluate candidates (limit to keep time bounded)
        cand_list = sorted(candidates)
        if len(cand_list) > 200:
            # Sample to keep evaluation fast
            random.shuffle(cand_list)
            cand_list = cand_list[:200]
        
        for v in cand_list:
            if v in used:
                continue
            # Compute new sums and prods incrementally
            new_sums_count = len(sums)
            new_prods_count = len(prods)
            new_sum_vals = 0
            new_prod_vals = 0
            
            for a in A:
                s = a + v
                if s not in sums:
                    new_sum_vals += 1
                p = a * v
                if p not in prods:
                    new_prod_vals += 1
            
            # Also v+v and v*v (diagonal)
            s_diag = v + v
            if s_diag not in sums:
                new_sum_vals += 1
            p_diag = v * v
            if p_diag not in prods:
                new_prod_vals += 1
            
            total_sums = new_sums_count + new_sum_vals
            total_prods = new_prods_count + new_prod_vals
            cost = max(total_sums, total_prods)
            
            if cost < best_cost:
                best_cost = cost
                best_candidate = v
        
        if best_candidate is None:
            # Fallback: pick smallest unused
            v = 1
            while v in used:
                v += 1
            best_candidate = v
        
        A.append(best_candidate)
        used.add(best_candidate)
        A.sort()
        
        # Update sums and prods
        for i in range(len(A) - 1):
            ai = A[i]
            sums.add(ai + best_candidate)
            prods.add(ai * best_candidate)
        sums.add(best_candidate + best_candidate)
        prods.add(best_candidate * best_candidate)
        
        if time.time() > deadline:
            break
    
    if len(A) < n:
        # Fill remaining
        v = 1
        while len(A) < n:
            if v not in used:
                A.append(v)
                used.add(v)
            v += 1
        A.sort()
    
    return A

def _local_refine(A, deadline, max_rounds=5000):
    """
    Local refinement: try replacing each element with a better one.
    """
    n = len(A)
    current_cost = _objective(A)
    best_A = A[:]
    best_cost = current_cost
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(n):
            if time.time() > deadline:
                break
            
            old_val = A[idx]
            # Try nearby values and structured candidates
            trial_vals = set()
            # Nearby
            for d in range(-5, 6):
                v = old_val + d
                if 1 <= v <= 10000:
                    trial_vals.add(v)
            # Small values
            for v in range(1, 25):
                trial_vals.add(v)
            # Products of pairs in A (excluding idx)
            others = [A[i] for i in range(n) if i != idx]
            for a in others[:10]:
                for m in range(2, 8):
                    v = a * m
                    if 1 <= v <= 10000:
                        trial_vals.add(v)
            
            for v in trial_vals:
                if v == old_val or v in [A[i] for i in range(n) if i != idx]:
                    continue
                A[idx] = v
                new_cost = _objective(A)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = A[:]
                    break
                else:
                    A[idx] = old_val
        
        if not improved:
            break
    
    return best_A, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    n = 20
    max_val = 10000
    
    best_A = list(range(1, 21))
    best_cost = _objective(best_A)
    
    # Strategy 1: Greedy construction with different seeds
    seed_sets = [
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 5, 7],
        [1, 3, 5, 7],
        [1, 2, 4, 8],
        [1, 2, 3, 5, 7],
    ]
    
    for seed in seed_sets:
        if time.time() > deadline:
            break
        A = _greedy_construct(n, max_val, deadline, start_elements=seed)
        if A is None or len(A) != n:
            continue
        cost = _objective(A)
        if cost < best_cost:
            best_cost = cost
            best_A = A[:]
    
    # Strategy 2: Refine the best greedy result
    if time.time() < deadline:
        refined_A, refined_cost = _local_refine(best_A, deadline, max_rounds=3000)
        if refined_cost < best_cost:
            best_cost = refined_cost
            best_A = refined_A
    
    # Strategy 3: Try a few more greedy constructions with randomness
    for trial in range(5):
        if time.time() > deadline:
            break
        random_seed = [random.randint(1, 50) for _ in range(3)]
        random_seed = sorted(set(random_seed))
        A = _greedy_construct(n, max_val, deadline, start_elements=random_seed)
        if A is None or len(A) != n:
            continue
        cost = _objective(A)
        if cost < best_cost:
            best_cost = cost
            best_A = A[:]
        # Quick refine
        if time.time() < deadline:
            refined_A, refined_cost = _local_refine(A, deadline, max_rounds=1000)
            if refined_cost < best_cost:
                best_cost = refined_cost
                best_A = refined_A
    
    # Strategy 4: Final refinement on the best
    if time.time() < deadline:
        refined_A, refined_cost = _local_refine(best_A, deadline, max_rounds=5000)
        if refined_cost < best_cost:
            best_cost = refined_cost
            best_A = refined_A
    
    return sorted(best_A)

# EVOLVE-BLOCK-END
