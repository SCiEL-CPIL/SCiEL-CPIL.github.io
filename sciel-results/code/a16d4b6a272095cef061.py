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

def _generate_lattice_sets(n, max_val, max_i=6, max_j=4):
    """
    Generate sets from the 2^i * 3^j lattice.
    This structurally constrains |A*A| to O(k^2) since products
    of 2^i*3^j * 2^k*3^l = 2^(i+k)*3^(j+l) have overlapping exponents.
    """
    results = []
    
    # Generate all lattice points
    lattice = {}
    for i in range(max_i + 1):
        for j in range(max_j + 1):
            val = (2 ** i) * (3 ** j)
            if val <= max_val:
                lattice[val] = (i, j)
    
    lattice_vals = sorted(lattice.keys())
    
    # Try different subsets of the lattice
    # Strategy 1: Consecutive lattice points
    for start in range(len(lattice_vals) - n + 1):
        subset = lattice_vals[start:start + n]
        results.append(subset)
    
    # Strategy 2: Random subsets of lattice
    for _ in range(50):
        if len(lattice_vals) >= n:
            subset = random.sample(lattice_vals, n)
            results.append(sorted(subset))
    
    # Strategy 3: Dense low region
    low_vals = [v for v in lattice_vals if v <= 100]
    if len(low_vals) >= n:
        for _ in range(30):
            subset = random.sample(low_vals, n)
            results.append(sorted(subset))
    
    # Strategy 4: Mix of low and medium
    low_vals = [v for v in lattice_vals if v <= 50]
    med_vals = [v for v in lattice_vals if 50 < v <= 500]
    if len(low_vals) >= n // 2 and len(med_vals) >= n // 2:
        for _ in range(30):
            k_low = random.randint(n // 3, n - n // 3)
            subset = random.sample(low_vals, min(k_low, len(low_vals)))
            subset += random.sample(med_vals, min(n - len(subset), len(med_vals)))
            if len(subset) == n:
                results.append(sorted(set(subset)))
    
    return results

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
        
        # Small integers 1..80
        for v in range(1, 81):
            if v not in used:
                candidates.add(v)
        
        # Lattice points (2^i * 3^j)
        for i in range(0, 8):
            for j in range(0, 5):
                v = (2 ** i) * (3 ** j)
                if 1 <= v <= max_val and v not in used:
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
            for m in range(2, 12):
                v = a * m
                if 1 <= v <= max_val and v not in used:
                    candidates.add(v)
        
        # Some random candidates for diversity
        for _ in range(40):
            v = random.randint(1, max_val)
            if v not in used:
                candidates.add(v)
        
        # Evaluate candidates (limit to keep time bounded)
        cand_list = sorted(candidates)
        if len(cand_list) > 300:
            # Sample to keep evaluation fast
            random.shuffle(cand_list)
            cand_list = cand_list[:300]
        
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

def _local_refine_lattice(A, deadline, max_rounds=2000):
    """
    Local refinement that prefers lattice points (2^i * 3^j) for replacements.
    This exploits the structural advantage of the lattice for product sets.
    """
    n = len(A)
    current_cost = _objective(A)
    best_A = A[:]
    best_cost = current_cost
    
    # Precompute lattice points
    lattice_points = []
    for i in range(0, 8):
        for j in range(0, 5):
            v = (2 ** i) * (3 ** j)
            if v <= 10000:
                lattice_points.append(v)
    lattice_points = sorted(set(lattice_points))
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(n):
            if time.time() > deadline:
                break
            
            old_val = A[idx]
            trial_vals = set()
            
            # Lattice points (high priority)
            trial_vals.update(lattice_points[:80])
            
            # Nearby values
            for d in range(-15, 16):
                v = old_val + d
                if 1 <= v <= 10000:
                    trial_vals.add(v)
            
            # Small values
            for v in range(1, 40):
                trial_vals.add(v)
            
            # Products of pairs in A (excluding idx)
            others = [A[i] for i in range(n) if i != idx]
            for a in others[:15]:
                for m in range(2, 15):
                    v = a * m
                    if 1 <= v <= 10000:
                        trial_vals.add(v)
            
            # Sums of pairs
            for i in range(len(others)):
                for j in range(i + 1, min(i + 8, len(others))):
                    v = others[i] + others[j]
                    if 1 <= v <= 10000:
                        trial_vals.add(v)
            
            # Log-spaced around old_val
            for mult in [0.5, 0.67, 0.75, 1.25, 1.33, 1.5, 2, 3]:
                v = int(old_val * mult)
                if 1 <= v <= 10000:
                    trial_vals.add(v)
            
            # Filter and limit
            other_set = set(A[i] for i in range(n) if i != idx)
            trial_vals = [v for v in trial_vals if v != old_val and v not in other_set and 1 <= v <= 10000]
            
            # Prioritize lattice points and small values
            trial_vals.sort(key=lambda v: (0 if v in lattice_points else 1, v))
            
            if len(trial_vals) > 100:
                trial_vals = trial_vals[:100]
            
            for v in trial_vals:
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
                continue
        
        if not improved:
            # Try random restarts within the neighborhood
            for _ in range(5):
                if time.time() > deadline:
                    break
                idx = random.randint(0, n - 1)
                old_val = A[idx]
                other_set = set(A[i] for i in range(n) if i != idx)
                
                # Try a random lattice point or small value
                if random.random() < 0.6:
                    v = random.choice(lattice_points[:60])
                else:
                    v = random.randint(1, 100)
                
                if v not in other_set and v != old_val and 1 <= v <= 10000:
                    A[idx] = v
                    new_cost = _objective(A)
                    if new_cost <= current_cost:
                        if new_cost < current_cost:
                            current_cost = new_cost
                            improved = True
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_A = A[:]
                    else:
                        A[idx] = old_val
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
    
    # Strategy 1: Lattice-based construction (2^i * 3^j)
    lattice_sets = _generate_lattice_sets(n, max_val)
    for ls in lattice_sets[:30]:
        if time.time() > deadline:
            break
        if len(set(ls)) != n:
            continue
        cost = _objective(ls)
        if cost < best_cost:
            best_cost = cost
            best_A = ls[:]
        # Quick lattice-aware refine
        if time.time() < deadline:
            refined_A, refined_cost = _local_refine_lattice(ls, deadline, max_rounds=500)
            if refined_cost < best_cost:
                best_cost = refined_cost
                best_A = refined_A
    
    # Strategy 2: Greedy construction with different seeds
    seed_sets = [
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 5, 7],
        [1, 3, 5, 7],
        [1, 2, 4, 8],
        [1, 2, 3, 5, 7],
        [1, 2, 4, 8, 16],
        [1, 3, 9, 27],
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
    
    # Strategy 3: Lattice-aware refinement on the best result
    if time.time() < deadline:
        refined_A, refined_cost = _local_refine_lattice(best_A, deadline, max_rounds=2000)
        if refined_cost < best_cost:
            best_cost = refined_cost
            best_A = refined_A
    
    # Strategy 4: Try more greedy constructions with randomness
    for trial in range(10):
        if time.time() > deadline:
            break
        # Mix of lattice and random seeds
        lattice_seed = [(2 ** i) * (3 ** j) for i in range(3) for j in range(2) if (2 ** i) * (3 ** j) <= 100]
        random_seed = [random.choice(lattice_seed) for _ in range(3)]
        random_seed = sorted(set(random_seed))
        if len(random_seed) < 3:
            random_seed = [1, 2, 3]
        A = _greedy_construct(n, max_val, deadline, start_elements=random_seed)
        if A is None or len(A) != n:
            continue
        cost = _objective(A)
        if cost < best_cost:
            best_cost = cost
            best_A = A[:]
        # Quick lattice-aware refine
        if time.time() < deadline:
            refined_A, refined_cost = _local_refine_lattice(A, deadline, max_rounds=800)
            if refined_cost < best_cost:
                best_cost = refined_cost
                best_A = refined_A
    
    # Strategy 5: Final lattice-aware refinement on the best
    if time.time() < deadline:
        refined_A, refined_cost = _local_refine_lattice(best_A, deadline, max_rounds=3000)
        if refined_cost < best_cost:
            best_cost = refined_cost
            best_A = refined_A
    
    return sorted(best_A)

# EVOLVE-BLOCK-END
