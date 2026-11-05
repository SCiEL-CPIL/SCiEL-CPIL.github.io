"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _compute_cost(A):
    """
    Compute max(|A+A|, |A*A|) for a list A of distinct integers.
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
    return max(len(sums), len(prods))

def _compute_cost_detailed(A):
    """Return (sum_size, prod_size, max_size) for a set A."""
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    ss, ps = len(sums), len(prods)
    return (ss, ps, max(ss, ps))

def _generate_23_smooth(n=20, max_val=10000):
    """
    Generate n distinct values of the form 2^i * 3^j <= max_val.
    These have structured product sets since (2^i 3^j)(2^k 3^l) = 2^(i+k) 3^(j+l).
    """
    vals = set()
    for i in range(15):  # 2^14 = 16384 > 10000, so i up to 13
        for j in range(10):  # 3^9 = 19683 > 10000, so j up to 8
            v = (2 ** i) * (3 ** j)
            if 1 <= v <= max_val:
                vals.add(v)
    # Sort and take first n
    sorted_vals = sorted(vals)
    if len(sorted_vals) >= n:
        return sorted_vals[:n]
    # Fill with small integers if needed
    candidate = 1
    while len(sorted_vals) < n:
        if candidate not in vals:
            sorted_vals.append(candidate)
            vals.add(candidate)
        candidate += 1
    return sorted(sorted_vals)[:n]

def _generate_23_smooth_random(n=20, max_val=10000):
    """
    Randomly select n distinct values from the set of 2^i * 3^j <= max_val.
    """
    vals = set()
    for i in range(15):
        for j in range(10):
            v = (2 ** i) * (3 ** j)
            if 1 <= v <= max_val:
                vals.add(v)
    vals_list = sorted(vals)
    if len(vals_list) >= n:
        return sorted(random.sample(vals_list, n))
    # Fallback
    candidate = 1
    while len(vals_list) < n:
        if candidate not in vals:
            vals_list.append(candidate)
            vals.add(candidate)
        candidate += 1
    return sorted(vals_list[:n])

def _generate_hybrid_smooth(n=20, max_val=10000):
    """
    Mix of 2^i * 3^j values with some small consecutive integers.
    """
    A = set()
    # Add some small consecutive values
    for i in range(1, 6):
        A.add(i)
    # Add 2^i * 3^j values
    for i in range(15):
        for j in range(10):
            if len(A) >= n:
                break
            v = (2 ** i) * (3 ** j)
            if 1 <= v <= max_val and v not in A:
                A.add(v)
        if len(A) >= n:
            break
    # Fill if needed
    candidate = 1
    while len(A) < n:
        if candidate not in A:
            A.add(candidate)
        candidate += 1
    return sorted(list(A)[:n])

def _local_search_smooth(exponents, n, max_val, deadline, best_A, best_cost, max_steps=5000):
    """
    Local search in the (i,j) exponent space for 2^i * 3^j representation.
    exponents is a list of (i, j) pairs.
    """
    def exps_to_vals(exps):
        vals = []
        seen = set()
        for i, j in exps:
            v = (2 ** i) * (3 ** j)
            if v <= max_val and v not in seen:
                vals.append(v)
                seen.add(v)
        return vals
    
    current_exps = exponents[:]
    current_A = exps_to_vals(current_exps)
    
    # If we don't have enough distinct values, fill with small ints
    if len(current_A) < n:
        seen = set(current_A)
        candidate = 1
        while len(current_A) < n:
            if candidate not in seen:
                current_A.append(candidate)
                seen.add(candidate)
            candidate += 1
        current_A = sorted(current_A[:n])
        # Convert back to exponents (approximate)
        current_exps = []
        for v in current_A:
            # Find best (i,j) representation
            best_i, best_j = 0, 0
            for i in range(15):
                for j in range(10):
                    if (2 ** i) * (3 ** j) == v:
                        best_i, best_j = i, j
                        break
                if (2 ** best_i) * (3 ** best_j) == v:
                    break
            current_exps.append((best_i, best_j))
    
    current_cost = _compute_cost(current_A)
    if current_cost < best_cost:
        best_cost = current_cost
        best_A = current_A[:]
    
    no_improve = 0
    
    while time.time() < deadline and no_improve < max_steps:
        idx = random.randint(0, len(current_exps) - 1)
        old_i, old_j = current_exps[idx]
        
        r = random.random()
        if r < 0.4:
            # Small perturbation in exponent space
            di = random.randint(-1, 1)
            dj = random.randint(-1, 1)
        elif r < 0.7:
            # Medium perturbation
            di = random.randint(-2, 2)
            dj = random.randint(-2, 2)
        else:
            # Larger jump
            di = random.randint(-3, 3)
            dj = random.randint(-3, 3)
        
        new_i = max(0, old_i + di)
        new_j = max(0, old_j + dj)
        
        if new_i == old_i and new_j == old_j:
            no_improve += 1
            continue
        
        # Apply
        current_exps[idx] = (new_i, new_j)
        new_A = exps_to_vals(current_exps)
        
        # If we lost distinctness, try to fix
        if len(new_A) < n:
            seen = set(new_A)
            candidate = 1
            while len(new_A) < n:
                if candidate not in seen:
                    new_A.append(candidate)
                    seen.add(candidate)
                candidate += 1
            new_A = sorted(new_A[:n])
        
        new_cost = _compute_cost(new_A)
        
        if new_cost < current_cost:
            current_cost = new_cost
            current_A = new_A
            no_improve = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_A = current_A[:]
        else:
            current_exps[idx] = (old_i, old_j)
            no_improve += 1
    
    return best_A, best_cost, current_A, current_cost

def _local_search_generic(current_A, n, max_val, deadline, best_A, best_cost, max_steps=3000):
    """
    Generic local search on the value space.
    """
    current_cost = _compute_cost(current_A)
    no_improve = 0
    
    while time.time() < deadline and no_improve < max_steps:
        idx = random.randint(0, n - 1)
        old_val = current_A[idx]
        
        r = random.random()
        if r < 0.3:
            delta = random.randint(-10, 10)
            new_val = old_val + delta
        elif r < 0.5:
            delta = random.randint(-100, 100)
            new_val = old_val + delta
        elif r < 0.7:
            if old_val <= 10:
                new_val = random.randint(1, 50)
            elif old_val <= 100:
                new_val = random.randint(10, 500)
            elif old_val <= 1000:
                new_val = random.randint(100, 5000)
            else:
                new_val = random.randint(1000, max_val)
        else:
            new_val = random.randint(1, max_val)
        
        if new_val < 1:
            new_val = 1
        elif new_val > max_val:
            new_val = max_val
            
        if new_val == old_val:
            no_improve += 1
            continue
        
        is_distinct = True
        for i, v in enumerate(current_A):
            if i != idx and v == new_val:
                is_distinct = False
                break
        
        if not is_distinct:
            no_improve += 1
            continue
        
        current_A[idx] = new_val
        new_cost = _compute_cost(current_A)
        
        if new_cost < current_cost:
            current_cost = new_cost
            no_improve = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_A = current_A[:]
        else:
            current_A[idx] = old_val
            no_improve += 1
    
    return best_A, best_cost, current_A, current_cost

def _simulated_annealing(current_A, n, max_val, deadline, best_A, best_cost, max_steps=2000):
    """
    Simulated annealing to escape local minima.
    """
    current_cost = _compute_cost(current_A)
    temperature = 10.0
    cooling = 0.995
    
    for _ in range(max_steps):
        if time.time() >= deadline:
            break
        
        idx = random.randint(0, n - 1)
        old_val = current_A[idx]
        
        delta = random.randint(-20, 20)
        new_val = old_val + delta
        if new_val < 1:
            new_val = 1
        elif new_val > max_val:
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
        new_cost = _compute_cost(current_A)
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_A = current_A[:]
        else:
            current_A[idx] = old_val
        
        temperature *= cooling
    
    return best_A, best_cost, current_A, current_cost

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    n = 20
    max_val = 10000
    
    best_A = None
    best_cost = float('inf')
    
    # Strategy 1: 2^i * 3^j smooth numbers
    smooth_A = _generate_23_smooth(n, max_val)
    cost = _compute_cost(smooth_A)
    if cost < best_cost:
        best_cost = cost
        best_A = smooth_A[:]
    
    # Local search in exponent space from smooth start
    exps = []
    for v in smooth_A:
        best_i, best_j = 0, 0
        found = False
        for i in range(15):
            for j in range(10):
                if (2 ** i) * (3 ** j) == v:
                    best_i, best_j = i, j
                    found = True
                    break
            if found:
                break
        exps.append((best_i, best_j))
    
    best_A, best_cost, _, _ = _local_search_smooth(exps, n, max_val, deadline, best_A, best_cost, max_steps=3000)
    
    # Strategy 2: Random smooth selections
    for _ in range(5):
        if time.time() >= deadline - 5:
            break
        rand_smooth = _generate_23_smooth_random(n, max_val)
        cost = _compute_cost(rand_smooth)
        if cost < best_cost:
            best_cost = cost
            best_A = rand_smooth[:]
        
        # Local search from this start
        exps = []
        for v in rand_smooth:
            best_i, best_j = 0, 0
            found = False
            for i in range(15):
                for j in range(10):
                    if (2 ** i) * (3 ** j) == v:
                        best_i, best_j = i, j
                        found = True
                        break
                if found:
                    break
            exps.append((best_i, best_j))
        
        best_A, best_cost, _, _ = _local_search_smooth(exps, n, max_val, deadline, best_A, best_cost, max_steps=1500)
    
    # Strategy 3: Hybrid smooth + small ints
    if time.time() < deadline - 10:
        hybrid_A = _generate_hybrid_smooth(n, max_val)
        cost = _compute_cost(hybrid_A)
        if cost < best_cost:
            best_cost = cost
            best_A = hybrid_A[:]
        
        # Generic local search from hybrid
        exps = []
        for v in hybrid_A:
            best_i, best_j = 0, 0
            found = False
            for i in range(15):
                for j in range(10):
                    if (2 ** i) * (3 ** j) == v:
                        best_i, best_j = i, j
                        found = True
                        break
                if found:
                    break
            exps.append((best_i, best_j))
        
        best_A, best_cost, _, _ = _local_search_smooth(exps, n, max_val, deadline, best_A, best_cost, max_steps=1500)
    
    # Strategy 4: Generic local search on best found so far
    if time.time() < deadline - 5 and best_A is not None:
        best_A, best_cost, _, _ = _local_search_generic(best_A[:], n, max_val, deadline, best_A, best_cost, max_steps=2000)
    
    # Strategy 5: Simulated annealing
    if time.time() < deadline - 3 and best_A is not None:
        best_A, best_cost, _, _ = _simulated_annealing(best_A[:], n, max_val, deadline, best_A, best_cost, max_steps=1000)
    
    # Final polish
    if time.time() < deadline - 1 and best_A is not None:
        best_A, best_cost, _, _ = _local_search_generic(best_A[:], n, max_val, deadline, best_A, best_cost, max_steps=500)
    
    # Ensure valid output
    if best_A is None:
        best_A = list(range(1, n + 1))
    
    return best_A

# EVOLVE-BLOCK-END
