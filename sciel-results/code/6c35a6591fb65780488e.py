import time
import random
import math
from itertools import combinations

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set))

def compute_metrics_incremental(A, sum_set, prod_set, i, j):
    """
    Incrementally update sum_set and prod_set by adding pairs involving index i or j.
    Returns the new max(|sum_set|, |prod_set|).
    Assumes A is a list and we're adding the pair (i,j) where i<=j.
    """
    a, b = A[i], A[j]
    sum_set.add(a + b)
    prod_set.add(a * b)
    return max(len(sum_set), len(prod_set))

def collision_score_pair(x, y, existing_sums, existing_prods):
    """
    Score how 'good' adding y to a set containing x is, based on collision potential.
    Lower is better (more collisions = smaller sets).
    """
    s = x + y
    p = x * y
    score = 0
    # If this sum/product is already in existing sets, that's a collision (good)
    if s in existing_sums:
        score -= 10  # bonus for sum collision
    if p in existing_prods:
        score -= 10  # bonus for product collision
    return score

def compute_metrics_fast(A):
    """Faster computation using set operations where possible."""
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            sum_set.add(a + A[j])
            prod_set.add(a * A[j])
    return max(len(sum_set), len(prod_set))

def evaluate_structured_set(A):
    """
    Evaluate a structured set and return (score, is_valid).
    A set is valid if it has exactly 20 distinct positive integers <= 10000.
    """
    if len(set(A)) != 20:
        return float('inf'), False
    if any(x < 1 or x > 10000 for x in A):
        return float('inf'), False
    if any(not isinstance(x, int) for x in A):
        return float('inf'), False
    return compute_metrics(A), True

def generate_arithmetic_progressions():
    """
    Generate candidate sets that are arithmetic progressions or subsets thereof.
    For an AP {a, a+d, a+2d, ..., a+(n-1)d}, the sumset has at most 2n-1 elements
    and the product set structure depends on the parameters.
    """
    candidates = []
    
    # Pure APs: {a + i*d for i in range(20)}
    for a in range(1, 100):
        for d in range(1, 200):
            A = [a + i * d for i in range(20)]
            if max(A) <= 10000:
                candidates.append(A)
    
    # Subsets of APs with specific patterns
    # e.g., every other term, or first k terms
    for a in range(1, 50):
        for d in range(1, 500):
            # Take 20 terms from a longer AP
            for start in range(0, 10):
                A = [a + (start + i) * d for i in range(20)]
                if max(A) <= 10000:
                    candidates.append(A)
    
    return candidates

def generate_geometric_progressions():
    """
    Generate candidate sets that are geometric progressions.
    For GP {a, a*r, a*r^2, ..., a*r^(n-1)}, the product set has structure.
    """
    candidates = []
    
    for a in range(1, 20):
        for r in range(2, 10):
            A = [a * (r ** i) for i in range(20)]
            if max(A) <= 10000:
                candidates.append(A)
    
    # Mixed: a * r^i * s^j for small ranges
    for a in range(1, 10):
        for r in range(2, 5):
            for s in range(2, 5):
                if r == s:
                    continue
                A = []
                for i in range(0, 10):
                    for j in range(0, 10 - i):
                        val = a * (r ** i) * (s ** j)
                        if val <= 10000 and val not in A:
                            A.append(val)
                if len(A) >= 20:
                    candidates.append(sorted(A)[:20])
    
    return candidates

def generate_smooth_number_sets():
    """
    Generate sets of smooth numbers (2,3,5,7-smooth) that form structured patterns.
    """
    candidates = []
    
    # All 2,3-smooth numbers up to some limit
    smooth_23 = set()
    for i in range(0, 15):
        for j in range(0, 10):
            val = (1 << i) * (3 ** j)
            if val <= 10000:
                smooth_23.add(val)
    smooth_23 = sorted(smooth_23)
    
    # Take consecutive subsets
    for start in range(0, max(0, len(smooth_23) - 20)):
        A = smooth_23[start:start + 20]
        if len(A) == 20:
            candidates.append(A)
    
    # 2,3,5-smooth
    smooth_235 = set()
    for i in range(0, 15):
        for j in range(0, 10):
            for k in range(0, 8):
                val = (1 << i) * (3 ** j) * (5 ** k)
                if val <= 10000:
                    smooth_235.add(val)
    smooth_235 = sorted(smooth_235)
    
    for start in range(0, max(0, len(smooth_235) - 20)):
        A = smooth_235[start:start + 20]
        if len(A) == 20:
            candidates.append(A)
    
    return candidates

def generate_lattice_sets():
    """
    Generate sets from 2D lattices: {a * r^i * s^j} for i,j in a range.
    This creates dense product sets due to multiplicative structure.
    """
    candidates = []
    
    for a in range(1, 15):
        for r in [2, 3, 4, 5]:
            for s in [2, 3, 4, 5]:
                if r == s:
                    continue
                # Generate all a*r^i*s^j and pick 20
                vals = set()
                for i in range(0, 12):
                    for j in range(0, 12):
                        val = a * (r ** i) * (s ** j)
                        if val <= 10000:
                            vals.add(val)
                vals = sorted(vals)
                if len(vals) >= 20:
                    # Take various subsets of 20
                    candidates.append(vals[:20])
                    if len(vals) >= 40:
                        # Skip every other
                        candidates.append(vals[::2][:20])
    
    return candidates

def local_search_structured(initial_set, deadline, max_iter=5000):
    """
    Local search with structured moves: replace one element with a nearby smooth number
    or adjust the set to maintain arithmetic/geometric structure.
    """
    current = sorted(initial_set)
    current_score = compute_metrics(current)
    best = current[:]
    best_score = current_score
    
    current_set = set(current)
    
    # Generate pool of replacement candidates: smooth numbers near current elements
    pool = set()
    for x in current:
        # Add nearby smooth numbers
        for offset in range(-50, 51):
            val = x + offset
            if val < 1 or val > 10000:
                continue
            # Check if smooth (2,3,5,7-smooth)
            v = val
            for p in [2, 3, 5, 7]:
                while v % p == 0:
                    v //= p
            if v == 1:
                pool.add(val)
        # Also add pure smooth numbers in range
        for i in range(0, 12):
            for j in range(0, 8):
                for k in range(0, 6):
                    val = (1 << i) * (3 ** j) * (5 ** k)
                    if 1 <= val <= 10000:
                        pool.add(val)
    
    pool = sorted(pool)
    
    iteration = 0
    while iteration < max_iter and time.time() < deadline:
        iteration += 1
        improved = False
        
        for i in range(len(current)):
            old_val = current[i]
            
            # Try replacements
            for new_val in pool:
                if new_val in current_set or new_val == old_val:
                    continue
                
                current[i] = new_val
                current_set.discard(old_val)
                current_set.add(new_val)
                
                new_score = compute_metrics(current)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best = current[:]
                else:
                    current[i] = old_val
                    current_set.discard(new_val)
                    current_set.add(old_val)
            
            if not improved and time.time() > deadline - 5:
                break
        
        if not improved:
            break
    
    return best, best_score

def greedy_collision_minimization(pool, deadline, k=20):
    """
    Greedy construction that builds a set of size k from pool by always picking
    the next element that minimizes the growth of max(|A+A|, |A*A|).
    This is a collision-minimization heuristic inspired by graph density.
    """
    if len(pool) < k:
        return sorted(pool)
    
    # Start with the smallest element (often 1 gives good structure)
    A = [pool[0]]
    sum_set = set()
    prod_set = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            sum_set.add(A[i] + A[j])
            prod_set.add(A[i] * A[j])
    
    used = {A[0]}
    
    while len(A) < k and time.time() < deadline:
        best_candidate = None
        best_score = float('inf')
        best_new_sums = None
        best_new_prods = None
        
        for cand in pool:
            if cand in used:
                continue
            
            # Compute new sums and products
            new_sums = set()
            new_prods = set()
            for a in A:
                new_sums.add(a + cand)
                new_prods.add(a * cand)
            
            # Total sizes
            total_sums = len(sum_set | new_sums)
            total_prods = len(prod_set | new_prods)
            score = max(total_sums, total_prods)
            
            if score < best_score:
                best_score = score
                best_candidate = cand
                best_new_sums = new_sums
                best_new_prods = new_prods
        
        if best_candidate is None:
            break
        
        A.append(best_candidate)
        used.add(best_candidate)
        sum_set |= best_new_sums
        prod_set |= best_new_prods
    
    return sorted(A)

def local_search_with_collision_pool(initial_set, deadline, max_iter=5000):
    """
    Enhanced local search that uses a collision-aware pool and tracks
    incremental sum/product sets for faster evaluation.
    """
    current = sorted(initial_set)
    n = len(current)
    
    # Build initial sum and product sets
    sum_set = set()
    prod_set = set()
    for i in range(n):
        for j in range(i, n):
            sum_set.add(current[i] + current[j])
            prod_set.add(current[i] * current[j])
    
    current_score = max(len(sum_set), len(prod_set))
    best = current[:]
    best_score = current_score
    best_sum_set = sum_set.copy()
    best_prod_set = prod_set.copy()
    
    current_set = set(current)
    
    # Generate collision-aware pool
    pool = set()
    # Add smooth numbers
    for i in range(0, 14):
        for j in range(0, 10):
            for k in range(0, 8):
                val = (1 << i) * (3 ** j) * (5 ** k)
                if 1 <= val <= 10000:
                    pool.add(val)
    # Add numbers near current elements
    for x in current:
        for offset in range(-30, 31):
            val = x + offset
            if 1 <= val <= 10000:
                pool.add(val)
    
    # Filter pool to exclude current elements
    pool = [x for x in pool if x not in current_set]
    pool.sort()
    
    # Limit pool size for speed
    if len(pool) > 500:
        # Keep elements that are "close" to existing ones
        pool = [x for x in pool if any(abs(x - y) < 200 for y in current)][:500]
    
    iteration = 0
    while iteration < max_iter and time.time() < deadline:
        iteration += 1
        improved = False
        
        for i in range(n):
            old_val = current[i]
            
            # Compute sums and products involving index i
            old_sums_with_i = set()
            old_prods_with_i = set()
            for j in range(n):
                old_sums_with_i.add(old_val + current[j])
                old_prods_with_i.add(old_val * current[j])
            
            for new_val in pool:
                if new_val in current_set or new_val == old_val:
                    continue
                
                # Compute new sums and products involving new_val
                new_sums_with_val = set()
                new_prods_with_val = set()
                for j in range(n):
                    if j != i:
                        new_sums_with_val.add(new_val + current[j])
                    else:
                        new_sums_with_val.add(new_val + new_val)
                    new_prods_with_val.add(new_val * (current[j] if j != i else new_val))
                
                # Estimate new total sizes
                # Sums not involving i remain
                sums_without_i = sum_set - old_sums_with_i
                prods_without_i = prod_set - old_prods_with_i
                
                total_sums = len(sums_without_i | new_sums_with_val)
                total_prods = len(prods_without_i | new_prods_with_val)
                new_score = max(total_sums, total_prods)
                
                if new_score < current_score:
                    # Accept the move
                    current[i] = new_val
                    current_set.discard(old_val)
                    current_set.add(new_val)
                    
                    # Update sets
                    sum_set = sums_without_i | new_sums_with_val
                    prod_set = prods_without_i | new_prods_with_val
                    current_score = new_score
                    improved = True
                    
                    if new_score < best_score:
                        best_score = new_score
                        best = current[:]
                        best_sum_set = sum_set.copy()
                        best_prod_set = prod_set.copy()
                    
                    # Update pool
                    if old_val in pool:
                        pool.remove(old_val)
                    if new_val not in pool and new_val not in current_set:
                        pool.append(new_val)
                        pool.sort()
                    break  # Move to next i
                
            if time.time() > deadline - 5:
                break
        
        if not improved:
            break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s limit
    
    best_set = None
    best_score = float('inf')
    
    # Phase 0: Greedy collision minimization from a rich pool
    if time.time() < deadline - 20:
        # Build a diverse pool of candidate numbers
        pool = set()
        # Smooth numbers
        for i in range(0, 14):
            for j in range(0, 10):
                for k in range(0, 8):
                    val = (1 << i) * (3 ** j) * (5 ** k)
                    if 1 <= val <= 10000:
                        pool.add(val)
        # Small integers
        for x in range(1, 101):
            pool.add(x)
        # Some structured numbers
        for a in range(1, 20):
            for d in range(1, 50):
                for i in range(20):
                    val = a + i * d
                    if 1 <= val <= 10000:
                        pool.add(val)
        
        pool_list = sorted(pool)
        greedy_result = greedy_collision_minimization(pool_list, deadline - 5, k=20)
        greedy_score = compute_metrics(greedy_result)
        
        if greedy_score < best_score:
            best_score = greedy_score
            best_set = greedy_result
    
    # Phase 1: Evaluate structured families
    structured_families = [
        generate_arithmetic_progressions,
        generate_geometric_progressions,
        generate_smooth_number_sets,
        generate_lattice_sets,
    ]
    
    for gen_func in structured_families:
        if time.time() > deadline - 15:
            break
        
        candidates = gen_func()
        
        # Evaluate candidates, keeping track of best
        for cand in candidates:
            if time.time() > deadline - 15:
                break
            
            score, valid = evaluate_structured_set(cand)
            if valid and score < best_score:
                best_score = score
                best_set = sorted(cand)
    
    # Phase 2: Local search with collision-aware pool
    if best_set is not None and time.time() < deadline - 10:
        best_set, best_score = local_search_with_collision_pool(best_set, deadline, max_iter=8000)
    
    # Phase 3: Additional local search if time permits
    if best_set is not None and time.time() < deadline - 5:
        best_set, best_score = local_search_structured(best_set, deadline, max_iter=3000)
    
    # Final validation
    if best_set is None:
        best_set = list(range(1, 21))
    
    return sorted(best_set)
