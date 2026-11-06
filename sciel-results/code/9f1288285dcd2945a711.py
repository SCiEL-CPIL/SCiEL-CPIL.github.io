import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # Safety margin

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers using NumPy."""
    A_arr = np.array(A, dtype=np.int64)
    n = len(A_arr)
    
    # Generate all pairs (i, j) with i <= j
    # There are n*(n+1)/2 such pairs
    indices = np.triu_indices(n)
    
    # Vectorized sum and product calculation
    sums = A_arr[indices[0]] + A_arr[indices[1]]
    products = A_arr[indices[0]] * A_arr[indices[1]]
    
    # Count unique values
    # np.unique is efficient for sorting and finding unique elements
    unique_sums = len(np.unique(sums))
    unique_products = len(np.unique(products))
    
    return max(unique_sums, unique_products)

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

def _generate_rational_gp(start, num, den, n):
    """
    Generate n distinct integers from a rational geometric progression:
    a_k = start * (num/den)^k
    This helps create multiplicative collisions.
    """
    A = []
    current = float(start)
    ratio = float(num) / float(den)
    
    for _ in range(n):
        val = int(round(current))
        if val < 1:
            val = 1
        if val > MAX_VAL:
            break
        if val not in A:
            A.append(val)
        current *= ratio
        
    # If we didn't get enough, fill with small increments
    if len(A) < n:
        last = A[-1] if A else start
        while len(A) < n:
            last += 1
            if last > MAX_VAL: break
            if last not in A:
                A.append(last)
            
    return A[:n]

def _generate_two_gp_cluster(base1, base2, n1, n2):
    """
    Generate a set that is a union of two geometric progressions:
    Cluster 1: start1 * base1^k for k=0..n1-1
    Cluster 2: start2 * base2^k for k=0..n2-1
    This maximizes multiplicative collisions via unique prime factorization.
    """
    A = set()
    
    # Cluster 1: powers of base1
    val = 1
    for _ in range(n1):
        if val <= MAX_VAL:
            A.add(val)
        else:
            break
        val *= base1
    
    # Cluster 2: offset powers of base2
    start2 = base1  # Start after the first cluster's base
    val = start2
    for _ in range(n2):
        if val <= MAX_VAL:
            A.add(val)
        else:
            break
        val *= base2
    
    # If we don't have enough, add more from either cluster or fill
    if len(A) < n1 + n2:
        # Try adding more from cluster 1
        val = base1 ** n1
        while len(A) < n1 + n2 and val <= MAX_VAL:
            A.add(val)
            val *= base1
        
        # Still not enough, fill with consecutive integers
        if len(A) < n1 + n2:
            max_val = max(A) if A else 1
            while len(A) < n1 + n2:
                max_val += 1
                if max_val > MAX_VAL:
                    break
                A.add(max_val)
    
    return sorted(list(A)[:n1 + n2])

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing to minimize the objective."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 100.0
    cooling_rate = 0.995
    
    while temperature > 0.1:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to modify
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try a few random changes
        accepted = False
        for _ in range(30):
            if time.time() - start_time > time_limit:
                break
            
            # Strategy: Mix of small perturbations and larger jumps
            if rng.random() < 0.5:
                # Small perturbation
                scale = max(1, old_val // 10)
                new_val = old_val + rng.randint(-scale, scale)
            else:
                # Larger jump or swap-like move
                # Try replacing with a value from a different range
                if rng.random() < 0.5:
                    new_val = rng.randint(1, MAX_VAL)
                else:
                    # Try to move towards the median or extremes
                    new_val = old_val + rng.randint(-max(1, old_val//2), max(1, old_val//2))
            
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
                accepted = True
                break
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _two_gp_local_search(A, time_limit, rng):
    """
    Specialized local search that treats A as two geometric clusters.
    Swaps elements between clusters and adjusts exponents to balance sumset/product set.
    """
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    
    # Identify potential cluster structure
    # Sort and try to partition into two groups
    mid = len(A) // 2
    
    # Try different partition points
    for partition in range(5, 15):
        if time.time() - start_time > time_limit:
            break
        
        cluster1 = A[:partition]
        cluster2 = A[partition:]
        
        # Optimize each cluster separately
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Randomly pick which cluster to modify
            if rng.random() < 0.5:
                # Modify cluster 1
                idx = rng.randint(0, len(cluster1) - 1)
                old_val = cluster1[idx]
                
                # Try geometric-like moves
                if rng.random() < 0.5:
                    # Scale by small factor
                    new_val = old_val * rng.choice([2, 3, 4])
                else:
                    # Small perturbation
                    scale = max(1, old_val // 5)
                    new_val = old_val + rng.randint(-scale, scale)
                
                if 1 <= new_val <= MAX_VAL and new_val not in A:
                    cluster1[idx] = new_val
                    A_new = sorted(cluster1 + cluster2)
                    new_obj = _objective(A_new)
                    
                    if new_obj < current_obj:
                        A = A_new
                        current_obj = new_obj
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_A = list(A)
            else:
                # Modify cluster 2
                idx = rng.randint(0, len(cluster2) - 1)
                old_val = cluster2[idx]
                
                if rng.random() < 0.5:
                    # Scale by small factor
                    new_val = old_val * rng.choice([2, 3, 4])
                else:
                    # Small perturbation
                    scale = max(1, old_val // 5)
                    new_val = old_val + rng.randint(-scale, scale)
                
                if 1 <= new_val <= MAX_VAL and new_val not in A:
                    cluster2[idx] = new_val
                    A_new = sorted(cluster1 + cluster2)
                    new_obj = _objective(A_new)
                    
                    if new_obj < current_obj:
                        A = A_new
                        current_obj = new_obj
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_A = list(A)
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5  # Leave margin for safety
    
    rng = random.Random(42)
    best_A = None
    best_obj = float('inf')
    
    starts = []
    
    # Start 1: Arithmetic Progression
    starts.append(list(range(1, 21)))
    
    # Start 2: Rational GP (1, 2/1) -> Powers of 2
    starts.append(_generate_rational_gp(1, 2, 1, 20))
    
    # Start 3: Rational GP (1, 3/2) -> 1, 1.5, 2.25... -> 1, 2, 3, 5, 8... (Fibonacci-like)
    starts.append(_generate_rational_gp(1, 3, 2, 20))
    
    # Start 4: Rational GP (1, 5/4)
    starts.append(_generate_rational_gp(1, 5, 4, 20))
    
    # Start 5: Random
    starts.append(sorted(rng.sample(range(1, MAX_VAL + 1), 20)))
    
    # Start 6: Mixed (Small AP + Large GP)
    small_ap = list(range(1, 11))
    large_gp = _generate_rational_gp(50, 2, 1, 10)
    # Ensure distinctness
    combined = small_ap + [x for x in large_gp if x not in small_ap]
    if len(combined) >= 20:
        starts.append(sorted(combined[:20]))
    else:
        # Fill gaps
        while len(combined) < 20:
            combined.append(combined[-1] + 1)
        starts.append(sorted(combined))
    
    # NEW: Start 7-10: Two geometric cluster configurations
    # These maximize multiplicative collisions via unique prime factorization
    
    # Start 7: Powers of 2 (10 terms) + Powers of 3 (10 terms), offset
    starts.append(_generate_two_gp_cluster(2, 3, 10, 10))
    
    # Start 8: Powers of 2 (12 terms) + Powers of 3 (8 terms)
    starts.append(_generate_two_gp_cluster(2, 3, 12, 8))
    
    # Start 9: Powers of 2 (8 terms) + Powers of 3 (12 terms)
    starts.append(_generate_two_gp_cluster(2, 3, 8, 12))
    
    # Start 10: Powers of 2 (10 terms) + Powers of 5 (10 terms)
    starts.append(_generate_two_gp_cluster(2, 5, 10, 10))
    
    # Start 11: Powers of 3 (10 terms) + Powers of 5 (10 terms)
    starts.append(_generate_two_gp_cluster(3, 5, 10, 10))
    
    # Run local search on each start
    for i, A in enumerate(starts):
        if not _is_valid(A):
            # Fix if invalid
            A = sorted(set(A))
            if len(A) < 20:
                available = [x for x in range(1, MAX_VAL+1) if x not in A]
                A += rng.sample(available, 20 - len(A))
            elif len(A) > 20:
                A = A[:20]
            A = sorted(A)
        
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 1:
            continue
        
        # Allocate time proportional to remaining budget
        # Give first few starts more time
        alloc = max(0.5, remaining_time / (len(starts) - i))
        
        # Use specialized two-GP search for starts 7-11 (indices 6-10)
        if i >= 6:
            A_opt, obj_opt = _two_gp_local_search(A, alloc, rng)
        else:
            A_opt, obj_opt = _local_search(A, alloc, rng)
        
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_A = A_opt
    
    # Final refinement on the best found
    remaining = time_budget - (time.time() - start_time)
    if remaining > 0.5 and best_A:
        # Use two-GP search if the best looks like it has cluster structure
        best_A_sorted = sorted(best_A)
        # Check if there's a natural partition
        mid = 10
        cluster1 = best_A_sorted[:mid]
        cluster2 = best_A_sorted[mid:]
        
        # Use the more specialized search
        A_ref, obj_ref = _two_gp_local_search(best_A, remaining, rng)
        if obj_ref < best_obj:
            best_A = A_ref
            best_obj = obj_ref
    
    if not best_A:
        best_A = list(range(1, 21))
        
    return sorted(best_A)

