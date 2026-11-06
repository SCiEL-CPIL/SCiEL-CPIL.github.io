"""Rewritten program for sum-product extremal set construction."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    n = 20
    max_val = 10000
    
    def compute_sets(A):
        """Compute A+A and A*A sets."""
        sums = set()
        prods = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return sums, prods
    
    def objective(A):
        """Compute max(|A+A|, |A*A|)."""
        sums, prods = compute_sets(A)
        return max(len(sums), len(prods))
    
    # Build a diverse candidate pool from geometric sequences
    def build_candidate_pool():
        pool = set()
        # Geometric sequences with various bases
        for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50]:
            x = 1
            while x <= max_val:
                pool.add(x)
                if x > max_val // base:
                    break
                x *= base
        # Add some arithmetic progressions for sum-collisions
        for start in range(1, 50):
            for step in [2, 3, 4, 5, 6, 8, 10, 12]:
                x = start
                while x <= max_val:
                    pool.add(x)
                    x += step
        # Add powers of small numbers
        for base in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            x = base
            while x <= max_val:
                pool.add(x)
                x *= base
        # Add some structured numbers
        for i in range(1, 101):
            for j in range(1, 101):
                v = i * j
                if v <= max_val:
                    pool.add(v)
        return sorted(pool)
    
    candidate_pool = build_candidate_pool()
    
    def greedy_build(pool, rng):
        """Greedy construction with collision-aware scoring."""
        A = []
        current_sums = set()
        current_prods = set()
        
        # Start with a small seed
        candidates = list(pool)
        rng.shuffle(candidates)
        
        # Try to find a good initial pair
        best_pair = None
        best_pair_score = float('inf')
        for i in range(min(50, len(candidates))):
            for j in range(i+1, min(100, len(candidates))):
                a, b = candidates[i], candidates[j]
                s = {a+a, a+b, b+b}
                p = {a*a, a*b, b*b}
                score = max(len(s), len(p))
                if score < best_pair_score:
                    best_pair_score = score
                    best_pair = (a, b)
        
        if best_pair:
            a, b = best_pair
            A.append(a)
            A.append(b)
            current_sums = {a+a, a+b, b+b}
            current_prods = {a*a, a*b, b*b}
        else:
            A.append(candidates[0])
            current_sums = {2*candidates[0]}
            current_prods = {candidates[0]**2}
        
        used = set(A)
        
        while len(A) < n and time.time() < deadline - 5:
            best_elem = None
            best_score = float('inf')
            best_new_sums = None
            best_new_prods = None
            
            for elem in pool:
                if elem in used:
                    continue
                # Compute new sums and products
                new_sums = set()
                new_prods = set()
                for a in A:
                    new_sums.add(a + elem)
                    new_prods.add(a * elem)
                new_sums.add(elem + elem)
                new_prods.add(elem * elem)
                
                # Count how many are truly new
                new_sums_only = new_sums - current_sums
                new_prods_only = new_prods - current_prods
                
                total_sums = len(current_sums) + len(new_sums_only)
                total_prods = len(current_prods) + len(new_prods_only)
                score = max(total_sums, total_prods)
                
                if score < best_score:
                    best_score = score
                    best_elem = elem
                    best_new_sums = new_sums_only
                    best_new_prods = new_prods_only
            
            if best_elem is None:
                break
            
            A.append(best_elem)
            used.add(best_elem)
            current_sums |= best_new_sums
            current_prods |= best_new_prods
        
        return A
    
    def local_search(A, pool, rng, max_time=10):
        """Local search with swap-based refinement."""
        A = list(A)
        current_obj = objective(A)
        used = set(A)
        start_time = time.time()
        
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            # Try swapping each element with a candidate
            for i in range(len(A)):
                if time.time() - start_time >= max_time:
                    break
                old_elem = A[i]
                # Try replacing with elements from pool
                for new_elem in pool:
                    if new_elem in used:
                        continue
                    # Quick check: estimate
                    A[i] = new_elem
                    used.remove(old_elem)
                    used.add(new_elem)
                    
                    new_obj = objective(A)
                    if new_obj < current_obj:
                        current_obj = new_obj
                        improved = True
                        break
                    else:
                        # Revert
                        A[i] = old_elem
                        used.remove(new_elem)
                        used.add(old_elem)
        
        return A, current_obj
    
    best_A = None
    best_obj = float('inf')
    
    # Multiple restarts with different random seeds
    rng = random.Random(42)
    
    # First, try greedy with the full pool
    while time.time() < deadline - 15:
        A = greedy_build(candidate_pool, rng)
        if len(A) < n:
            # Pad with random elements
            while len(A) < n:
                e = rng.choice(candidate_pool)
                if e not in A:
                    A.append(e)
        
        A, obj = local_search(A, candidate_pool, rng, max_time=5)
        
        if obj < best_obj:
            best_obj = obj
            best_A = list(A)
        
        # Generate new candidates by perturbing the best
        if best_A and time.time() < deadline - 20:
            perturbed = list(best_A)
            for _ in range(rng.randint(1, 3)):
                idx = rng.randint(0, len(perturbed) - 1)
                old = perturbed[idx]
                new = rng.choice(candidate_pool)
                if new not in perturbed:
                    perturbed[idx] = new
            perturbed, pobj = local_search(perturbed, candidate_pool, rng, max_time=3)
            if pobj < best_obj:
                best_obj = pobj
                best_A = list(perturbed)
    
    # Final local search on best
    if best_A and time.time() < deadline - 5:
        best_A, best_obj = local_search(best_A, candidate_pool, rng, max_time=10)
    
    if best_A is None or len(best_A) < n:
        # Fallback
        best_A = list(range(1, n + 1))
    
    return best_A
# EVOLVE-BLOCK-END
