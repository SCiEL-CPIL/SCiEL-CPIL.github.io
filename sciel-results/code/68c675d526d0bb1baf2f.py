import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _score_fast(A):
    """Return max(|A+A|, |A*A|) for sorted list A of length 20."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            ss.add(ai + A[j])
            ps.add(ai * A[j])
    return max(len(ss), len(ps))

def _score_components(A):
    """Return (|A+A|, |A*A|) for sorted list A of length 20."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            ss.add(ai + A[j])
            ps.add(ai * A[j])
    return len(ss), len(ps)

def _score_with_bound(A, current_best):
    """Return max(|A+A|, |A*A|) but stop early if either set exceeds current_best."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s = ai + A[j]
            ss.add(s)
            if len(ss) >= current_best:
                return current_best
            p = ai * A[j]
            ps.add(p)
            if len(ps) >= current_best:
                return current_best
    return max(len(ss), len(ps))

def _score_with_bound_components(A, current_best):
    """Return (|A+A|, |A*A|) but stop early if either set exceeds current_best."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s = ai + A[j]
            ss.add(s)
            if len(ss) >= current_best:
                return current_best, current_best
            p = ai * A[j]
            ps.add(p)
            if len(ps) >= current_best:
                return current_best, current_best
    return len(ss), len(ps)

def _generate_smooth_numbers(limit):
    """Generate 5-smooth numbers up to limit."""
    smooth = set()
    for i in range(1, limit + 1):
        x = i
        while x % 2 == 0: x //= 2
        while x % 3 == 0: x //= 3
        while x % 5 == 0: x //= 5
        if x == 1:
            smooth.add(i)
    return sorted(smooth)

def _try_geometric_base(b, n=20, maximum=10000):
    """Try a geometric progression with base b, then refine with multiplicative perturbations."""
    # Generate geometric sequence: b^0, b^1, b^2, ...
    # But we need exactly n distinct positive integers <= maximum
    # For base b, the sequence is b^0=1, b^1=b, b^2=b^2, ...
    # We need b^(n-1) <= maximum
    
    if b <= 1:
        return None
    
    # Check if b^(n-1) <= maximum
    val = 1
    for _ in range(n-1):
        val *= b
        if val > maximum:
            return None
    
    # Generate the geometric sequence
    A = [1]
    for i in range(1, n):
        A.append(A[-1] * b)
    
    if max(A) > maximum or len(set(A)) != n:
        return None
    
    score = _score_fast(A)
    
    # Now try to refine by replacing some elements with nearby smooth numbers
    # to break the exact geometric structure and reduce sum/product set sizes
    smooth_nums = _generate_smooth_numbers(maximum)
    smooth_set = set(smooth_nums)
    
    # For each position, try swapping with nearby smooth numbers
    improved = True
    while improved:
        improved = False
        for i in range(n):
            old_val = A[i]
            A_set = set(A)
            A_set.discard(old_val)
            
            # Try nearby smooth numbers (within a factor of 3)
            candidates = [s for s in smooth_nums if old_val // 4 <= s <= old_val * 4]
            
            for new_val in candidates:
                if new_val == old_val or new_val in A_set:
                    continue
                
                A[i] = new_val
                sc = _score_with_bound(A, score)
                if sc < score:
                    score = sc
                    improved = True
                    A.sort()
                    break
                A[i] = old_val
            
            if improved:
                break
    
    return A, score

def _local_search(A, best_score, pool, deadline):
    """Local search to improve A with multiple restarts and adaptive pool."""
    n = len(A)
    A = list(A)
    A.sort()
    improved = True
    stall_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        # Build a focused pool around current elements
        local_pool = set()
        for v in A:
            for p in pool:
                if 0.5 * v <= p <= 2.0 * v:
                    local_pool.add(p)
            # Also add nearby integers
            for delta in range(-3, 4):
                candidate = v + delta
                if 1 <= candidate <= 10000:
                    local_pool.add(candidate)
        
        for i in range(n):
            old_val = A[i]
            A_set = set(A)
            A_set.discard(old_val)
            
            # Try candidates from local pool first, then broader pool
            candidates = sorted(local_pool)
            for new_val in candidates:
                if new_val == old_val or new_val in A_set:
                    continue
                
                A[i] = new_val
                sc = _score_with_bound(A, best_score)
                if sc < best_score:
                    best_score = sc
                    improved = True
                    A.sort()
                    break
                A[i] = old_val
            
            if improved:
                break
            # If no improvement from local pool, try broader pool
            for new_val in pool:
                if new_val == old_val or new_val in A_set:
                    continue
                
                A[i] = new_val
                sc = _score_with_bound(A, best_score)
                if sc < best_score:
                    best_score = sc
                    improved = True
                    A.sort()
                    break
                A[i] = old_val
            
            if improved:
                break
        
        if not improved:
            stall_count += 1
            # Try random perturbations to escape local minimum
            found_improvement = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                i = random.randint(0, n-1)
                old_val = A[i]
                new_val = random.choice(pool)
                if new_val == old_val or new_val in A:
                    continue
                
                A[i] = new_val
                sc = _score_with_bound(A, best_score)
                if sc < best_score:
                    best_score = sc
                    found_improvement = True
                    A.sort()
                    break
                A[i] = old_val
            
            if not found_improvement:
                if stall_count >= 3:
                    break
                # Try a more aggressive random restart
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    i = random.randint(0, n-1)
                    old_val = A[i]
                    # Pick a value far from current
                    new_val = random.choice(pool)
                    if new_val == old_val or new_val in A:
                        continue
                    
                    A[i] = new_val
                    sc = _score_with_bound(A, best_score)
                    if sc < best_score:
                        best_score = sc
                        found_improvement = True
                        A.sort()
                        break
                    A[i] = old_val
                
                if not found_improvement:
                    break
            else:
                stall_count = 0
    
    return A, best_score

def _simulated_annealing(A, best_score, pool, deadline):
    """Simulated annealing to escape local minima."""
    n = len(A)
    A = list(A)
    A.sort()
    current_score = _score_fast(A)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        i = random.randint(0, n-1)
        old_val = A[i]
        new_val = random.choice(pool)
        if new_val == old_val or new_val in A:
            continue
        
        A[i] = new_val
        new_score = _score_fast(A)
        
        if new_score < current_score:
            current_score = new_score
            A.sort()
            if current_score < best_score:
                best_score = current_score
        else:
            # Accept with probability exp(-(new_score - current_score)/T)
            delta = new_score - current_score
            if random.random() < math.exp(-delta / T):
                current_score = new_score
                A.sort()
            else:
                A[i] = old_val
        
        T *= cooling
    
    return A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 20
    maximum = 10000
    
    # Generate smooth numbers
    smooth_nums = _generate_smooth_numbers(maximum)
    
    # Strategy 1: Start with first 20 smooth numbers
    best_A = smooth_nums[:n]
    best_score = _score_fast(best_A)
    
    # Create a comprehensive pool
    pool = list(smooth_nums)
    # Add some non-smooth numbers for diversity
    for i in range(1, maximum + 1, 7):
        if i not in set(smooth_nums):
            pool.append(i)
    pool.sort()
    
    # Local search from initial set
    best_A, best_score = _local_search(best_A, best_score, pool, deadline)
    
    # Simulated annealing to escape local minima
    if time.time() < deadline - 10:
        best_A, best_score = _simulated_annealing(best_A, best_score, pool, deadline - 5)
    
    # Try different initial sets
    for trial in range(8):
        if time.time() > deadline - 15:
            break
        
        # Generate diverse initial sets
        if trial % 3 == 0:
            # Random smooth numbers
            random.shuffle(smooth_nums)
            candidate = smooth_nums[:n]
        elif trial % 3 == 1:
            # Mix of small and large smooth numbers
            small = smooth_nums[:10]
            large = smooth_nums[-10:]
            candidate = small[:5] + large[:5] + smooth_nums[10:20]
        else:
            # Arithmetic-like progression with perturbations
            candidate = []
            for i in range(n):
                base_val = 1 + i * 500
                candidate.append(base_val + random.randint(-10, 10))
            candidate = sorted(set(candidate))
            while len(candidate) < n:
                candidate.append(random.choice(smooth_nums))
            candidate = sorted(candidate[:n])
        
        candidate.sort()
        
        sc = _score_with_bound(candidate, best_score)
        if sc < best_score:
            best_score = sc
            best_A = candidate
        
        # Local search from this candidate
        max_val = max(candidate)
        local_pool = [s for s in pool if s <= max_val * 2]
        local_pool.sort()
        best_A, best_score = _local_search(best_A, best_score, local_pool, deadline)
        
        # Simulated annealing if time permits
        if time.time() < deadline - 10:
            best_A, best_score = _simulated_annealing(best_A, best_score, local_pool, deadline - 5)
    
    # Final polish: intensive local search
    if time.time() < deadline - 3:
        best_A, best_score = _local_search(best_A, best_score, pool, deadline - 1)
    
    # Final validation
    if len(best_A) != n or len(set(best_A)) != n or any(x <= 0 or x > maximum for x in best_A):
        best_A = list(range(1, n+1))
    
    return best_A

# EVOLVE-BLOCK-END
