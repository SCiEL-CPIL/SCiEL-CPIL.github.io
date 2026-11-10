import time
import random
import sys

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

def _generate_smooth_numbers(limit):
    """Generate 5-smooth numbers up to limit."""
    smooth = set()
    p2 = [1]
    p3 = [1]
    p5 = [1]
    # Simple generation
    for i in range(1, limit + 1):
        x = i
        while x % 2 == 0: x //= 2
        while x % 3 == 0: x //= 3
        while x % 5 == 0: x //= 5
        if x == 1:
            smooth.add(i)
    return sorted(smooth)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin: evaluator kills at 90s
    
    n = 20
    maximum = 10000
    
    # Generate pool of candidate numbers: 5-smooth numbers
    smooth_nums = _generate_smooth_numbers(maximum)
    # Also include some other small numbers to allow diversity
    # But smooth numbers are best for multiplicative structure
    # Let's stick to 5-smooth for the main pool
    pool = smooth_nums
    
    # Initial candidate: First 20 smooth numbers
    # This tends to have small products and sums
    best_A = pool[:n]
    best_score = _score_fast(best_A)
    
    # Local Search
    # We try to swap elements in A with elements from the pool
    # to reduce the score
    
    # To speed up, we can restrict the pool to numbers that are "close" in magnitude
    # or just use the whole pool if it's not too large.
    # There are ~2000 5-smooth numbers <= 10000.
    # 20 * 2000 swaps * 210 checks = ~8.4M ops per iteration.
    # This might be too slow for many iterations in 85s.
    # We need a faster heuristic or smaller pool.
    
    # Let's use a smaller pool: top 500 smooth numbers
    # And maybe add some random small integers
    small_ints = list(range(1, 100))
    pool = list(set(pool[:500] + small_ints))
    pool.sort()
    
    # If the pool is still large, we might need to be selective.
    # Let's try a greedy improvement first.
    
    A = list(best_A)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all positions
        for i in range(n):
            old_val = A[i]
            A_set = set(A)
            A_set.discard(old_val)
            # Try swapping with candidates from pool
            # Sort candidates by some heuristic? Or just iterate.
            # To save time, we can check only a subset of the pool per iteration
            # or break early if no improvement found in a random sample.
            
            # Let's iterate through the pool
            for new_val in pool:
                if new_val == old_val or new_val in A_set:
                    continue
                
                A[i] = new_val
                sc = _score_with_bound(A, best_score)
                if sc < best_score:
                    best_score = sc
                    improved = True
                    # Sort A to keep it consistent for future checks?
                    # _score_fast doesn't require sorted, but it's good for debugging.
                    A.sort()
                    break
                A[i] = old_val
            
            if improved:
                # Restart the loop to try other positions with the new A
                # Actually, we can just continue to next i, but the set changed.
                # It's safer to break and restart the outer while loop
                # to re-evaluate all positions with the new best_score bound.
                break
        
        # If no improvement in a full pass, try random swaps to escape local minima
        if not improved:
            # Try random perturbations
            found_improvement = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                i = random.randint(0, n-1)
                old_val = A[i]
                # Pick a random candidate from pool
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
                # If stuck, try a more aggressive random restart?
                # Or just break if we've been stuck for a while.
                # For now, break to avoid wasting time.
                # We could also try to swap two elements at once (2-swap)
                # But 1-swap is usually enough for this size.
                break

    # Final validation
    if len(A) != n or len(set(A)) != n or any(x <= 0 or x > maximum for x in A):
        A = list(range(1, n+1))
        
    return A

# EVOLVE-BLOCK-END
