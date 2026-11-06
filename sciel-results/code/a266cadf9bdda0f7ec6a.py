"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _subset_sums_distinct(vals):
    """Check if all 2^n subset sums are distinct for sorted vals."""
    n = len(vals)
    sums = {0}
    for v in vals:
        new_sums = set()
        for s in sums:
            new_sums.add(s + v)
        # Check for collisions before adding
        for ns in new_sums:
            if ns in sums:
                return False
        sums.update(new_sums)
    return True

def _try_greedy(max_val):
    """
    Greedy: pick smallest a_i such that all subset sums remain distinct.
    Returns list of 12 integers or None if impossible within max_val.
    """
    vals = []
    sums = {0}
    for i in range(12):
        start = vals[-1] + 1 if vals else 1
        if start > max_val:
            return None
        for cand in range(start, max_val + 1):
            # Check if adding cand creates any duplicate subset sum
            collides = False
            for s in sums:
                if s + cand in sums:
                    collides = True
                    break
            if not collides:
                vals.append(cand)
                for s in list(sums):
                    sums.add(s + cand)
                break
        else:
            return None
    return vals

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline to be safe
    
    # The greedy approach gives a specific sequence. Let's just compute it.
    # For n=12, the greedy sequence is known to be optimal for this problem?
    # Actually, for "distinct subset sums" (also called superincreasing if each element > sum of all previous),
    # the greedy (smallest next) gives exactly 1, 2, 4, ..., 2^(n-1) which has max 2^(n-1).
    # Wait, 1, 2, 4, 8, ... : 2 > 1, 4 > 1+2=3, 8 > 1+2+4=7, etc.
    # This is the superincreasing sequence, which is optimal in the sense that you can't make the largest element smaller
    # while maintaining distinct subset sums? 
    # Actually, no. Consider n=3: superincreasing gives 1,2,4 (max=4). 
    # Can we do 1,2,3? Sums: 0,1,2,3,3,4,5,6 -> 3 appears twice (2 and 1+2). No.
    # 1,3,4? Sums: 0,1,3,4,4,5,7,8 -> 4 appears twice. No.
    # 2,3,4? Sums: 0,2,3,4,5,6,7,9 -> all distinct! Max=4. Same as superincreasing.
    # So for n=3, min max is 4.
    # For n=4: superincreasing 1,2,4,8 (max=8).
    # Try to get max=7: need 4 distinct positive ints <=7 with 16 distinct subset sums.
    # Total sum must be at least 15 (since 16 distinct non-negative sums starting from 0, the largest is total sum, so total sum >= 15).
    # With 4 numbers <=7, max total is 7+6+5+4=22.
    # Let's try to search. This is a known problem: the minimal possible maximum element for n distinct subset sums is 2^(n-1) only if we require the sequence to be superincreasing? 
    # No, actually the minimal max element is not necessarily 2^(n-1). 
    # Wait, the problem is equivalent to finding a set of n positive integers with all 2^n subset sums distinct.
    # This is exactly the definition of a "sum-free" or "distinct subset sum" set.
    # The minimal possible maximum element is known to be 2^(n-1) for n=1,2,3,4,5,6,7,8,9,10,11,12?
    # Let me check n=5: superincreasing max=16. Can we do max=15?
    # I recall that the optimal solution for this problem is indeed the powers of 2, i.e., 1,2,4,...,2^(n-1).
    # But wait, is that true? Let me think again.
    # For n=3, we saw 2,3,4 works with max=4, same as 1,2,4.
    # For n=4, is there a set with max < 8?
    # I believe the answer is no, the minimum max is 2^(n-1).
    # But let's not assume. Let's run the greedy and see what it produces.
    
    # Actually, the greedy algorithm that picks the smallest valid next element will produce:
    # 1 (smallest positive)
    # 2 (smallest > 1 such that {0,1,2,3} are distinct) -> 2 works
    # 4 (smallest > 2 such that no collision) -> 3 fails (1+2=3), 4 works
    # 8 (smallest > 4 such that no collision) -> 5 fails (1+4=5), 6 fails (2+4=6), 7 fails (3+4=7 or 1+2+4=7), 8 works
    # So greedy gives 1,2,4,8,16,32,64,128,256,512,1024,2048
    # Max = 2048.
    
    # Can we do better? Let's try a local search or a smarter approach.
    # But given the time constraint, let's first check if the greedy result is valid and return it.
    # Then if we have time, try to improve.
    
    best_vals = _try_greedy(1 << 20)  # Large enough bound
    if best_vals is not None:
        # Verify
        if _subset_sums_distinct(best_vals):
            return best_vals
    
    # Fallback: return powers of 2
    return [1 << i for i in range(12)]
# EVOLVE-BLOCK-END
