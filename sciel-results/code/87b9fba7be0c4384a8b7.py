"""
Strategy: Local Search with Random Restarts to minimize size of maximal no-isosceles set.
1. Generate a random maximal set (greedy with shuffled order).
2. Try to remove points: if removing a point leaves the set maximal (i.e., no new point can be added), remove it.
3. Try to swap points: remove one, try to add another different point that keeps the set valid. If the new set is smaller (impossible in swap) or allows further removals, accept. 
   Actually, since we want minimum size, we focus on removals.
   Wait, "maximal" means no point *can* be added. 
   If we remove a point p, the set S' = S \ {p}. Is S' maximal? 
   S' is maximal if no point q in [0,N)^2 \ S' can be added to S' without creating an isosceles triangle.
   If S' is NOT maximal, it means there exists some q not in S' such that S' U {q} is a valid no-isosceles set.
   But we want the set itself to be maximal. So we can only remove p if S \ {p} is still maximal.
   
   Let's check maximality efficiently.
   A set S is maximal if for every p not in S, S U {p} contains an isosceles triangle.
   This is equivalent to: for every p not in S, there exist a, b in S such that (p, a, b) is isosceles.
   
   Algorithm for one step:
   For each p in S:
       S_temp = S - {p}
       Check if S_temp is maximal.
       To check if S_temp is maximal:
           For each candidate q in [0,N)^2 not in S_temp:
               Check if S_temp U {q} is valid (no isosceles).
               If it is valid, then S_temp is NOT maximal (because q could be added).
               So S_temp is maximal iff NO such q exists.
       
       This check is expensive: O(N^4 * N^2) roughly?
       N=16, N^2=256 points.
       S size ~ 30-40.
       Candidates ~ 256 - 30 = 226.
       For each candidate, check against all pairs in S_temp (~30^2/2 = 450 pairs).
       226 * 450 ~ 100k ops per removal check.
       30 points to check ~ 3M ops.
       This is fast enough for many iterations in 300s.
"""

import time
import random
import itertools
import sys

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
POINT_SET = set(POINTS)
DEADLINE = 320.0  # Safety margin

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _is_valid_set(S):
    """Check if set S contains no isosceles triangles."""
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            for k in range(j+1, len(S_list)):
                if _is_isosceles(S_list[i], S_list[j], S_list[k]):
                    return False
    return True

def _is_maximal(S):
    """Check if set S is maximal (no point can be added)."""
    S_set = set(S)
    for p in POINTS:
        if p in S_set:
            continue
        # Check if S U {p} is valid
        # Optimization: only check triangles involving p
        valid = True
        for a in S:
            for b in S:
                if a == b:
                    continue
                if _is_isosceles(p, a, b):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            # If S U {p} is valid, then S is not maximal
            return False
    return True

def _greedy_maximal(order):
    """Construct a maximal set by greedily adding points in given order."""
    chosen = []
    chosen_set = set()
    for p in order:
        if p in chosen_set:
            continue
        # Check if p can be added
        can_add = True
        for a in chosen:
            for b in chosen:
                if a == b:
                    continue
                if _is_isosceles(p, a, b):
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            chosen.append(p)
            chosen_set.add(p)
    return chosen

def construct():
    start_time = time.time()
    
    # 1. Get an initial solution using random greedy
    best_set = None
    best_size = float('inf')
    
    # Try a few random initializations
    for _ in range(20):
        if time.time() > DEADLINE:
            break
        random.shuffle(POINTS)
        S = _greedy_maximal(POINTS)
        if len(S) < best_size:
            best_size = len(S)
            best_set = set(S)
        
        # Local search: try to remove points
        # We iterate until no more removals are possible or time runs out
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            S_list = list(best_set)
            random.shuffle(S_list)
            for p in S_list:
                if time.time() > DEADLINE:
                    break
                S_temp = best_set - {p}
                # Check if S_temp is maximal
                if _is_maximal(S_temp):
                    # It is maximal, so we can remove p
                    best_set = S_temp
                    best_size -= 1
                    improved = True
                    # Restart the loop to check other points again
                    break
        
        # If we found a better solution, continue local search on it
        # (The while loop above already does this for the current best_set)
        
    # Final safety check: ensure we return a valid list
    if best_set is None:
        # Fallback to deterministic greedy if something went wrong
        S = _greedy_maximal(POINTS)
        return S
    return list(best_set)

# Keep the original function name
# The task says "Keep the function names and signatures of the seed program."
# Seed program has: construct()
# It also has _isosceles(a, b, c) but that's internal.
# The public interface is construct().

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Size: {len(res)}")
