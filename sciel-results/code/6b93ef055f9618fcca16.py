"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    # The universe of candidates
    N = 22
    candidates = list(range(1, N + 1))
    
    # We want to find a subset S of [1..22] such that all subset sums of reciprocals are distinct.
    # This is equivalent to saying that the set of reciprocals {1/x for x in S} is "dissociated".
    # If we let x_i be binary variables indicating if i is in S, we want to maximize sum(x_i).
    
    # The condition "all subset sums distinct" means that for any two distinct subsets A, B of S, 
    # sum_{i in A} 1/i != sum_{j in B} 1/j.
    # This is a very strong constraint. 
    # Note: If the set is dissociated, then 2^k distinct sums are generated.
    # Since the sums are rational, we can work with integers to avoid floating point issues, 
    # but the number of subsets grows as 2^k. For k=10, 2^10 = 1024 subsets. 
    # For k=12, 2^12 = 4096 subsets.
    # The pairwise inequality constraints would be O(4^k) if we naively enumerate all pairs of subsets.
    # However, there is a simpler necessary and sufficient condition for a set of positive numbers to be dissociated?
    # Not exactly simple, but for reciprocals of integers, there are known bounds.
    # Actually, the problem is equivalent to finding the largest subset of {1..22} that is "sum-free" in the sense of subset sums? No, it's distinct subset sums.
    
    # Let's use a different perspective:
    # If we select a set S, we require that the map from subsets of S to R (sum of reciprocals) is injective.
    # This implies that no element can be written as a sum of reciprocals of other elements in S?
    # No, that's for "complete sequences" or similar.
    # The condition is simply: for any two disjoint subsets A, B of S, sum_{a in A} 1/a != sum_{b in B} 1/b.
    # This is equivalent to saying that the only solution to sum_{i in S} c_i / i = 0 with c_i in {-1, 0, 1} is c_i = 0 for all i.
    # This is a linear independence problem over Q.
    # The vectors v_i = (1/i) are just scalars. Wait, they are scalars.
    # So we are asking for linear independence of the set of numbers {1/x : x in S} over the integers with coefficients in {-1, 0, 1}?
    # No, the coefficients are determined by the subset.
    # Let S = {x_1, ..., x_k}.
    # We require that for any choice of signs s_i in {-1, 0, 1}, not all zero, sum s_i / x_i != 0.
    # This is equivalent to saying that the numbers 1/x_i are linearly independent over Q?
    # No, linear independence over Q means sum q_i / x_i = 0 implies q_i = 0 for rational q_i.
    # Here the coefficients are restricted to {-1, 0, 1}.
    # However, if the numbers are linearly independent over Q, then certainly the subset sums are distinct.
    # Are 1/x_i linearly independent over Q?
    # Yes, if x_i are distinct integers, the fractions 1/x_i are linearly independent over Q.
    # Proof: Suppose sum c_i / x_i = 0 with c_i integers. Then sum c_i * (L / x_i) = 0 where L is LCM of x_i.
    # This doesn't immediately prove independence for arbitrary integers c_i.
    # Actually, 1, 1/2, 1/3, ... are linearly independent over Q.
    # Proof: Let 1/x_1, ..., 1/x_k be distinct unit fractions.
    # Suppose sum_{i=1}^k c_i / x_i = 0.
    # Multiply by L = lcm(x_1, ..., x_k).
    # sum c_i (L/x_i) = 0.
    # This is a linear combination of integers. It does NOT imply c_i = 0.
    # Example: 1/2 + 1/3 = 1/6 + 1/2? No.
    # Example: 1/2 = 1/3 + 1/6. So {1/2, 1/3, 1/6} are NOT linearly independent.
    # And indeed, subset sums: {1/2} and {1/3, 1/6} have the same sum.
    # So the condition is NOT just linear independence. It is specifically that no non-trivial relation with coefficients in {-1, 0, 1} exists.
    # This is equivalent to the set being "dissociated".
    
    # How to find the largest such set in [1, 22]?
    # We can use a backtracking search or ILP.
    # Given the small N=22, we can try to build the set greedily or via search.
    # A known result: The largest dissociated subset of {1, ..., N} is related to the "binary" or "superincreasing" structures.
    # If we pick powers of 2: 1, 2, 4, 8, 16. Reciprocals: 1, 1/2, 1/4, 1/8, 1/16.
    # Subset sums of powers of 2 reciprocals are distinct (binary representation).
    # Size 5.
    # Can we do better?
    # The set {1, 2, 3, 5, 7, 11, 13, 17, 19} has size 9.
    # Let's verify if it's valid.
    # It's a known heuristic that primes work well?
    
    # Let's implement a fast verification and a greedy/search approach to maximize size.
    # We will iterate through possible sizes k from N down to 1.
    # For each k, we try to find a subset of size k that is dissociated.
    # To check if a set S is dissociated, we can check if any subset sum collision exists.
    # Number of subsets is 2^k. For k=10, 1024 sums. We can compute all sums and check for duplicates.
    # Sums are rational. We can represent them as fractions or high-precision floats.
    # Fractions are safer.
    
    from fractions import Fraction
    
    def is_dissociated(S):
        if not S:
            return True
        # Generate all subset sums
        sums = set()
        sums.add(Fraction(0, 1))
        # Efficiently generate subset sums
        # For small k, simple iteration is fine.
        # k <= 15 is manageable (32768 sums).
        k = len(S)
        if k > 15:
            # Heuristic check or more efficient algorithm needed?
            # For now, if k is large, we might rely on known properties or random sampling.
            # But let's try to stick to exact check for k <= 15.
            pass
        
        # Generate all 2^k sums
        # Use bitwise or recursion
        sums = {Fraction(0,1)}
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + Fraction(1, x))
            sums.update(new_sums)
            
        return len(sums) == (1 << k)
        
    # Search strategy:
    # Start with large k, try to find a valid set.
    # To speed up, we can use a greedy construction or local search.
    # However, the space of subsets of size k from 22 is C(22, k).
    # C(22, 10) is 646646. That's small enough to brute force if the check is fast.
    # C(22, 11) is 705432.
    # C(22, 12) is 646646.
    # C(22, 13) is 497420.
    
    # So we can brute force all subsets of size k from 22 down to 1 until we find one that is dissociated.
    # We need to optimize the `is_dissociated` check.
    # Using Fractions is slow. Using floats might have precision issues.
    # We can use integers: Multiply all sums by L = lcm(S).
    # Then we are checking if all subset sums of {L/x for x in S} are distinct.
    # L can be very large (lcm of 1..22 is ~232792560).
    # L/x fits in int64? 232792560 / 1 = 232M. Sum of 10 such terms is < 2.3B. Fits in int64.
    # Sum of 15 such terms is < 3.5B. Fits in int64.
    # So we can use integer sums!
    
    def is_dissociated_int(S):
        if not S:
            return True
        from math import lcm
        L = 1
        for x in S:
            L = lcm(L, x)
        # Map x -> L/x
        vals = [L // x for x in S]
        # Check if all subset sums of vals are distinct
        k = len(S)
        if k > 20:
            # Too many subsets, use heuristic?
            # But max k is 22. 2^22 is 4M.
            # 4M sums might be too slow in Python if done naively.
            # But we only need to check for collisions.
            # We can use a set of sums.
            pass
        
        sums = set()
        sums.add(0)
        for v in vals:
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            sums.update(new_sums)
            if len(sums) < (1 << (len(vals) - 1 + 1)): # Early exit if collision detected?
                 # Actually, we just check at the end or periodically.
                 pass
        
        return len(sums) == (1 << k)
        
    # Let's refine the search.
    # We want the MAXIMUM cardinality.
    # Start from k=22 down to 1.
    # For each k, iterate through combinations.
    # To make it faster, we can prune.
    # If a subset of size k is not dissociated, its supersets are not necessarily not dissociated?
    # Actually, if A is not dissociated, any superset B containing A is not dissociated.
    # So if we find a collision in a subset, we can prune all supersets containing that subset.
    # This suggests a backtracking search with pruning.
    
    def backtrack(current_set, available, k_target, time_limit):
        if time.time() - start_time > time_limit:
            return None
        if len(current_set) == k_target:
            if is_dissociated_int(current_set):
                return current_set
            return None
        
        # Pruning: If the remaining available elements + current_set < k_target, fail
        if len(current_set) + len(available) < k_target:
            return None
            
        # Heuristic: Try to include larger numbers first? Or smaller?
        # Larger numbers have smaller reciprocals, less likely to cause collisions?
        # Actually, small numbers have large reciprocals, more likely to cause "large" sums that don't overlap?
        # Let's try sorting available in descending order.
        
        for i, x in enumerate(available):
            new_set = current_set + [x]
            # Check if new_set is potentially dissociated?
            # We can check if new_set is dissociated. If not, prune.
            # This is expensive but effective for pruning.
            if is_dissociated_int(new_set):
                result = backtrack(new_set, available[i+1:], k_target, time_limit)
                if result is not None:
                    return result
            else:
                # If new_set is not dissociated, any superset of new_set is not dissociated.
                # So we can skip x entirely for this branch?
                # Yes.
                continue
                
        return None

    # Try sizes from 22 down to 1
    for k in range(N, 0, -1):
        # Initialize available as sorted descending
        available = sorted(candidates, reverse=True)
        result = backtrack([], available, k, timeout)
        if result is not None:
            return result
            
    # Fallback
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]

# EVOLVE-BLOCK-END
