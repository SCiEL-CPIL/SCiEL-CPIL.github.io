<<<<<<< SEARCH
import random
import time
import itertools

def is_sidon_4(subset):
    """Check if a 4-element subset is Sidon."""
    # Sort for consistency? Not needed for sum uniqueness check.
    # Sums:
    # s0+s0, s0+s1, s0+s2, s0+s3
    # s1+s1, s1+s2, s1+s3
    # s2+s2, s2+s3
    # s3+s3
    s0, s1, s2, s3 = subset
    sums = [
        2*s0, s0+s1, s0+s2, s0+s3,
        2*s1, s1+s2, s1+s3,
        2*s2, s2+s3,
        2*s3
    ]
    return len(set(sums)) == 10

def count_sidon_4_subsets(s):
    """Count number of Sidon 4-subsets in set s."""
    count = 0
    # Optimization: Precompute sums? 
    # For n=12, 495 subsets.
    # Just iterate.
    for combo in itertools.combinations(s, 4):
        if is_sidon_4(combo):
            count += 1
    return count
=======
import random
import time
import itertools
import numpy as np

def count_sidon_4_subsets_fast(s):
    """Count number of Sidon 4-subsets in set s using vectorized approach."""
    arr = np.array(s, dtype=np.int64)
    n = len(arr)
    count = 0
    
    # For each 4-combination, check if all 10 pair sums (including diagonals) are unique
    for i in range(n-3):
        for j in range(i+1, n-2):
            for k in range(j+1, n-1):
                for l in range(k+1, n):
                    # Compute all 10 sums
                    a, b, c, d = arr[i], arr[j], arr[k], arr[l]
                    sums = np.array([2*a, a+b, a+c, a+d, 2*b, b+c, b+d, 2*c, c+d, 2*d], dtype=np.int64)
                    if len(np.unique(sums)) == 10:
                        count += 1
    return count

def count_sidon_4_subsets(s):
    """Count number of Sidon 4-subsets in set s."""
    return count_sidon_4_subsets_fast(s)

def build_conflict_graph(s):
    """
    Build a conflict graph where nodes are elements of s.
    Two elements conflict if they share a pair sum with any other pair in s.
    For Sidon property, we need all unordered pair sums (including i<=j) to be unique.
    A pair (i,j) conflicts with pair (k,l) if i+j == k+l and {i,j} != {k,l}.
    """
    arr = list(s)
    n = len(arr)
    # conflict[i][j] = True means element i and j cannot both be in a Sidon subset
    # Actually, we need: if including both i and j creates a sum collision with some other pair
    # Simpler: for each pair (i,j), compute sum. If any other pair (k,l) has same sum, 
    # then including both (i,j) and (k,l) violates Sidon.
    # For maximum clique: nodes are elements, edge between i,j means they CAN coexist in Sidon set.
    # i,j can coexist if 2*i != 2*j (always true for distinct), and i+j doesn't equal any other pair sum
    # that involves only elements NOT in {i,j}... this is complex.
    
    # Alternative: greedily build a set maximizing sum collisions
    # For each candidate element, count how many sum-collisions it creates with existing set
    return None
>>>>>>> REPLACE
