"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def construct():
    # Try to find a valid set of size > 31 using randomized search
    # A set is valid if no two subsets of different cardinalities have the same sum
    
    def check_valid(arr):
        """Check if arr satisfies the cardinality-separated subset sum condition."""
        n = len(arr)
        # For each cardinality k, compute all subset sums of size k
        # Then check that the sets of sums for different k are disjoint
        sum_sets = [set() for _ in range(n + 1)]
        
        # Use iterative approach to build subset sums by cardinality
        # sums[k] = set of all sums of k-element subsets
        # Start with sums[0] = {0}
        sum_sets[0] = {0}
        
        for i in range(n):
            x = arr[i]
            # Update in reverse to avoid using x multiple times
            for k in range(i, -1, -1):
                if k < n:
                    for s in sum_sets[k]:
                        sum_sets[k + 1].add(s + x)
        
        # Check disjointness: for each pair of different cardinalities,
        # their sum sets must be disjoint
        for k in range(n + 1):
            for m in range(k + 1, n + 1):
                if sum_sets[k] & sum_sets[m]:
                    return False
        return True
    
    # Try to find a set of size 32+ using greedy/randomized approach
    # Start with the known valid set of size 31 and try to add elements
    
    # First, try the base AP: 226..256 (size 31)
    base = list(range(226, 257))  # 226, 227, ..., 256 -> size 31
    
    # Try to find a larger valid set by randomized search
    best_size = 31
    best_set = base
    
    # Try various strategies to find larger sets
    
    # Strategy 1: Try APs with different starts
    for start in range(1, 220):
        for n in range(32, 45):
            end = start + n - 1
            if end > 256:
                break
            arr = list(range(start, end + 1))
            if check_valid(arr):
                if n > best_size:
                    best_size = n
                    best_set = arr
                    return list(best_set)
    
    # Strategy 2: Try to extend the base set with one more element
    # The base set is {226, ..., 256}. Try adding x from 1 to 255 not in base
    for x in range(1, 256):
        if x in base:
            continue
        candidate = base + [x]
        # Sort for consistency
        candidate.sort()
        if check_valid(candidate):
            return candidate
    
    # Strategy 3: Randomized search for size 32
    for _ in range(5000):
        # Generate a random set of size 32
        arr = random.sample(range(1, 257), 32)
        if check_valid(arr):
            return arr
    
    # Strategy 4: Try geometric-like sequences
    for base_val in range(2, 10):
        arr = []
        val = 1
        while val <= 256 and len(arr) < 40:
            arr.append(val)
            val *= base_val
        if len(arr) > best_size:
            if check_valid(arr):
                return arr
    
    # Strategy 5: Try powers of 2
    arr = [1, 2, 4, 8, 16, 32, 64, 128]
    if check_valid(arr):
        return arr
    
    # Fallback: return the known valid set of size 31
    return list(best_set)
# EVOLVE-BLOCK-END
