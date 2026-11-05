import time
from math import lcm
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    N = 22
    # Calculate LCM of 1..22
    L = 1
    for i in range(1, N + 1):
        L = lcm(L, i)
        
    # Weights w_i = L // i
    weights = [0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = L // i
        
    # We want to find the largest k such that there exists a subset of indices
    # of size k where the corresponding weights have distinct subset sums.
    
    # We will search for k from N down to 1.
    # To make the search faster, we can use backtracking.
    
    # Helper to check if we can add weight w to current sums
    def can_add(sums, w):
        # sums is a set of ints
        # new_sums = sums U (sums + w)
        # We can generate the new part and check for intersection
        # Or just create the union and check size
        # Creating union is O(2^c)
        # Intersection check:
        # if (sums + w) intersects sums, then collision.
        # (sums + w) is {s+w for s in sums}
        # We can check if any s+w is in sums.
        # This is O(2^c) lookups.
        # Creating the set is also O(2^c).
        # Let's just create the new set for simplicity and speed in Python (C implementation of set union)
        # But we need to return the new set if valid.
        new_sums = set()
        for s in sums:
            new_sums.add(s)
            new_sums.add(s + w)
        return new_sums if len(new_sums) == 2 * len(sums) else None

    # Backtracking function
    # idx: current index being considered (from N down to 1)
    # count: number of elements chosen so far
    # sums: set of current subset sums
    # target_k: desired size
    # path: list of chosen indices
    
    # To avoid deep recursion overhead, we can use a stack or just recursion.
    # Depth is 22, so recursion is fine.
    
    def dfs(idx, count, sums, target_k, path):
        if time.time() - start_time > deadline:
            return None
            
        if count == target_k:
            return list(path)
            
        if idx == 0:
            return None
            
        # Pruning: not enough elements left
        if target_k - count > idx:
            return None
            
        # Try to include idx
        # Order of elements: we process N, N-1, ..., 1
        # So 'idx' is the value of the element?
        # Let's map index in DFS to value.
        # We iterate values from N down to 1.
        
        w = weights[idx]
        new_sums = can_add(sums, w)
        if new_sums is not None:
            path.append(idx)
            res = dfs(idx - 1, count + 1, new_sums, target_k, path)
            if res is not None:
                return res
            path.pop()
            
        # Try to exclude idx
        res = dfs(idx - 1, count, sums, target_k, path)
        if res is not None:
            return res
            
        return None

    # Search for largest k
    for k in range(N, 0, -1):
        if time.time() - start_time > deadline:
            break
        # Heuristic: We can try to find a solution for k.
        # If we find one, return it.
        # To speed up finding a solution for a specific k,
        # we might want to bias the search.
        # But for now, simple DFS should work for N=22.
        
        # One optimization: 
        # If we are looking for k, and we've already found a set of size k in a previous 
        # iteration? No, we go down from N.
        
        # Let's add a simple heuristic: 
        # If we fail to find a set of size k, we try k-1.
        
        # To make the DFS faster, we can shuffle the order of "include" vs "exclude"?
        # Or sort indices? The indices are fixed 1..22.
        
        # Let's try to run the DFS.
        # If it takes too long, we might need a better heuristic.
        # But N=22 is small.
        
        res = dfs(N, 0, {0}, k, [])
        if res is not None:
            return res
            
    # Fallback: return a known small set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]

# EVOLVE-BLOCK-END
