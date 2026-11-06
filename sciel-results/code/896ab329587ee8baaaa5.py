import time
import sys
from math import gcd
from functools import reduce

# Set recursion limit
sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin before 90s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    """
    Returns a nonempty subset of [1, 22] with all reciprocal subset sums distinct.
    Maximizes cardinality.
    
    Strategy:
    1. Convert reciprocal sums to integer sums by multiplying by L = LCM(1..22).
    2. Use backtracking search with exact integer arithmetic.
    3. Heuristic: Prioritize elements with larger weights (smaller indices) to spread sums apart.
    """
    
    # Precompute LCM of 1..22
    L = reduce(_lcm, range(1, 23), 1)
    
    # Weights w[i] = L // i
    # We want to select a subset of indices {1..22}
    # Such that the set of weights {w[i] for i in subset} is dissociated
    # i.e., all subset sums of the weights are distinct.
    
    weights = {i: L // i for i in range(1, 23)}
    
    # Sort indices by descending weight (ascending index)
    # This helps in "spreading out" the sums early, reducing collision probability
    sorted_indices = list(range(1, 23))
    
    best_set = [1]
    
    # Helper to check if a list of indices forms a dissociated set
    def is_dissociated(indices):
        sums = {0}
        for idx in indices:
            w = weights[idx]
            new_sums = set()
            for s in sums:
                ns = s + w
                new_sums.add(ns)
            # If any new sum is in existing sums, collision
            if new_sums & sums:
                return False
            sums = sums.union(new_sums)
        return True

    # Backtracking search for target size K
    def find_set_of_size_k(k):
        # We use a recursive function that builds the set
        # current_set: list of selected indices
        # current_sums: set of all subset sums of the weights in current_set
        # start_pos: index in sorted_indices to start looking from
        
        def backtrack(start_pos, current_set, current_sums):
            if _time_left() < 0.5:
                return None
            
            if len(current_set) == k:
                return list(current_set)
            
            remaining_needed = k - len(current_set)
            available = len(sorted_indices) - start_pos
            if available < remaining_needed:
                return None
            
            for i in range(start_pos, len(sorted_indices)):
                idx = sorted_indices[i]
                w = weights[idx]
                
                # Check if adding w causes a collision
                # New sums are {s + w for s in current_sums}
                # Collision if (s1 + w) == s2 for some s1, s2 in current_sums
                # This is equivalent to checking if {s + w} intersects current_sums
                
                # Optimization: 
                # If we sort current_sums, we can do a faster check? 
                # But set intersection is O(min(N, M)). 
                # Let's just generate the new sums and check intersection.
                
                # To avoid generating the full set if a collision is found early,
                # we can iterate and check.
                # However, we need the full new set for the recursive call if valid.
                
                # Let's check for collision first
                collision = False
                for s in current_sums:
                    if (s + w) in current_sums:
                        collision = True
                        break
                
                if collision:
                    continue
                
                # If no collision, compute new_sums for recursion
                # Note: new_sums will be disjoint from current_sums
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + w)
                
                # Recurse
                # We pass i+1 to ensure we pick distinct indices in increasing order of sorted_indices
                # This avoids permutations of the same set
                res = backtrack(i + 1, current_set + [idx], new_sums)
                if res is not None:
                    return res
                    
            return None

        # Try to find a set of size k
        # Initial sums: {0}
        # Initial set: []
        return backtrack(0, [], {0})

    # Try sizes from 12 down to 1
    # Max possible is likely 11 or 12.
    # 2^12 = 4096 sums.
    # 2^13 = 8192 sums.
    
    for k in range(12, 0, -1):
        if _time_left() < 2.0:
            break
            
        res = find_set_of_size_k(k)
        if res is not None:
            # Verify result with exact integers (redundant but safe)
            if is_dissociated(res):
                return res
            else:
                # Should not happen
                best_set = res
                return res
        
    # Fallback: Greedy
    # If backtracking fails or times out, use a simple greedy approach
    # This might not be optimal but ensures a valid result
    greedy_set = []
    current_sums = {0}
    for idx in sorted_indices:
        w = weights[idx]
        collision = False
        for s in current_sums:
            if (s + w) in current_sums:
                collision = True
                break
        if not collision:
            greedy_set.append(idx)
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + w)
            current_sums = new_sums
            
    if greedy_set:
        return greedy_set
        
    return [1]

if __name__ == "__main__":
    s = construct()
    print(f"Found set: {s}")
    print(f"Size: {len(s)}")
    # Verify
    L = reduce(_lcm, range(1, 23), 1)
    sums = {0}
    valid = True
    for x in s:
        w = L // x
        new_sums = set()
        for ss in sums:
            ns = ss + w
            new_sums.add(ns)
        if new_sums & sums:
            valid = False
            break
        sums = sums.union(new_sums)
    print(f"Valid: {valid}")
    print(f"Num sums: {len(sums)} (Expected {1 << len(s)})")
