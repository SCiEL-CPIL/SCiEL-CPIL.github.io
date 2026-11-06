"""Randomized greedy with bitset conflict check for 4-AP-free subset of [1,512]."""
import time
import random

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    n = 512
    best_size = 0
    best_set = set()
    
    # Precompute for each x, the list of (d, other_indices) triples
    # that would form a 4-AP with x
    # For x to conflict with S, there exists d>0 such that 3 of the 4
    # elements {x-3d, x-2d, x-d, x, x+d, x+2d, x+3d} intersecting with S
    # Actually: 4-AP is (a, a+d, a+2d, a+3d). x is in it if x = a+jd for j in {0,1,2,3}
    # The other 3 elements must all be in S.
    
    # For efficiency, precompute for each x the list of (d, other3) where other3
    # are the other 3 elements of the 4-AP containing x
    conflicts = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for d in range(1, n):
            # x as 1st: x, x+d, x+2d, x+3d
            a1, a2, a3 = x + d, x + 2*d, x + 3*d
            if a3 <= n:
                conflicts[x].append((a1, a2, a3))
            # x as 2nd: x-d, x, x+d, x+2d
            a1, a2, a3 = x - d, x + d, x + 2*d
            if a1 >= 1 and a3 <= n:
                conflicts[x].append((a1, a2, a3))
            # x as 3rd: x-2d, x-d, x, x+d
            a1, a2, a3 = x - 2*d, x - d, x + d
            if a1 >= 1 and a3 <= n:
                conflicts[x].append((a1, a2, a3))
            # x as 4th: x-3d, x-2d, x-d, x
            a1, a2, a3 = x - 3*d, x - 2*d, x - d
            if a1 >= 1:
                conflicts[x].append((a1, a2, a3))
    
    def can_add(x, S):
        """Check if x can be added to S without creating a 4-AP."""
        for a1, a2, a3 in conflicts[x]:
            if a1 in S and a2 in S and a3 in S:
                return False
        return True
    
    def greedy_randomized(order, S):
        """Add elements from order to S if they don't create a 4-AP."""
        for x in order:
            if x not in S and can_add(x, S):
                S.add(x)
        return S
    
    # Try multiple random orderings
    random.seed(42)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Generate a random permutation
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Also try a few more structured approaches if time permits
    if time.time() < deadline - 5:
        # Try descending order
        S = set()
        for x in range(n, 0, -1):
            if can_add(x, S):
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
        
        # Try ascending order (original)
        S = set()
        for x in range(1, n + 1):
            if can_add(x, S):
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Try local search on the best set: remove elements and try to add others
    if time.time() < deadline - 3:
        S = best_set.copy()
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Try removing each element and see if we can add more
            for x in list(S):
                S.remove(x)
                # Try adding elements not in S
                added = False
                for y in range(1, n + 1):
                    if y not in S and can_add(y, S):
                        S.add(y)
                        added = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
                    improved = True
                else:
                    # Restore x
                    S.add(x)
    
    return sorted(best_set) if best_set else [1]

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"First 20: {result[:20]}")
