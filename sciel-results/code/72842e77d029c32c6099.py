import time
import sys

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # 1. Greedy heuristic to find a good initial solution
    best_range = float('inf')
    best_solution = None
    
    # We can sample L or just iterate all L. 10^6 iterations is fast enough for a simple check.
    # Let's iterate all L to be safe and find the best greedy solution.
    for L in range(1, max_val + 1):
        candidates = []
        for d in divisors:
            # Find closest multiple of d to L
            k = L // d
            m1 = k * d
            m2 = (k + 1) * d
            if m1 < 1:
                m1 = d
            # Pick the one closer to L
            if abs(m1 - L) <= abs(m2 - L):
                candidates.append(m1)
            else:
                candidates.append(m2)
        
        # Resolve duplicates greedily
        # This is not a perfect solver, just a heuristic
        # If duplicates exist, we might just skip this L or try to fix it simply
        # For now, let's just check if they are distinct. If not, try next L?
        # This is weak. Let's just calculate range if distinct, else ignore.
        # Actually, we can just use the first distinct set we find with a small range.
        
        if len(set(candidates)) == n:
            rng = max(candidates) - min(candidates)
            if rng < best_range:
                best_range = rng
                best_solution = candidates[:]
                
    # If greedy didn't find anything distinct (unlikely), fallback to divisors
    if best_solution is None:
        best_solution = divisors[:]
        best_range = max(divisors) - min(divisors)
        
    # 2. Refinement: Search for smaller W
    # We search W from 1 to best_range - 1
    # For each W, search L from 1 to max_val
    
    def has_matching(L, W, divisors):
        # Returns the assignment if a perfect matching exists, else None
        # Candidates are values in [L, L+W]
        # For each divisor i, we need a value v in [L, L+W] such that v % divisors[i] == 0
        
        # Map value to index in candidate list?
        # Better: Build adjacency list for bipartite graph
        # Left: divisors (0..n-1)
        # Right: values in [L, L+W] that are multiples of at least one divisor
        
        # To speed up, we can just iterate values in [L, L+W] and see which divisors they satisfy
        # Then run matching.
        
        # Since W is small, the number of values is small.
        # Let's create a list of (value, divisors_it_satisfies)
        val_to_divs = {}
        for v in range(L, L + W + 1):
            divs = []
            for i, d in enumerate(divisors):
                if v % d == 0:
                    divs.append(i)
            if divs:
                val_to_divs[v] = divs
                
        # Now we have a set of values. We need to assign each divisor to a unique value.
        # This is equivalent to finding a matching that covers all divisors.
        # We can use Kuhn's algorithm.
        
        # Let's order values by how many divisors they satisfy (most constrained first?)
        # Or just iterate values.
        
        # Actually, simpler:
        # For each divisor, the available values are those in val_to_divs that contain the divisor.
        
        # Let's use a recursive backtracking with memoization or simple DFS.
        # Given N=10, it's fast.
        
        used_values = set()
        
        def backtrack(idx):
            if idx == n:
                return True
            d = divisors[idx]
            # Try all values in [L, L+W] that are multiples of d
            # We can precompute multiples for this d in this range
            # But we already have val_to_divs. Let's just iterate all v in range.
            # Optimization: Only iterate v that are multiples of d.
            start_k = (L + d - 1) // d
            for k in range(start_k, (L + W) // d + 1):
                v = k * d
                if v < L or v > L + W:
                    continue
                if v not in used_values:
                    used_values.add(v)
                    if backtrack(idx + 1):
                        return True
                    used_values.remove(v)
            return False
            
        if backtrack(0):
            # Reconstruct solution
            # We need to store the choices. Let's modify backtrack to store path.
            pass
        return None

    # The above has_matching is a bit messy for reconstruction.
    # Let's implement a cleaner matching check that returns the assignment.
    
    def find_assignment(L, W, divisors):
        n = len(divisors)
        # Precompute candidates for each divisor
        candidates = [[] for _ in range(n)]
        for i, d in enumerate(divisors):
            start_k = (L + d - 1) // d
            end_k = (L + W) // d
            for k in range(start_k, end_k + 1):
                candidates[i].append(k * d)
                
        # If any divisor has no candidates, return None
        for i in range(n):
            if not candidates[i]:
                return None
                
        # Backtracking
        assignment = [None] * n
        used = set()
        
        def bt(idx):
            if idx == n:
                return True
            for v in candidates[idx]:
                if v not in used:
                    used.add(v)
                    assignment[idx] = v
                    if bt(idx + 1):
                        return True
                    used.remove(v)
                    assignment[idx] = None
            return False
            
        if bt(0):
            return assignment
        return None

    for W in range(1, best_range):
        # Time check
        if time.time() > 55:
            break
            
        found = False
        for L in range(1, max_val + 1):
            if time.time() > 58:
                return best_solution
            sol = find_assignment(L, W, divisors)
            if sol is not None:
                # Validate just in case
                if len(set(sol)) == n and all(1 <= x <= max_val and x % divisors[i] == 0 for i, x in enumerate(sol)):
                    return sol
        # If loop finishes without finding, continue to next W

    return best_solution
