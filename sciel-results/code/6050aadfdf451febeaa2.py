import time
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    # Use a deadline slightly before the hard limit to allow for safe return
    deadline = time.time() + 85.0
    
    # Calculate LCM of all divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    lcm_all = reduce(lcm, divisors)
    
    # Kuhn's algorithm for bipartite matching
    # We want to match each divisor index i to a unique value from candidates[i]
    def find_assignment(candidates):
        # candidates: list of lists of integers
        # Sort indices by number of candidates (most constrained first) to speed up search
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        
        # match_val_to_idx[value] = divisor_index
        match_val_to_idx = {}
        
        # visited set for the current DFS path
        # Since n is small, we can use a set of values
        def try_kuhn(u, visited):
            # u is the divisor index
            for v in candidates[u]:
                if v in visited:
                    continue
                visited.add(v)
                # If v is not matched, or the divisor currently matched to v can be rematched
                if v not in match_val_to_idx or try_kuhn(match_val_to_idx[v], visited):
                    match_val_to_idx[v] = u
                    return True
            return False
        
        # Try to find a matching for each divisor in order of constraint
        for u in order:
            visited = set()
            if not try_kuhn(u, visited):
                return None
        
        # Reconstruct assignment
        assignment = [0] * n
        for v, u in match_val_to_idx.items():
            assignment[u] = v
            
        return assignment

    # Iterate span W from minimum possible (n-1) upwards
    # The maximum span is bounded, but 1000 is a safe upper bound for this small set
    # We stop if we find a solution
    for W in range(n - 1, 1000):
        if time.time() > deadline:
            break
            
        # Iterate window start L
        # We only need to check L up to lcm_all because the pattern repeats
        for L in range(1, lcm_all + 1):
            if time.time() > deadline:
                break
            
            # Check if window [L, L+W] is within bounds
            # The problem states integers must be <= 1000000
            if L + W > MAX_V:
                continue

            candidates = []
            valid_window = True
            
            for d in divisors:
                # Find multiples of d in [L, L+W]
                # First multiple >= L: ceil(L/d) * d
                # Last multiple <= L+W: floor((L+W)/d) * d
                first_k = (L + d - 1) // d
                last_k = (L + W) // d
                
                if first_k > last_k:
                    valid_window = False
                    break
                
                # Generate the list of candidates
                # Using list comprehension for speed
                cands = [k * d for k in range(first_k, last_k + 1)]
                candidates.append(cands)
            
            if not valid_window:
                continue
            
            # Try to assign distinct values
            result = find_assignment(candidates)
            
            if result is not None:
                # Verify the solution just in case
                if len(set(result)) != n:
                    continue
                for i, d in enumerate(divisors):
                    if result[i] % d != 0:
                        continue
                if any(v < 1 or v > MAX_V for v in result):
                    continue
                
                # Return the result as a list of ints
                return [int(v) for v in result]
        
        # If we finish all L for a given W without success, increment W
    
    # Fallback: Return a valid solution if not found
    # Using the divisors themselves (they are distinct and valid)
    return [int(d) for d in divisors]
