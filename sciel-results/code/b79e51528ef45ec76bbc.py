import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    
    # Calculate LCM of divisors
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a * b // gcd(a, b)

    M = 1
    for d in divisors:
        M = lcm(M, d)
        
    # M is 300300
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_val = float('inf')
    best_assign = None
    
    # We search for W from 0 upwards
    # Max W is unlikely to be huge. Let's cap at 1000 for safety, 
    # but usually it's < 100.
    for W in range(0, 1000):
        if time.time() > deadline:
            break
            
        # For each possible start L in [1, M]
        # Note: Due to periodicity, we only need to check L in [1, M]
        # However, the values x_i must be <= 1000000.
        # If L + W > 1000000, we might have issues with upper bound?
        # The problem says "positive integers <= 1000000".
        # If we find a solution in [L, L+W], the values are L+k.
        # We must ensure max(x) <= 1000000.
        # So L <= 1000000 - W.
        # Also L >= 1.
        # And we only need to check L in [1, M] because of periodicity of the *pattern*.
        # But the *values* shift.
        # If a pattern works at L, it works at L + k*M.
        # We just need to find ONE valid L.
        # The smallest valid L in [1, M] gives the smallest values?
        # Not necessarily, but any valid L in [1, M] yields values <= M + W.
        # M = 300300. M + 1000 < 1000000. So the upper bound constraint is always satisfied for L in [1, M] and reasonable W.
        
        for L in range(1, M + 1):
            if time.time() > deadline:
                break
                
            # Check if each divisor has a multiple in [L, L+W]
            # and collect candidates
            candidates = []
            valid = True
            for d in divisors:
                # First multiple >= L
                k = (L + d - 1) // d
                x = k * d
                # List of multiples in [L, L+W]
                cands = []
                if x <= L + W:
                    # Add all multiples
                    while x <= L + W:
                        cands.append(x)
                        x += d
                if not cands:
                    valid = False
                    break
                candidates.append(cands)
                
            if not valid:
                continue
                
            # Try to find distinct representatives
            # Use backtracking with MRV
            # Sort by number of candidates
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))
            
            used = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = indices[idx]
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                return False
            
            # Optimization: Check time inside backtracking? No, too slow.
            # Backtracking is fast enough for N=10.
            
            if backtrack(0):
                # Construct result
                res = [0] * n
                # We need to map back from sorted indices
                # Wait, the backtrack function uses 'indices' order.
                # We need to store the assignment.
                
                # Let's rewrite backtrack to store assignment
                assignment = [0] * n
                def backtrack_store(idx):
                    if idx == n:
                        return True
                    i = indices[idx]
                    for val in candidates[i]:
                        if val not in used:
                            used.add(val)
                            assignment[i] = val
                            if backtrack_store(idx + 1):
                                return True
                            used.remove(val)
                    return False
                
                if backtrack_store(0):
                    current_val = max(assignment) - min(assignment)
                    # This W is the minimal possible for this L?
                    # Actually, we are iterating W globally.
                    # If we found a solution for W, it's the global min W?
                    # Yes, because we iterate W from 0 up.
                    # And for a fixed W, we just need ANY L.
                    # So the first solution found is optimal.
                    
                    # Ensure types are int
                    return [int(x) for x in assignment]
                    
        # If no solution found for this W, try next W

    # If loop finishes without finding (should not happen for W < 1000),
    # return best found or a fallback
    if best_assign is not None:
        return [int(x) for x in best_assign]
    
    # Fallback: return divisors themselves (if distinct and valid, which they are)
    # But they might not be minimal.
    # Just return the divisors as a valid, if not optimal, solution.
    return [int(d) for d in divisors]
