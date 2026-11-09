import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 85.0
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    M = reduce(lcm, divisors)
    
    # We will iterate W and L.
    # To speed up, we can break early if time is up.
    
    for W in range(9, 1000):
        if time.time() > deadline:
            break
            
        # Iterate L. 
        # Note: We don't need to check all L if we are smart about it, 
        # but 30k * 1000 W is 30M checks. If each check is ~10us, that's 300s. Too slow.
        # We need to optimize the inner loop or reduce L checks.
        
        # Is there a way to skip L?
        # The condition is: there exist distinct $x_i$ such that $x_i \equiv 0 \pmod{d_i}$ and $L \le x_i \le L+W$.
        # This is equivalent to: The sets $S_i(L) = \{x \in [L, L+W] \mid x \equiv 0 \pmod{d_i}\}$ have a System of Distinct Representatives (SDR).
        
        # For small W, the sets are small.
        # Let's estimate the cost.
        # If W=10, M=30030.
        # 30030 iterations.
        # In each iteration, we build 10 lists.
        # List building: 10 * (W/d_i) ops. Sum W/d_i approx 10 * 10 * (1/6+...+1/39) ~ 10 * 10 * 0.5 = 50 ops.
        # Backtracking: Depth 10. Branching factor small.
        # Total ops per L ~ 100-500.
        # 30030 * 500 = 15M ops.
        # Python can do ~10M-50M simple ops/sec.
        # So W=10 takes ~1-3 seconds.
        # If optimal W is 20, total time ~ 20 * 3 = 60s. Risky.
        
        # Optimization:
        # 1. Use a faster check for SDR existence?
        # 2. Reduce L range?
        #    The pattern of "validity" of L might have period M. We already use that.
        # 3. Can we jump L?
        #    If a window [L, L+W] fails, does [L+1, L+1+W] fail? Not necessarily.
        
        # Let's try to optimize the Python code first.
        
        for L in range(1, M + 1):
            if time.time() > deadline:
                break
                
            candidates = []
            valid = True
            
            # Pre-calculate bounds
            upper = L + W
            
            for d in divisors:
                first_k = (L + d - 1) // d
                last_k = upper // d
                if first_k > last_k:
                    valid = False
                    break
                # Generate candidates
                # Using list comprehension is fast
                c = [k * d for k in range(first_k, last_k + 1)]
                candidates.append(c)
                
            if not valid:
                continue
                
            # Solve CSP
            # We want to assign distinct values from candidates[i] to variable i.
            # Heuristic: Order variables by len(candidates[i]) ascending.
            # But we need to keep track of original indices.
            
            # For small domains, simple backtracking is fine.
            # Let's use a recursive function with memoization? No, state is 'used' set.
            
            # Optimization: If any candidate set has size 1, that value is forced.
            # We can propagate constraints.
            
            # Let's implement a specialized solver for this small case.
            
            # Sort variables by domain size
            # indices = list(range(n))
            # indices.sort(key=lambda i: len(candidates[i]))
            
            # But we need to return in original order.
            
            assignment = [None] * n
            used = set()
            
            # Order of processing
            order = list(range(n))
            order.sort(key=lambda i: len(candidates[i]))
            
            def solve(idx):
                if idx == n:
                    return True
                
                i = order[idx]
                
                # Try each candidate
                for val in candidates[i]:
                    if val in used:
                        continue
                    
                    # Check if assigning val to i causes any other unassigned variable to have no options?
                    # Forward Checking
                    # For simplicity, we just check 'used'. 
                    # Given the small size, simple backtracking might be enough.
                    # But if we assign val, and another variable j only had {val}, it fails.
                    # We can check this cheaply.
                    
                    assign_ok = True
                    # Check forward constraints
                    # Actually, just checking 'used' is necessary but not sufficient for FC.
                    # Let's do simple FC:
                    for j in order[idx+1:]:
                        # If candidates[j] becomes empty after removing val?
                        # We don't remove from candidates[j] in this simple version.
                        # We just check if val is the ONLY option for j?
                        # No, we check if all options for j are in 'used' OR is val?
                        # This is getting complex.
                        
                        # Given the domains are very small (<= 3 or 4 for W=10),
                        # simple backtracking without FC is likely fast enough.
                        # Let's stick to simple backtracking first.
                        pass
                    
                    assignment[i] = val
                    used.add(val)
                    
                    if solve(idx + 1):
                        return True
                        
                    used.remove(val)
                    assignment[i] = None
                    
                return False
            
            if solve(0):
                # Verify
                # return [int(assignment[i]) for i in range(n)]
                return [int(v) for v in assignment]
                
    # Fallback
    return [int(d) for d in divisors]
