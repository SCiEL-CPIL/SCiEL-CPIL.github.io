import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
    # L = 30030

    start_time = time.time()
    deadline = start_time + 85.0

    for W in range(9, 200):
        if time.time() > deadline:
            break
            
        # Precompute valid A for this W
        # valid_A[A] is True if every divisor has at least one multiple in [A, A+W]
        valid_A = [True] * L
        
        for d in divisors:
            # For this divisor, mark A as invalid if no multiple in [A, A+W]
            # A multiple exists if floor((A+W)/d) > floor((A-1)/d)
            # Or simply: ((A + d - 1) // d) * d <= A + W
            # We can iterate A and check.
            # To be faster, we can note that the pattern repeats every d.
            # For A in [0, d):
            #   first_mult = ((A + d - 1) // d) * d
            #   if first_mult > A + W: invalid
            # The set of invalid A in [0, d) is a contiguous block?
            # Let's check d=6, W=2.
            # A=0: first=0, 0<=2 OK
            # A=1: first=6, 6<=3 NO
            # A=2: first=6, 6<=4 NO
            # A=3: first=6, 6<=5 NO
            # A=4: first=6, 6<=6 OK
            # A=5: first=6, 6<=7 OK
            # Invalid: 1, 2, 3. Contiguous.
            # In general, for a fixed d and W, the invalid A's in [0, d) form a contiguous range?
            # Condition: ((A + d - 1) // d) * d > A + W
            # Let k = (A + d - 1) // d. Then k*d > A + W.
            # k is roughly A/d.
            # It turns out the invalid region is contiguous.
            
            # Let's just loop A in range(L) for simplicity, but break early if possible?
            # 30030 * 10 divisors = 300k operations per W.
            # If W goes up to 50, that's 15M operations. This is fast in Python.
            
            for A in range(L):
                if not valid_A[A]:
                    continue
                first = ((A + d - 1) // d) * d
                if first > A + W:
                    valid_A[A] = False
        
        # Now iterate only over valid A
        for A in range(L):
            if not valid_A[A]:
                continue
            if time.time() > deadline:
                break
            
            # Generate candidates
            candidates = []
            for d in divisors:
                first = ((A + d - 1) // d) * d
                cands = []
                m = first
                while m <= A + W:
                    cands.append(m)
                    m += d
                candidates.append(cands)
            
            # Backtracking
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            # Quick check: union of candidates must have size >= n
            all_vals = set()
            for c in candidates:
                all_vals.update(c)
            if len(all_vals) < n:
                continue
                
            used = set()
            assignment = [None] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = order[idx]
                for v in candidates[i]:
                    if v not in used:
                        used.add(v)
                        assignment[i] = v
                        if backtrack(idx + 1):
                            return True
                        used.discard(v)
                        assignment[i] = None
                return False
            
            if backtrack(0):
                return [int(v) for v in assignment]

    return [int(d) for d in divisors]
