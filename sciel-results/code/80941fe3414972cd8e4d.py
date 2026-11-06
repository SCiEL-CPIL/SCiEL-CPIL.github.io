import time

def construct():
    """
    Construct distinct three-term unit-fraction representations of 4/n for n=3..256.
    Returns 254 rows; each is None or [x, y, z] with x<y<z, all distinct, <= 10^12.
    """
    deadline = time.time() + 85
    MAX_DEN = 10**12
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # Also 1/x > 4/n - 4/n - 4/n (trivial)
        # Actually, since x < y < z, 1/x is the largest term.
        # 1/x > 4/n / 3 => x < 3n/4
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            B = n * x
            
            # We look for y = k such that z = B*k / (A*k - B) is integer
            # Conditions:
            # 1. A*k - B > 0  => k > B/A
            # 2. z > y => B*k / (A*k - B) > k => B > A*k - B => 2B > A*k => k < 2B/A
            # 3. A*k - B divides B*k
            
            k_min = B // A + 1
            k_max = (2 * B) // A  # floor(2B/A) is the max integer k < 2B/A
            
            # Optimization: If k_min > k_max, no solution for this x
            if k_min > k_max:
                continue
                
            for k in range(k_min, k_max + 1):
                if k == x:
                    continue
                    
                D = A * k - B
                # D > 0 is guaranteed by k > B/A
                
                # Check divisibility
                if (B * k) % D != 0:
                    continue
                    
                z = (B * k) // D
                
                # Check z > y (k)
                if z <= k:
                    continue
                    
                # Check z != x
                if z == x:
                    continue
                    
                # Check bound
                if z > MAX_DEN:
                    continue
                    
                found = [x, k, z]
                break
                
            if found:
                break
                
        answers.append(found)
        
    return answers
