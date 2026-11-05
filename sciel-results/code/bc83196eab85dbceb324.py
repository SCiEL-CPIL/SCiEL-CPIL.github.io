import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 30
    half = 15
    
    inv = [1.0 / i for i in range(1, n + 1)]
    
    # Generate A
    sums_A = [0.0]
    codes_A = [0]
    
    for i in range(half):
        inv_i = inv[i]
        new_sums = []
        new_codes = []
        # Using local variables for speed
        app_s = new_sums.append
        app_c = new_codes.append
        for s, c in zip(sums_A, codes_A):
            app_s(s - inv_i)
            app_c(c * 3)
            
            app_s(s)
            app_c(c * 3 + 1)
            
            app_s(s + inv_i)
            app_c(c * 3 + 2)
        
        sums_A = new_sums
        codes_A = new_codes
        
        if time.time() > deadline - 25.0:
            # If we are running out of time, we might skip the rest of generation?
            # But we need both halves.
            # If we can't finish, we return seed.
            # But 25s margin should be enough for 15 steps.
            pass

    # Generate B
    sums_B = [0.0]
    codes_B = [0]
    
    for i in range(half, n):
        inv_i = inv[i]
        new_sums = []
        new_codes = []
        app_s = new_sums.append
        app_c = new_codes.append
        for s, c in zip(sums_B, codes_B):
            app_s(s - inv_i)
            app_c(c * 3)
            
            app_s(s)
            app_c(c * 3 + 1)
            
            app_s(s + inv_i)
            app_c(c * 3 + 2)
        
        sums_B = new_sums
        codes_B = new_codes
        
        if time.time() > deadline - 15.0:
            pass

    if time.time() > deadline - 5.0:
        return [0] * 28 + [1, -1]

    # Sort B
    # To save memory, we can sort a list of indices?
    # No, sorting tuples is fine.
    pairs_B = list(zip(sums_B, codes_B))
    pairs_B.sort(key=lambda x: x[0])
    
    sorted_sums_B = [p[0] for p in pairs_B]
    sorted_codes_B = [p[1] for p in pairs_B]
    
    best_val = float('inf')
    best_code_A = None
    best_code_B = None
    
    len_B = len(sorted_sums_B)
    
    for i in range(len(sums_A)):
        if time.time() > deadline - 1.0:
            break
            
        s_A = sums_A[i]
        target = -s_A
        idx = bisect.bisect_left(sorted_sums_B, target)
        
        # Check neighbors
        # idx can be 0 to len_B
        # We check idx-1, idx, idx+1
        
        # idx-1
        if idx > 0:
            j = idx - 1
            val = s_A + sorted_sums_B[j]
            if val != 0.0:
                abs_val = abs(val)
                if abs_val < best_val:
                    best_val = abs_val
                    best_code_A = codes_A[i]
                    best_code_B = sorted_codes_B[j]
                    
        # idx
        if idx < len_B:
            j = idx
            val = s_A + sorted_sums_B[j]
            if val != 0.0:
                abs_val = abs(val)
                if abs_val < best_val:
                    best_val = abs_val
                    best_code_A = codes_A[i]
                    best_code_B = sorted_codes_B[j]
                    
        # idx+1
        if idx + 1 < len_B:
            j = idx + 1
            val = s_A + sorted_sums_B[j]
            if val != 0.0:
                abs_val = abs(val)
                if abs_val < best_val:
                    best_val = abs_val
                    best_code_A = codes_A[i]
                    best_code_B = sorted_codes_B[j]
                    
    if best_code_A is None:
        return [0] * 28 + [1, -1]
        
    coeffs = [0] * n
    c_A = best_code_A
    for i in range(half - 1, -1, -1):
        digit = c_A % 3
        c_A //= 3
        coeffs[i] = digit - 1
        
    c_B = best_code_B
    for i in range(n - 1, half - 1, -1):
        digit = c_B % 3
        c_B //= 3
        coeffs[i] = digit - 1
        
    return coeffs
