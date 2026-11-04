import math

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # Search over asymmetric digit sets D = {0, 1, M}
    # and bases B large enough to avoid carries.
    
    for M in range(2, 21):
        D = [0, 1, M]
        # Calculate local sum and diff set sizes
        local_sums = set()
        local_diffs = set()
        for a in D:
            for b in D:
                local_sums.add(a + b)
                local_diffs.add(a - b)
        
        size_sums = len(local_sums)
        size_diffs = len(local_diffs)
        
        if size_diffs <= size_sums:
            continue
            
        R = size_diffs / size_sums
        
        max_sum_digit = max(local_sums)
        
        # B must be > max_sum_digit to avoid carries
        for B in range(max_sum_digit + 1, max_sum_digit + 15):
            for n in range(1, 12):
                # Generate U to calculate exact score?
                # Or use approximation?
                # Since B is large, no carries.
                # |U+U| = size_sums^n
                # |U-U| = size_diffs^n
                # max U = M * (B^n - 1) / (B - 1)
                
                # However, we must ensure 0 is in U (it is)
                # And U is finite.
                
                # Let's construct U for n=1 to 3 to be safe, 
                # and for larger n use the formula.
                # Actually, the formula is exact if no carries.
                
                max_u = M * (B**n - 1) // (B - 1)
                if max_u == 0:
                    continue
                    
                num = (size_diffs ** n) / (size_sums ** n)
                den = math.log(2 * max_u + 1)
                if den == 0:
                    continue
                score = 1 + math.log(num) / den
                
                if score > best_score:
                    # We need to return the actual set U
                    # For large n, generating U might be expensive but n<=11 is small
                    U = []
                    for mask in range(3**n):
                        val = 0
                        temp = mask
                        for j in range(n):
                            digit = temp % 3
                            temp //= 3
                            val += D[digit] * (B ** j)
                        U.append(val)
                    best_score = score
                    best = U

    return best
