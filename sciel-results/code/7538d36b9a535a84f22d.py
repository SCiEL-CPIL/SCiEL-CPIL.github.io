def construct():
    n = 256
    S = set()
    
    def can_add(num, S):
        # Check if adding num to S violates any constraint
        S_list = list(S)
        
        # Case 1: num is 'a', need b,c in S with num < b <= c and num | (b+c)
        for k in range(3, 6):
            target = k * num
            if target > 512:
                break
            b_min = max(num + 1, target - 256)
            b_max = min(target // 2, 256)
            for b in range(b_min, b_max + 1):
                c = target - b
                if b <= c and b in S and c in S:
                    return False
        
        # Case 2: num is 'b', for each a in S with a < num, need c in S with num <= c and a | (num + c)
        for a in S:
            if a >= num:
                continue
            # Need c in S with c >= num and a | (num + c)
            # num + c = k*a for some k
            # c = k*a - num >= num => k*a >= 2*num => k >= 2*num/a
            # c <= 256 => k*a - num <= 256 => k <= (256+num)/a
            k_min = max(3, (2 * num + a - 1) // a)  # k*a >= 2*num + a (since c>=num, num+c>=2*num, and need k*a >= num+c >= 2*num, but also c>=num means k*a-num>=num so k*a>=2*num)
            # Actually: c = k*a - num >= num => k*a >= 2*num
            # And c >= num (since b=num <= c)
            # k*a >= 2*num
            # c = k*a - num <= 256 => k*a <= 256 + num
            k_lo = max(3, (2 * num + a - 1) // a)
            k_hi = (256 + num) // a
            for k in range(k_lo, k_hi + 1):
                c = k * a - num
                if num <= c <= 256 and c in S:
                    return False
        
        # Case 3: num is 'c', for each a < b in S with a < b <= num, need a | (b + num)
        for a in S:
            if a >= num:
                continue
            for b in S:
                if a < b <= num:
                    if (b + num) % a == 0:
                        return False
        
        return True
    
    for num in range(256, 0, -1):
        if can_add(num, S):
            S.add(num)
    
    return sorted(S)
