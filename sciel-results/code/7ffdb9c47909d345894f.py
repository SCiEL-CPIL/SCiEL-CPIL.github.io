import time
import sys

def construct():
    start_time = time.time()
    deadline = 60.0 # Safety margin
    
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // __import__('math').gcd(a, b)
    
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
        
    # a[i-1] = L // i
    a = [L // i for i in range(1, 31)]
    
    # Split indices
    idx1 = range(0, 15) # 0 to 14 corresponds to 1 to 15
    idx2 = range(15, 30) # 15 to 29 corresponds to 16 to 30
    
    # Generate sums for first half
    # We store: sum_value -> encoded_coefficients
    # Encoding: base 3, 0->-1, 1->0, 2->1? Or just -1,0,1 mapped to 0,1,2.
    # Let's use 0 for -1, 1 for 0, 2 for 1.
    
    def gen_sums_dict(indices, a_list):
        # Start with sum 0, coeff 0 (all zeros)
        # We need to map encoded coeff to actual coeffs later
        # dict: sum -> encoded
        d = {0: 0}
        for i in indices:
            ai = a_list[i]
            new_d = {}
            for s, enc in d.items():
                # Option 1: coeff -1 (encoded 0)
                new_s1 = s - ai
                new_enc1 = enc * 3 + 0
                if new_s1 not in new_d:
                    new_d[new_s1] = new_enc1
                # Option 2: coeff 0 (encoded 1)
                new_s2 = s
                new_enc2 = enc * 3 + 1
                if new_s2 not in new_d:
                    new_d[new_s2] = new_enc2
                # Option 3: coeff 1 (encoded 2)
                new_s3 = s + ai
                new_enc3 = enc * 3 + 2
                if new_s3 not in new_d:
                    new_d[new_s3] = new_enc3
            d = new_d
        return d

    d1 = gen_sums_dict(idx1, a)
    
    # Check time
    if time.time() - start_time > deadline - 10:
         # Fallback to seed
         return [0]*28 + [1, -1]

    # Generate sums for second half and check
    # We iterate through all possibilities of second half
    # To avoid storing d2, we just generate on the fly
    # But we need to check against d1
    
    # We can recursively generate sums of second half
    # and check if 1 - s2 is in d1
    
    found = False
    best_coeffs = None
    
    def search_second(i, current_sum, current_enc):
        nonlocal found, best_coeffs
        if found:
            return
        
        if time.time() - start_time > deadline - 5:
            return
            
        if i == len(idx2):
            target = 1 - current_sum
            if target in d1:
                enc1 = d1[target]
                # Decode enc1 and current_enc
                coeffs1 = [0]*15
                e = enc1
                for k in range(14, -1, -1):
                    coeffs1[k] = (e % 3) - 1
                    e //= 3
                coeffs2 = [0]*15
                e = current_enc
                for k in range(14, -1, -1):
                    coeffs2[k] = (e % 3) - 1
                    e //= 3
                best_coeffs = coeffs1 + coeffs2
                found = True
            return
        
        ai = a[idx2[i]]
        # Try -1
        search_second(i+1, current_sum - ai, current_enc * 3 + 0)
        if found: return
        # Try 0
        search_second(i+1, current_sum, current_enc * 3 + 1)
        if found: return
        # Try 1
        search_second(i+1, current_sum + ai, current_enc * 3 + 2)
        
    search_second(0, 0, 0)
    
    if found:
        return best_coeffs
        
    # If 1 not found, try to find minimal non-zero sum
    # This is harder. We can collect all sums from d1 and d2 and find closest to 0 (excluding 0)
    # But d1 has 14M entries. d2 has 14M entries.
    # We can iterate d2, and for each s2, we want to find s1 in d1 such that |s1+s2| is min and non-zero.
    # This requires d1 keys to be sorted.
    
    if time.time() - start_time > deadline - 5:
        return [0]*28 + [1, -1]
        
    keys1 = sorted(d1.keys())
    best_val = float('inf')
    best_enc_pair = None
    
    # Iterate d2
    # To save time, we can just check small integers k = 1, -1, 2, -2 ...
    # But we don't know if they are reachable.
    # Let's try to find if any small k is reachable.
    
    # Actually, if 1 wasn't found, maybe -1?
    # My search above only looked for 1.
    # Let's check -1 as well in the first pass?
    # Or just run search for k in range(-10, 11)?
    
    # Let's do a second pass for k in [-5, 5]
    for k in [-5, 5, -4, 4, -3, 3, -2, 2]:
        if time.time() - start_time > deadline - 2:
            break
        found = False
        def search_second_k(i, current_sum, current_enc, target_k):
            nonlocal found, best_coeffs
            if found:
                return
            if i == len(idx2):
                target = target_k - current_sum
                if target in d1:
                    enc1 = d1[target]
                    coeffs1 = [0]*15
                    e = enc1
                    for kk in range(14, -1, -1):
                        coeffs1[kk] = (e % 3) - 1
                        e //= 3
                    coeffs2 = [0]*15
                    e = current_enc
                    for kk in range(14, -1, -1):
                        coeffs2[kk] = (e % 3) - 1
                        e //= 3
                    best_coeffs = coeffs1 + coeffs2
                    found = True
                return
            ai = a[idx2[i]]
            search_second_k(i+1, current_sum - ai, current_enc * 3 + 0, target_k)
            if found: return
            search_second_k(i+1, current_sum, current_enc * 3 + 1, target_k)
            if found: return
            search_second_k(i+1, current_sum + ai, current_enc * 3 + 2, target_k)
            
        search_second_k(0, 0, 0, k)
        if found:
            return best_coeffs
            
    # Fallback to seed if nothing found
    return [0]*28 + [1, -1]
