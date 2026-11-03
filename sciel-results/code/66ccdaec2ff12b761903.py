import numpy as np

def construct():
    D = 11
    vectors = []
    
    # --- Generate 2*E8 minimal vectors in first 8 dims ---
    # 2*E8 is the set of integer vectors z in Z^8 such that:
    # 1. All z_i are even and sum(z) is divisible by 4
    # 2. All z_i are odd and sum(z) is divisible by 4
    
    # We look for vectors with norm squared <= 8.
    
    # Case 1: All even components.
    # z_i in {-2, 0, 2}.
    # Norm squared is 4 * (number of non-zero components).
    # To have norm sq <= 8, we can have at most 2 non-zero components.
    # Non-zero components must be +/- 2.
    # Sum must be divisible by 4.
    
    # Subcase 1a: 1 non-zero component.
    # z = (+/- 2, 0, ...). Sum = +/- 2. Not div by 4.
    
    # Subcase 1b: 2 non-zero components.
    # z = (+/- 2, +/- 2, 0, ...).
    # Sum = 2*s1 + 2*s2 = 2(s1+s2).
    # s1, s2 in {1, -1}.
    # s1+s2 is 2, 0, or -2.
    # Sum is 4, 0, or -4. All div by 4.
    # So all such vectors are valid.
    
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
                    
    # Case 2: All odd components.
    # z_i in {-1, 1}.
    # Norm squared is 8 * (number of components) = 8 * 8 = 64? No.
    # z_i are +/- 1. Norm squared is sum(z_i^2) = 8.
    # This is <= 8.
    # Sum must be divisible by 4.
    # Sum = sum(z_i).
    # Let p be count of 1s, m be count of -1s. p+m=8.
    # Sum = p - m = p - (8-p) = 2p - 8.
    # 2p - 8 div 4 => 2p div 4 => p is even.
    
    for mask in range(2**8):
        # Generate vector of +/- 1 based on mask
        # Bit 0 -> index 0, etc.
        # 1 -> +1, 0 -> -1 ?
        # Let's just check parity of number of -1s.
        # If number of -1s is even, number of +1s is even (since 8 is even).
        
        # Count 1s (or -1s)
        # Using bit count
        # Let 1 bit mean -1.
        minus_count = bin(mask).count('1')
        
        if minus_count % 2 == 0:
            v = [0] * D
            for k in range(8):
                if (mask >> k) & 1:
                    v[k] = -1
                else:
                    v[k] = 1
            vectors.append(v)
            
    # --- Generate 2*D3 minimal vectors in last 3 dims ---
    # 2*D3 is subset of 2*Z^3 where sum is div by 4.
    # D3 = {x in Z^3 | sum(x) even}.
    # 2*D3 = {2x | x in D3}.
    # z = 2x. z_i even.
    # sum(z) = 2 sum(x).
    # sum(x) even => sum(z) div 4.
    
    # Norm squared <= 8.
    # z_i in {-2, 0, 2}.
    # Max 2 non-zeros.
    # 1 non-zero: sum +/- 2 (not div 4).
    # 2 non-zeros: sum 2(s1+s2). Div 4 always.
    
    for i in range(3):
        for j in range(i + 1, 3):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    # Map local indices 0,1,2 to global 8,9,10
                    v[8 + i] = 2 * si
                    v[8 + j] = 2 * sj
                    vectors.append(v)
                
    return np.array(vectors, dtype=np.int64)
