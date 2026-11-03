import numpy as np

def construct():
    D = 11
    
    # Generate E8 minimal vectors
    # Type 1: +/-1, +/-1, 0...
    e8_type1 = []
    for i in range(8):
        for j in range(i+1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * 8
                    v[i] = si
                    v[j] = sj
                    e8_type1.append(v)
    
    # Type 2: +/-1/2 ... (8 entries)
    # To keep integer arithmetic, we can work with 2*v
    # But the prompt asks for integer vectors.
    # Wait, E8 minimal vectors of type 2 are half-integers.
    # If we use half-integers, they are not integer vectors.
    # The prompt requires "nonzero integer vectors".
    # So we CANNOT use the half-integer vectors of E8.
    
    # We can ONLY use the integer vectors of E8.
    # These are the 112 vectors of type (+/-1, +/-1, 0...).
    
    # So E8 integer minimal vectors = 112.
    
    # Let's re-evaluate D11.
    # D11 minimal vectors are integer: (+/-1, +/-1, 0...).
    # Count: 220.
    
    # Let's re-evaluate E8 + D3.
    # E8 integer part: 112 vectors.
    # D3 integer part: 12 vectors.
    # Total: 112 + 12 = 124.
    # Score: 124 / 22 = 5.63.
    
    # This is worse than D11 (220 -> 10.0).
    
    # So D11 is better.
    
    # Can we find a lattice with more integer minimal vectors?
    # A11: 110.
    # D11: 220.
    # E8 (int): 112.
    
    # What about A10 + A1? 90 + 2 = 92.
    
    # Is there any other lattice?
    # The seed program found ~56.
    # D11 gives 220.
    
    # Let's verify D11 validity again.
    # Vectors: +/-1 at i, +/-1 at j.
    # Norm sqrt(2).
    # Dist between u, v.
    # If u, v share no indices: dist sqrt(4) = 2.
    # If u, v share 1 index: e.g. (1,1,0) and (1,-1,0).
    #   u-v = (0,2,0). Norm 2.
    #   Wait.
    #   u = (1, 1, 0). v = (1, -1, 0).
    #   u-v = (0, 2, 0). Norm 2.
    #   Condition: dist >= R = sqrt(2).
    #   2 >= 1.414. OK.
    #   What if u = (1, 1, 0) and v = (-1, 1, 0)?
    #   u-v = (2, 0, 0). Norm 2. OK.
    #   What if u = (1, 1, 0) and v = (1, 1, 0)? Same.
    #   What if u = (1, 1, 0) and v = (-1, -1, 0)?
    #   u-v = (2, 2, 0). Norm sqrt(8) = 2.82. OK.
    
    # So D11 is valid. Size 220.
    
    # Can we add more?
    # What if we take all D11 vectors AND some others?
    # No, D11 is the full set of minimal vectors.
    
    # Is there a lattice with > 220 minimal integer vectors in 11D?
    # The kissing number of D11 is 220.
    # The kissing number of E8 is 240 (but 128 are half-integers).
    # The kissing number of A11 is 110.
    
    # So 220 is likely the best simple lattice construction.
    
    # Let's check if we can mix D11 with something else.
    # No, D11 is dense.
    
    # So I will implement D11.
    
    D11_vectors = []
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    D11_vectors.append(v)
    
    return np.array(D11_vectors, dtype=np.int64)
