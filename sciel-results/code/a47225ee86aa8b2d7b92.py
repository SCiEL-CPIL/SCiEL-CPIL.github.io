def _try_laderman_46(n=3, m=3, p=6):
    """
    Construct a rank-46 decomposition using Laderman's rank-23 algorithm for 3x3.
    """
    # Laderman's 23 terms for 3x3
    # Each term is (u_vec, v_vec, w_vec)
    # u_vec is 9-dim (flattened A)
    # v_vec is 9-dim (flattened B)
    # w_vec is 9-dim (flattened C)
    
    terms = []
    
    # Helper to create vectors
    def vec(i, j, val=1.0):
        v = np.zeros(9)
        v[i*3+j] = val
        return v
    
    # 1. a00 b00
    terms.append((vec(0,0), vec(0,0), vec(0,0)))
    # 2. a01 b10
    terms.append((vec(0,1), vec(1,0), vec(0,1)))
    # 3. a02 b20
    terms.append((vec(0,2), vec(2,0), vec(0,2)))
    # 4. a10 b01
    terms.append((vec(1,0), vec(0,1), vec(1,0)))
    # 5. a11 b11
    terms.append((vec(1,1), vec(1,1), vec(1,1)))
    # 6. a12 b21
    terms.append((vec(1,2), vec(2,1), vec(1,2)))
    # 7. a20 b02
    terms.append((vec(2,0), vec(0,2), vec(2,0)))
    # 8. a21 b12
    terms.append((vec(2,1), vec(1,2), vec(2,1)))
    # 9. a22 b22
    terms.append((vec(2,2), vec(2,2), vec(2,2)))
    
    # 10. (a00+a11)(b00+b11)
    u = vec(0,0) + vec(1,1)
    v = vec(0,0) + vec(1,1)
    w = vec(0,0) + vec(1,1) # This is not correct, w is for C
    # Actually, the output coefficients depend on the specific algorithm.
    # I will skip this if I'm not sure.
    
    # Since I am not 100% sure of the w vectors, I will not implement this.
    # I will instead improve the numerical search.
    return None
