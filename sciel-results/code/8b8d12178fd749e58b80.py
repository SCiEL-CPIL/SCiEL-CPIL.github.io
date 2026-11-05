<<<<<<< SEARCH
def construct():
    """
    Construct a tensor decomposition for 6x6 matrix multiplication.
    Uses Kronecker product of Strassen 2x2 (rank 7) and schoolbook 3x3 (rank 27).
    Total rank = 7 * 27 = 189.
    """
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Fallback: schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Primary: Kronecker product of Strassen 2x2 and schoolbook 3x3
    sU, sV, sW = _strassen_2x2()
    kU, kV, kW = _schoolbook_3x3()
    U_k, V_k, W_k = _kronecker_product(sU, sV, sW, kU, kV, kW, 2, 2, 2, 3, 3, 3)
    
    # Verify quickly
    if _verify(U_k, V_k, W_k, 6, 6, 6):
        return {"U": U_k.tolist(), "V": V_k.tolist(), "W": W_k.tolist()}
    
    # Fallback: block Strassen
    U_b, V_b, W_b = _block_strassen_6x6x6()
    if _verify(U_b, V_b, W_b, 6, 6, 6):
        return {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_b.tolist()}
    
    # If verification fails, fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
=======
def construct():
    """
    Construct a tensor decomposition for 6x6 matrix multiplication.
    Tries multiple strategies in order of decreasing rank:
    1. Kronecker product of schoolbook 2x2 (rank 4) and Strassen-like 3x3 (rank 19) = rank 76
    2. Kronecker product of Strassen 2x2 (rank 7) and schoolbook 3x3 (rank 27) = rank 189
    3. Block Strassen 6x6 = rank 189
    4. Schoolbook = rank 216
    """
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Fallback: schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Strategy 1: Try rank-19 3x3x3 decomposition, Kronecker with schoolbook 2x2
    s3U, s3V, s3W = _strassen_3x3()
    if s3U is not None:
        sb2U, sb2V, sb2W = _schoolbook(2, 2, 2)
        U_k1, V_k1, W_k1 = _kronecker_product(sb2U, sb2V, sb2W, s3U, s3V, s3W, 2, 2, 2, 3, 3, 3)
        if _verify(U_k1, V_k1, W_k1, 6, 6, 6):
            return {"U": U_k1.tolist(), "V": V_k1.tolist(), "W": W_k1.tolist()}
    
    # Strategy 2: Kronecker product of Strassen 2x2 and schoolbook 3x3
    sU, sV, sW = _strassen_2x2()
    kU, kV, kW = _schoolbook_3x3()
    U_k, V_k, W_k = _kronecker_product(sU, sV, sW, kU, kV, kW, 2, 2, 2, 3, 3, 3)
    
    if _verify(U_k, V_k, W_k, 6, 6, 6):
        return {"U": U_k.tolist(), "V": V_k.tolist(), "W": W_k.tolist()}
    
    # Strategy 3: Block Strassen
    U_b, V_b, W_b = _block_strassen_6x6x6()
    if _verify(U_b, V_b, W_b, 6, 6, 6):
        return {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_b.tolist()}
    
    # If verification fails, fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
>>>>>>> REPLACE
