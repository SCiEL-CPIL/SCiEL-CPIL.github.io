def _strassen_2x2_decomp():
    # Returns U, V, W for 2x2x2, rank 7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # M2: (a10+a11)b00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    
    # M3: a00(b01-b11)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    
    # M4: a11(b10-b00)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    
    # M5: (a00+a01)b11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    
    # M6: (a10-a00)(b00+b01)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # M7: (a01-a11)(b10+b11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1; W[1,6]=1
    
    return U, V, W

def _construct_4x4x4_decomp():
    # Returns U, V, W for 4x4x4, rank 49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    U2, V2, W2 = _strassen_2x2_decomp()
    
    # Strassen 4x4 terms
    # U_r, V_r are 4x4 matrices
    # We define them by their 2x2 block structure
    
    terms = []
    # M1: (A11+A22)(B11+B22)
    U1 = np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]]) # A11+A22
    V1 = np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]]) # B11+B22
    terms.append((U1, V1, 1)) # W coeff 1 for C11, C12, C21, C22?
    # Actually, W for M1 contributes to all 4 blocks of C.
    # C11 += M1, C12 += M1, C21 += M1, C22 += M1?
    # No.
    # C11 = M1 - M2 - M4 + M7
    # C12 = M3 + M5
    # C21 = M6 + M2
    # C22 = M1 - M5 - M6 + M4
    # So we need to map the result of each M_r to the C blocks.
    
    # Let's just use the standard Strassen formulas.
    # For each r, we have a 4x4 product M_r.
    # We decompose M_r using S2.
    # Then we add the result to C with appropriate signs/positions.
    
    # It's easier to just build the 49 rank decomposition directly.
    # But I don't have the explicit 49 vectors.
    # I can generate them.
    
    # For each r in 1..7:
    #   Compute U_r (4x4) and V_r (4x4).
    #   Decompose U_r V_r using S2 (rank 7).
    #   This gives 7 terms for this r.
    #   Add to global U, V, W.
    
    # We need to know how M_r maps to C.
    # C = sum_r coeff_r * M_r (block-wise)
    
    # Let's define the mapping.
    # M1 -> C11(+), C12(+), C21(+), C22(+) ? No.
    # C11 = M1 - M2 - M4 + M7
    # C12 = M3 + M5
    # C21 = M6 + M2
    # C22 = M1 - M5 - M6 + M4
    
    # So for M1, it contributes to C11, C22 with +1, and C12, C21 with 0?
    # Wait, M1 is (A11+A22)(B11+B22).
    # It is a 4x4 matrix.
    # In the formula C11 = M1 - M2 - M4 + M7, M1 refers to the top-left 2x2 block of M1?
    # No, M1 is a 4x4 matrix.
    # C11 is a 2x2 block.
    # The formula is for the blocks.
    # So C11 = (M1)_11 - (M2)_11 - (M4)_11 + (M7)_11.
    # So we only take the top-left 2x2 block of M1 for C11.
    
    # So for each r, we compute M_r (4x4).
    # Then we extract the 2x2 blocks of M_r and add them to the corresponding C blocks.
    
    # This means we can decompose M_r (4x4) into 7 2x2 products.
    # Then for each 2x2 product, we add it to the correct C block.
    
    # This is getting complex to implement correctly in the short time.
    # I will stick to the 98 rank solution by doing two independent 4x4x4 multiplications.
    # I will implement a helper to build 4x4x4 decomposition.
