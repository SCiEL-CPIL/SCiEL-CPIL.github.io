def _decomp_221():
    """2x2x1 matmul. Rank 4.
    U: 4x4, V: 2x4, W: 2x4
    U[i*2+j, r], V[j*1+k, r], W[k*2+i, r]
    n=2, m=2, p=1
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    col = 0
    for i in range(2):
        for j in range(2):
            U[i*2+j, col] = 1
            V[j, col] = 1 # j*1 + 0
            W[0*2+i, col] = 1 # k=0
            col += 1
    return U, V, W

def _decomp_213():
    """2x1x3 matmul. Rank 6.
    U: 2x6, V: 3x6, W: 6x6? No.
    n=2, m=1, p=3
    U: n*m = 2. V: m*p = 3. W: p*n = 6.
    U[i*1+j, r], V[j*3+k, r], W[k*2+i, r]
    """
    U = np.zeros((2, 6))
    V = np.zeros((3, 6))
    W = np.zeros((6, 6))
    col = 0
    for i in range(2):
        for j in range(1):
            for k in range(3):
                U[i, col] = 1
                V[j*3+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    return U, V, W

def _decomp_223():
    """2x2x3 matmul. Rank 11.
    Split B (2x3) into B_L (2x2) and B_R (2x1).
    A is 2x2.
    C = A B_L + A B_R.
    A B_L: 2x2x2, Rank 7.
    A B_R: 2x2x1, Rank 4.
    Total 11.
    """
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    sU, sV, sW = _strassen_222()
    dU, dV, dW = _decomp_221()
    
    col = 0
    # Part 1: A * B_L (2x2x2)
    for r in range(7):
        U[:, col] = sU[:, r]
        V[0:4, col] = sV[:, r] # B_L is rows 0,1 of B
        W[:, col] = sW[:, r]   # k in 0,1
        col += 1
        
    # Part 2: A * B_R (2x2x1)
    for r in range(4):
        U[:, col] = dU[:, r]
        V[4:6, col] = dV[:, r] # B_R is row 2? No, B is 2x3.
        # B indices: j in {0,1}, k in {0,1,2}.
        # B_L: j in {0,1}, k in {0,1}.
        # B_R: j in {0,1}, k in {2}.
        # V index j*p+k. p=3.
        # B_L: j*3 + 0, j*3 + 1.
        # B_R: j*3 + 2.
        # dV is for 2x1. V[j*1+k] where k=0. So V[j, 0].
        # We need to map dV[j, r] to V[j*3+2, col].
        
        # Let's re-verify _decomp_221 V shape.
        # n=2, m=2, p=1. V shape (m*p, rank) = (2, 4).
        # V[j*1+0, r].
        # So dV[0, r] corresponds to j=0. dV[1, r] to j=1.
        # We need to put these in V[0*3+2, col] and V[1*3+2, col].
        
        V[0*3+2, col] = dV[0, r]
        V[1*3+2, col] = dV[1, r]
        
        # W for 2x2x1. W[k*2+i]. k=0.
        # dW shape (2, 4). dW[0*2+i, r].
        # We need to put in W[0*2+i, col] for k=2?
        # No, W index is k*n+i.
        # For 2x2x1, k is always 0.
        # In the full 2x2x3, k for this part is 2.
        # So we need W[2*2+i, col] = dW[0*2+i, r].
        
        W[2*2+0, col] = dW[0, r]
        W[2*2+1, col] = dW[1, r]
        
        col += 1
    return U, V, W

def _decomp_233():
    """2x3x3 matmul. Rank 17.
    A (2x3), B (3x3).
    Split A into A_L (2x2), A_R (2x1).
    Split B into B_T (2x3), B_B (1x3).
    C = A_L B_T + A_R B_B.
    A_L B_T: 2x2x3. Rank 11.
    A_R B_B: 2x1x3. Rank 6.
    Total 17.
    """
    U = np.zeros((6, 17))
    V = np.zeros((9, 17))
    W = np.zeros((6, 17))
    
    u223, v223, w223 = _decomp_223()
    u213, v213, w213 = _decomp_213()
    
    col = 0
    # Part 1: A_L * B_T (2x2x3)
    # A_L is cols 0,1 of A. A indices i*3+j.
    # A_L: i in {0,1}, j in {0,1}.
    # u223 U is 2x2. U[i*2+j, r].
    # We need to map to U[i*3+j, col].
    
    # B_T is rows 0,1 of B. B indices j*3+k.
    # B_T: j in {0,1}, k in {0,1,2}.
    # v223 V is 2x3. V[j*3+k, r].
    # This matches directly with V[j*3+k, col] for j in 0,1.
    
    # W is 3x2. W[k*2+i, r].
    # Matches directly.
    
    for r in range(11):
        # U mapping
        U[0*3+0, col] = u223[0*2+0, r]
        U[0*3+1, col] = u223[0*2+1, r]
        U[1*3+0, col] = u223[1*2+0, r]
        U[1*3+1, col] = u223[1*2+1, r]
        
        # V mapping
        V[0*3+0, col] = v223[0*3+0, r]
        V[0*3+1, col] = v223[0*3+1, r]
        V[0*3+2, col] = v223[0*3+2, r]
        V[1*3+0, col] = v223[1*3+0, r]
        V[1*3+1, col] = v223[1*3+1, r]
        V[1*3+2, col] = v223[1*3+2, r]
        
        # W mapping
        W[:, col] = w223[:, r]
        
        col += 1
        
    # Part 2: A_R * B_B (2x1x3)
    # A_R is col 2 of A. A indices i*3+2.
    # u213 U is 2x1. U[i*1+0, r] = U[i, r].
    # Map to U[i*3+2, col].
    
    # B_B is row 2 of B. B indices 2*3+k.
    # v213 V is 1x3. V[0*3+k, r] = V[k, r].
    # Map to V[2*3+k, col].
    
    # W is 3x2. W[k*2+i, r].
    # Matches directly.
    
    for r in range(6):
        U[0*3+2, col] = u213[0, r]
        U[1*3+2, col] = u213[1, r]
        
        V[2*3+0, col] = v213[0, r]
        V[2*3+1, col] = v213[1, r]
        V[2*3+2, col] = v213[2, r]
        
        W[:, col] = w213[:, r]
        
        col += 1
        
    return U, V, W

def _build_2x5x6_v2():
    """
    Build 2x5x6 decomposition using split 5=3+2, 6=3+3.
    A = [A1 (2x3) | A2 (2x2)]
    B = [B1 (3x3); B2 (2x3)]
    C = A1 B1 + A2 B2
    A1 B1: 2x3x3, Rank 17.
    A2 B2: 2x2x3, Rank 11.
    Total 28.
    """
    n, m, p = 2, 5, 6
    rank = 28
    
    U = np.zeros((n*m, rank)) # 10x28
    V = np.zeros((m*p, rank)) # 30x28
    W = np.zeros((p*n, rank)) # 12x28
    
    u233, v233, w233 = _decomp_233()
    u223, v223, w223 = _decomp_223()
    
    col = 0
    
    # Part 1: A1 * B1 (2x3x3)
    # A1: j in {0,1,2}. A indices i*5+j.
    # u233 U: i*3+j.
    # Map U[i*3+j, r] to U[i*5+j, col].
    
    # B1: j in {0,1,2}, k in {0,1,2}. B indices j*6+k.
    # v233 V: j*3+k.
    # Map V[j*3+k, r] to V[j*6+k, col].
    
    # W: k in {0,1,2}, i in {0,1}. W indices k*2+i.
    # w233 W: k*2+i.
    # Map W[k*2+i, r] to W[k*2+i, col].
    
    for r in range(17):
        # U
        for i in range(2):
            for j in range(3):
                U[i*5+j, col] = u233[i*3+j, r]
                
        # V
        for j in range(3):
            for k in range(3):
                V[j*6+k, col] = v233[j*3+k, r]
                
        # W
        for k in range(3):
            for i in range(2):
                W[k*2+i, col] = w233[k*2+i, r]
                
        col += 1
        
    # Part 2: A2 * B2 (2x2x3)
    # A2: j in {3,4}. A indices i*5+j.
    # u223 U: i*2+j (local j 0,1).
    # Map U[i*2+j, r] to U[i*5+3+j, col].
    
    # B2: j in {3,4}, k in {0,1,2}. B indices j*6+k.
    # v223 V: j*3+k (local j 0,1).
    # Map V[j*3+k, r] to V[3+j*6+k, col]? No.
    # Global j = 3 + local_j.
    # V index = global_j * 6 + k.
    # v223 index = local_j * 3 + k.
    
    # W: k in {0,1,2}, i in {0,1}. W indices k*2+i.
    # w223 W: k*2+i.
    # Map W[k*2+i, r] to W[k*2+i, col].
    
    for r in range(11):
        # U
        for i in range(2):
            for j in range(2):
                U[i*5+3+j, col] = u223[i*2+j, r]
                
        # V
        for j in range(2):
            for k in range(3):
                V[(3+j)*6+k, col] = v223[j*3+k, r]
                
        # W
        for k in range(3):
            for i in range(2):
                W[k*2+i, col] = w223[k*2+i, r]
                
        col += 1
        
    assert col == rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
