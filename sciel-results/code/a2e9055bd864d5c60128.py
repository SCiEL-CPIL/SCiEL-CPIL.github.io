import numpy as np

DIMENSIONS = [2, 5, 6]

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 matrix multiplication.
    Returns U, V, W of shape (4, 7) each.
    U[i, r] corresponds to A[i, j] coefficients
    V[j, r] corresponds to B[j, k] coefficients
    W[k, i, r] -> flattened W[k*2 + i, r] corresponds to C[k, i] coefficients
    """
    # 7 rank terms
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indexing: i in {0,1}, j in {0,1} -> idx = i*2 + j
    # k in {0,1}, i in {0,1} -> idx = k*2 + i
    
    # M1
    U[0*2+0, 0] = 1.0  # A00
    U[1*2+1, 0] = 1.0  # A11
    V[0*2+0, 0] = 1.0  # B00
    V[1*2+1, 0] = 1.0  # B11
    W[0*2+0, 0] = 1.0  # C00
    W[1*2+1, 0] = 1.0  # C11
    
    # M2
    U[1*2+0, 1] = 1.0  # A10
    U[1*2+1, 1] = 1.0  # A11
    V[0*2+0, 1] = 1.0  # B00
    W[1*2+0, 1] = 1.0  # C10
    W[1*2+1, 1] = -1.0 # C11
    
    # M3
    U[0*2+0, 2] = 1.0  # A00
    V[0*2+1, 2] = 1.0  # B01
    V[1*2+1, 2] = -1.0 # B11
    W[0*2+1, 2] = 1.0  # C01
    W[1*2+1, 2] = 1.0  # C11
    
    # M4
    U[1*2+1, 3] = 1.0  # A11
    V[1*2+0, 3] = 1.0  # B10
    V[0*2+0, 3] = -1.0 # B00
    W[0*2+0, 3] = 1.0  # C00
    W[1*2+0, 3] = 1.0  # C10
    
    # M5
    U[0*2+0, 4] = 1.0  # A00
    U[0*2+1, 4] = 1.0  # A01
    V[1*2+1, 4] = 1.0  # B11
    W[0*2+0, 4] = -1.0 # C00
    W[0*2+1, 4] = 1.0  # C01
    
    # M6
    U[1*2+0, 5] = 1.0  # A10
    U[0*2+0, 5] = -1.0 # A00
    V[0*2+0, 5] = 1.0  # B00
    V[0*2+1, 5] = 1.0  # B01
    W[1*2+1, 5] = 1.0  # C11
    
    # M7
    U[0*2+1, 6] = 1.0  # A01
    U[1*2+1, 6] = -1.0 # A11
    V[1*2+0, 6] = 1.0  # B10
    V[1*2+1, 6] = 1.0  # B11
    W[0*2+0, 6] = 1.0  # C00
    
    return U, V, W

def _trivial_2x1x2():
    """
    Trivial decomposition for 2x1 * 1x2 multiplication.
    Rank 2.
    U shape (2, 2), V shape (1, 2), W shape (2, 2)
    """
    U = np.zeros((2, 2))
    V = np.zeros((1, 2))
    W = np.zeros((2, 2))
    
    # Term 0: A0 * B0 -> C0
    U[0, 0] = 1.0
    V[0, 0] = 1.0
    W[0, 0] = 1.0
    
    # Term 1: A1 * B1 -> C1
    U[1, 1] = 1.0
    V[0, 1] = 1.0
    W[1, 1] = 1.0
    
    return U, V, W

def _block_2x3x2():
    """
    Decomposition for 2x3 * 3x2 multiplication.
    Split j dimension into j=0,1 (Strassen 2x2x2) and j=2 (Trivial 2x1x2).
    Rank 7 + 2 = 9.
    U shape (2*3, 9) = (6, 9)
    V shape (3*2, 9) = (6, 9)
    W shape (2*2, 9) = (4, 9)
    """
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _trivial_2x1x2()
    
    U = np.zeros((6, 9))
    V = np.zeros((6, 9))
    W = np.zeros((4, 9))
    
    # Part 1: j in {0, 1}
    # U indices: i in {0,1}, j in {0,1} -> i*3 + j
    for i in range(2):
        for j in range(2):
            U[i*3 + j, :7] = U1[i, :7]
    # V indices: j in {0,1}, k in {0,1} -> j*2 + k
    for j in range(2):
        for k in range(2):
            V[j*2 + k, :7] = V1[j, :7]
    # W indices: k in {0,1}, i in {0,1} -> k*2 + i
    for k in range(2):
        for i in range(2):
            W[k*2 + i, :7] = W1[k, i, 0] if False else W1[k*2+i, :7]
            
    # Part 2: j = 2
    # U indices: i in {0,1}, j=2 -> i*3 + 2
    for i in range(2):
        U[i*3 + 2, 7:9] = U2[i, :]
    # V indices: j=2, k in {0,1} -> 2*2 + k
    for k in range(2):
        V[2*2 + k, 7:9] = V2[0, :]
    # W indices: k in {0,1}, i in {0,1} -> k*2 + i
    for k in range(2):
        for i in range(2):
            W[k*2 + i, 7:9] = W2[k*2+i, :]
            
    return U, V, W

def _block_2x2x6():
    """
    Decomposition for 2x2 * 2x6 multiplication.
    Split k dimension into 3 blocks of size 2.
    Each block is Strassen 2x2x2.
    Rank 7 * 3 = 21.
    U shape (2*2, 21) = (4, 21)
    V shape (2*6, 21) = (12, 21)
    W shape (6*2, 21) = (12, 21)
    """
    U1, V1, W1 = _strassen_2x2x2()
    
    U = np.zeros((4, 21))
    V = np.zeros((12, 21))
    W = np.zeros((12, 21))
    
    for block in range(3):
        # k offset = block * 2
        # U is same for all k blocks in this local decomposition structure?
        # Wait, the tensor is T[i,j,k].
        # If we split k, the U part (coefficients of A[i,j]) is independent of k block?
        # No, the rank terms are specific to the (i,j,k) triple.
        # In the trivial decomposition, each (i,j,k) has its own rank term.
        # In Strassen, the rank terms couple i,j,k.
        # If we split k into blocks, we can use independent Strassen decompositions for each k-block.
        # For block b (k in {2b, 2b+1}):
        # The A part is the same A (2x2).
        # The B part is B[:, 2b:2b+2] (2x2).
        # The C part is C[2b:2b+2, :] (2x2).
        
        # U: A[i,j] coefficients. Same for all blocks?
        # Yes, the left factor U multiplies A. The structure of A is the same.
        # But the rank terms are distinct.
        # U[i, r] for r in block b.
        # In Strassen, U[i, r] depends on i and the specific rank term r.
        # So we just repeat the U1 pattern for each block.
        
        U[:, block*7 : (block+1)*7] = U1
        
        # V: B[j, k] coefficients.
        # For block b, local k' in {0,1} maps to global k = 2b + k'.
        # V index: j*6 + k.
        for j in range(2):
            for k_local in range(2):
                k_global = 2*block + k_local
                V[j*6 + k_global, block*7 : (block+1)*7] = V1[j, k_local, 0] if False else V1[j, :7][k_local] # V1 is (4,7) so V1[j, :] is row j.
                # Actually V1 is (4,7) where index is j*2+k_local? No, V1 was (4,7) from 2x2x2?
                # In _strassen_2x2x2, V is (4,7). Indices are j*2+k.
                # So V1[j*2+k_local, :] is the row.
                V[j*6 + k_global, block*7 : (block+1)*7] = V1[j*2 + k_local, :]
                
        # W: C[k, i] coefficients.
        # W index: k*n + i = k*2 + i.
        for k_local in range(2):
            k_global = 2*block + k_local
            for i in range(2):
                W[k_global*2 + i, block*7 : (block+1)*7] = W1[k_local*2 + i, :]
                
    return U, V, W

def _block_2x3x6():
    """
    Decomposition for 2x3 * 3x6 multiplication.
    Split k dimension into 3 blocks of size 2.
    Each block is block_2x3x2.
    Rank 9 * 3 = 27.
    U shape (2*3, 27) = (6, 27)
    V shape (3*6, 27) = (18, 27)
    W shape (6*2, 27) = (12, 27)
    """
    U1, V1, W1 = _block_2x3x2()
    
    U = np.zeros((6, 27))
    V = np.zeros((18, 27))
    W = np.zeros((12, 27))
    
    for block in range(3):
        U[:, block*9 : (block+1)*9] = U1
        
        # V: B[j, k] where j in {0,1,2}, k in {0..5}
        # For block b, k in {2b, 2b+1}
        for j in range(3):
            for k_local in range(2):
                k_global = 2*block + k_local
                V[j*6 + k_global, block*9 : (block+1)*9] = V1[j*2 + k_local, :]
                
        # W: C[k, i] where k in {0..5}, i in {0,1}
        for k_local in range(2):
            k_global = 2*block + k_local
            for i in range(2):
                W[k_global*2 + i, block*9 : (block+1)*9] = W1[k_local*2 + i, :]
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # Split m dimension into m=0,1 (2x2x6 block) and m=2,3,4 (2x3x6 block)
    # Block 1: i in {0,1}, j in {0,1}, k in {0..5} -> Rank 21
    U1, V1, W1 = _block_2x2x6()
    
    # Block 2: i in {0,1}, j in {2,3,4}, k in {0..5} -> Rank 27
    U2, V2, W2 = _block_2x3x6()
    
    total_rank = 21 + 27
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Assemble Block 1
    # U indices: i*m + j. i in {0,1}, j in {0,1}
    for i in range(2):
        for j in range(2):
            U[i*5 + j, :21] = U1[i*2 + j, :]
            
    # V indices: j*p + k. j in {0,1}, k in {0..5}
    for j in range(2):
        for k in range(6):
            V[j*6 + k, :21] = V1[j*2 + k, :] # V1 is (12, 21) from _block_2x2x6. Indices j*2+k?
            # In _block_2x2x6, V was (12, 21) with index j*6+k?
            # Let's re-verify _block_2x2x6 V shape.
            # _block_2x2x6 returns V of shape (12, 21).
            # In _block_2x2x6, V indices were j*6 + k_global.
            # So V1[j*6 + k, :] is correct.
            
    # W indices: k*n + i. k in {0..5}, i in {0,1}
    for k in range(6):
        for i in range(2):
            W[k*2 + i, :21] = W1[k*2 + i, :]
            
    # Assemble Block 2
    # U indices: i*m + j. i in {0,1}, j in {2,3,4}
    for i in range(2):
        for j in range(3):
            U[i*5 + (j+2), 21:48] = U2[i*3 + j, :]
            
    # V indices: j*p + k. j in {2,3,4}, k in {0..5}
    for j in range(3):
        for k in range(6):
            V[(j+2)*6 + k, 21:48] = V2[j*6 + k, :] # V2 is (18, 27) from _block_2x3x6. Indices j*6+k?
            # In _block_2x3x6, V was (18, 27) with index j*6+k?
            # In _block_2x3x6, V indices were j*6 + k_global.
            # So V2[j*6 + k, :] is correct.
            
    # W indices: k*n + i. k in {0..5}, i in {0,1}
    for k in range(6):
        for i in range(2):
            W[k*2 + i, 21:48] = W2[k*2 + i, :]
            
    return {"U": U, "V": V, "W": W}
