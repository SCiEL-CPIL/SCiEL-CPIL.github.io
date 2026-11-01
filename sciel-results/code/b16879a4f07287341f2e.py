"""Matrix multiplication 2x3x6 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_2x3x3_rank7():
    """
    Construct a rank-7 decomposition for 2x3x3 matrix multiplication.
    
    The multiplication tensor for C = A*B where A is 2x3, B is 3x3, C is 2x3:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    U shape: (6, 7)  -- indices (i*m+j, r)
    V shape: (9, 7)  -- indices (j*p+k, r)
    W shape: (6, 7)  -- indices (k*n+i, r)
    
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r] = delta_{i,i'}*delta_{k,k'}
    """
    n, m, p = 2, 3, 3
    rank = 7
    
    # We need 7 terms. Let me construct them using a systematic approach.
    # 
    # Key insight: For 2x3x3, we can use the following structure:
    # - 3 terms for row 0 of A (i=0): each handles one column of B
    # - 3 terms for row 1 of A (i=1): each handles one column of B
    # - 1 extra term to handle the cross-terms
    #
    # Actually, let me use a known construction. The rank of 2x3x3 is 7.
    # 
    # Here's a valid rank-7 decomposition:
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    #   U[0,0]=1, V[0,0]=1, W[0,0]=1
    # Term 1: A[0,1]*B[1,0] -> C[0,0]
    #   U[1,1]=1, V[3,1]=1, W[0,1]=1
    # Term 2: A[0,2]*B[2,0] -> C[0,0]
    #   U[2,2]=1, V[6,2]=1, W[0,2]=1
    # Term 3: A[0,0]*B[0,1] -> C[0,1]
    #   U[0,3]=1, V[1,3]=1, W[3,3]=1
    # Term 4: A[0,1]*B[1,1] -> C[0,1]
    #   U[1,4]=1, V[4,4]=1, W[3,4]=1
    # Term 5: A[0,2]*B[2,1] -> C[0,1]
    #   U[2,5]=1, V[7,5]=1, W[3,5]=1
    # Term 6: A[1,0]*B[0,2] -> C[1,2]
    #   U[3,6]=1, V[2,6]=1, W[6,6]=1
    #
    # Wait, this only gives 7 terms but doesn't cover all 18 products.
    # I need to be more systematic.
    
    # Let me use a different approach: solve for the decomposition directly.
    # 
    # For 2x3x3, the schoolbook has 18 terms (one for each i,j,k triple).
    # We need to find 7 rank-1 tensors that sum to the multiplication tensor.
    #
    # Let me try a structured approach based on known results.
    # The 2x3x3 tensor rank is 7, and one known decomposition uses:
    
    U = np.zeros((6, rank))
    V = np.zeros((9, rank))
    W = np.zeros((6, rank))
    
    # Indices: U[i*3+j, r], V[j*3+k, r], W[k*2+i, r]
    
    # Term 0: contributes to C[0,0] from A[0,0]*B[0,0]
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1: contributes to C[0,0] from A[0,1]*B[1,0]
    U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
    # Term 2: contributes to C[0,0] from A[0,2]*B[2,0]
    U[2, 2] = 1; V[6, 2] = 1; W[0, 2] = 1
    # Term 3: contributes to C[0,1] from A[0,0]*B[0,1]
    U[0, 3] = 1; V[1, 3] = 1; W[2, 3] = 1
    # Term 4: contributes to C[0,1] from A[0,1]*B[1,1]
    U[1, 4] = 1; V[4, 4] = 1; W[2, 4] = 1
    # Term 5: contributes to C[0,1] from A[0,2]*B[2,1]
    U[2, 5] = 1; V[7, 5] = 1; W[2, 5] = 1
    # Term 6: contributes to C[1,2] from A[1,0]*B[0,2]
    U[3, 6] = 1; V[2, 6] = 1; W[4, 6] = 1
    
    # Let me verify this... it's clearly incomplete. This approach won't work
    # with just 7 naive terms.
    
    # Let me try a completely different approach: use numerical optimization
    # to find a rank-7 decomposition.
    
    return None  # Fallback to schoolbook

def _find_2x3x3_decomposition():
    """
    Find a rank-7 decomposition for 2x3x3 matrix multiplication.
    
    Returns (U, V, W) numpy arrays with shapes (6,7), (9,7), (6,7) or None.
    """
    n, m, p = 2, 3, 3
    rank = 7
    
    # Construct a rank-7 decomposition using the following strategy:
    # Use terms that exploit the structure to cover multiple products at once.
    
    U = np.zeros((6, rank))
    V = np.zeros((9, rank))
    W = np.zeros((6, rank))
    
    # Term 0: Handle C[0,0] from all 3 products
    # A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] -> C[0,0]
    # This requires: u[0]=1, u[1]=1, u[2]=1 (for i=0, j=0,1,2)
    #               v[0]=1, v[3]=1, v[6]=1 (for j=0,1,2, k=0)
    #               w[0]=1 (for k=0, i=0)
    U[0, 0] = 1; U[1, 0] = 1; U[2, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1; V[6, 0] = 1
    W[0, 0] = 1
    
    # But this also contributes to other C entries! Let's check:
    # For C[0,1]: u[0]*v[1]*w[2] + u[1]*v[4]*w[2] + u[2]*v[7]*w[2] = 1*0*0 + 1*0*0 + 1*0*0 = 0 OK
    # For C[0,2]: similar = 0 OK
    # For C[1,0]: u[3]*v[0]*w[0] + u[4]*v[3]*w[0] + u[5]*v[6]*w[0] = 0*1*1 + 0*1*1 + 0*1*1 = 0 OK
    # Good, term 0 only contributes to C[0,0].
    
    # Term 1: Handle C[0,1] from all 3 products
    U[0, 1] = 1; U[1, 1] = 1; U[2, 1] = 1
    V[1, 1] = 1; V[4, 1] = 1; V[7, 1] = 1
    W[2, 1] = 1
    
    # Term 2: Handle C[0,2] from all 3 products
    U[0, 2] = 1; U[1, 2] = 1; U[2, 2] = 1
    V[2, 2] = 1; V[5, 2] = 1; V[8, 2] = 1
    W[4, 2] = 1
    
    # Term 3: Handle C[1,0] from all 3 products
    U[3, 3] = 1; U[4, 3] = 1; U[5, 3] = 1
    V[0, 3] = 1; V[3, 3] = 1; V[6, 3] = 1
    W[1, 3] = 1
    
    # Term 4: Handle C[1,1] from all 3 products
    U[3, 4] = 1; U[4, 4] = 1; U[5, 4] = 1
    V[1, 4] = 1; V[4, 4] = 1; V[7, 4] = 1
    W[3, 4] = 1
    
    # Term 5: Handle C[1,2] from all 3 products
    U[3, 5] = 1; U[4, 5] = 1; U[5, 5] = 1
    V[2, 5] = 1; V[5, 5] = 1; V[8, 5] = 1
    W[5, 5] = 1
    
    # That's only 6 terms! We need 7. But wait, let me verify if 6 terms suffice.
    # If 6 terms work, then rank <= 6, contradicting the known rank of 7.
    # So there must be an error in my construction.
    
    # Let me verify term 0 more carefully:
    # Contribution to C[i',k'] = sum_j U[i'*3+j, 0] * V[j*3+k', 0] * W[k'*2+i', 0]
    # For i'=0, k'=0: sum_j U[j,0]*V[j*3,0]*W[0,0] = U[0,0]*V[0,0]*W[0,0] + U[1,0]*V[3,0]*W[0,0] + U[2,0]*V[6,0]*W[0,0]
    # = 1*1*1 + 1*1*1 + 1*1*1 = 3, but we need 1!
    
    # Ah, I see the problem. Each term contributes 3 to C[0,0], not 1.
    # I need to scale: use 1/3 for each.
    
    # But the problem says entries should be integer or half-integer. 1/3 is not allowed!
    # So this approach doesn't work with the given constraints.
    
    return None

def construct():
    """
    Construct rank-minimal decomposition for 2x3x6 matrix multiplication.
    """
    # Try to find a rank-7 decomposition for 2x3x3
    decomp_233 = _find_2x3x3_decomposition()
    
    if decomp_233 is not None:
        U233, V233, W233 = decomp_233
        # Combine two copies for 2x3x6
        n, m, p = 2, 3, 6
        rank = 14
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # First half: k=0,1,2
        U[:, 0:7] = U233
        V[:, 0:7] = V233  # But V233 has shape (9,7) and we need (18,7)
        # Need to map V indices: for k in 0..2, V[j*3+k] in 2x3x3 maps to V[j*6+k] in 2x3x6
        V_full = np.zeros((18, 7))
        for j in range(3):
            for k in range(3):
                V_full[j*6+k, :] = V233[j*3+k, :]
        V[:, 0:7] = V_full
        
        W_full = np.zeros((12, 7))
        for k in range(3):
            for i in range(2):
                W_full[k*2+i, :] = W233[k*2+i, :]
        W[:, 0:7] = W_full
        
        # Second half: k=3,4,5
        U[:, 7:14] = U233
        V_full2 = np.zeros((18, 7))
        for j in range(3):
            for k in range(3):
                V_full2[j*6+(k+3), :] = V233[j*3+k, :]
        V[:, 7:14] = V_full2
        
        W_full2 = np.zeros((12, 7))
        for k in range(3):
            for i in range(2):
                W_full2[(k+3)*2+i, :] = W233[k*2+i, :]
        W[:, 7:14] = W_full2
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 36
    n, m, p = 2, 3, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
