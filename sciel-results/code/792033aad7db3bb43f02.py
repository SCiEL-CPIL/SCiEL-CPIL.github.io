"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 2, 3, 9
    
    def verify(U, V, W):
        """Verify: sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1 for all i,j,k."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Schoolbook decomposition: rank = n*m*p = 54
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Try structured rank-50 decomposition using block structure
    # Split inner dim 3 = 2+1:
    # A(2x3) = [A1(2x2) | a(2x1)], B(3x9) = [B1(2x9); b(1x9)]
    # C = A1*B1 + a*b
    # 
    # A1*B1 is MM_{2,2,9}, a*b is MM_{2,1,9} (rank 18)
    # If we can do MM_{2,2,9} at rank 32, total = 50
    #
    # For MM_{2,2,9}: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Use Strassen-like: for the 2x2 part, use 7 multiplications
    # But each "multiplication" here is scalar * (1x9 vector), costing 9 mults
    # 7 * 9 = 63 > 36. Not helpful.
    #
    # Alternative: direct construction with 32 terms
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in 0..8
    # Total 18 outputs, 36 products schoolbook.
    # 
    # Try: for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    # This is (1x2) * (2x9) = 1x9, rank 18 per i, total 36.
    #
    # I'll use schoolbook for now as it's guaranteed correct.
    
    U, V, W, rank = schoolbook()
    
    if not verify(U, V, W):
        raise ValueError("Schoolbook verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
