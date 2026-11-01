"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U,V,W exactly decompose the multiplication tensor."""
    rank = len(U[0]) if U else 0
    U3 = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V3 = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W3 = np.array(W, dtype=np.float64).reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    target = np.ones((n, m, p))
    return np.allclose(result, target, atol=tol, rtol=0)

def _strassen_2x2_decomp():
    """Return U,V,W for 2x2x2 Strassen decomposition (rank 7).
    U[i,j,c], V[j,k,c], W[k,i,c] such that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k.
    """
    # Strassen: M1..M7
    # C00 = M1 - M4 + M6 + M7
    # C01 = M2 + M4
    # C10 = M3 + M5
    # C11 = M1 - M2 + M5 + M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # For tensor: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # We need this to equal 1 for all (i,j,k) in the 2x2x2 case.
    #
    # Standard Strassen decomposition of the 2x2x2 matmul tensor:
    # The tensor T[i,j,k] has T[i,j,k]=1 for all i,j,k in {0,1}.
    #
    # Rank-7 decomposition:
    # c=0: U[0,0]=1,U[1,1]=1, V[0,0]=1,V[1,1]=1, W[0,0]=1,W[1,1]=1  -> M1
    # c=1: U[1,0]=1,U[1,1]=1, V[0,0]=1, W[0,1]=1  -> M2
    # c=2: U[0,0]=1, V[1,0]=1,V[1,1]=-1, W[0,1]=1  -> M3
    # c=3: U[1,1]=1, V[0,1]=1,V[1,0]=-1, W[0,1]=1  -> M4
    # c=4: U[0,0]=1,U[0,1]=1, V[1,1]=1, W[0,1]=1  -> M5
    # c=5: U[1,0]=1,U[0,0]=-1, V[0,0]=1,V[1,0]=1, W[0,1]=1  -> M6
    # c=6: U[0,1]=1,U[1,1]=-1, V[0,1]=1,V[1,1]=1, W[0,1]=1  -> M7
    #
    # Let me verify c=0: sum over (i,j,k) of U[i,j,0]*V[j,k,0]*W[k,i,0]
    # U[0,0,0]=1, U[1,1,0]=1, V[0,0,0]=1, V[1,1,0]=1, W[0,0,0]=1, W[1,1,0]=1
    # Contribution to T[0,0,0]: U[0,0,0]*V[0,0,0]*W[0,0,0] = 1
    # T[0,0,1]: U[0,0,0]*V[0,1,0]*W[1,0,0] = 1*0*0 = 0
    # T[0,1,0]: U[0,1,0]*V[1,0,0]*W[0,1,0] = 0*0*0 = 0
    # T[0,1,1]: 0
    # T[1,0,0]: U[1,0,0]*V[0,0,0]*W[0,1,0] = 0
    # T[1,1,0]: U[1,1,0]*V[1,0,0]*W[0,1,0] = 0
    # T[1,0,1]: 0
    # T[1,1,1]: U[1,1,0]*V[1,1,0]*W[1,1,0] = 1
    # So c=0 contributes 1 to T[0,0,0] and T[1,1,1]. Good for M1.
    #
    # This is getting complex. Let me use a known explicit rank-7 decomposition.
    #
    # Actually, for the all-ones tensor 2x2x2, a rank-7 decomposition is:
    # (from literature on Strassen)
    # Let me just use the standard Strassen formulas directly.
    
    U = np.zeros((2, 2, 7))
    V = np.zeros((2, 2, 7))
    W = np.zeros((2, 2, 7))
    
    # c=0: M1 = (A00+A11)(B00+B11)
    U[0,0,0] = 1; U[1,1,0] = 1
    V[0,0,0] = 1; V[1,1,0] = 1
    W[0,0,0] = 1; W[1,1,0] = 1
    
    # c=1: M2 = (A10+A11)B00
    U[1,0,1] = 1; U[1,1,1] = 1
    V[0,0,1] = 1
    W[0,1,1] = 1
    
    # c=2: M3 = A00(B10-B11)
    U[0,0,2] = 1
    V[1,0,2] = 1; V[1,1,2] = -1
    W[0,1,2] = 1
    
    # c=3: M4 = A11(B01-B10)
    U[1,1,3] = 1
    V[0,1,3] = 1; V[1,0,3] = -1
    W[0,1,3] = 1
    
    # c=4: M5 = (A00+A01)B11
    U[0,0,4] = 1; U[0,1,4] = 1
    V[1,1,4] = 1
    W[0,1,4] = 1
    
    # c=5: M6 = (A10-A00)(B00+B10)
    U[1,0,5] = 1; U[0,0,5] = -1
    V[0,0,5] = 1; V[1,0,5] = 1
    W[0,1,5] = 1
    
    # c=6: M7 = (A01-A11)(B01+B11)
    U[0,1,6] = 1; U[1,1,6] = -1
    V[0,1,6] = 1; V[1,1,6] = 1
    W[0,1,6] = 1
    
    return U, V, W

def _build_strassen_2x3x2():
    """Build rank-9 decomposition for 2x3x2 tensor (all ones).
    Uses Strassen (rank 7) for j=0,1 and 2 rank-1 terms for j=2.
    """
    U2, V2, W2 = _strassen_2x2_decomp()
    rank = 9
    U = np.zeros((2, 3, rank))
    V = np.zeros((3, 2, rank))
    W = np.zeros((2, 2, rank))
    
    # Copy Strassen part for j=0,1
    U[:, 0:2, 0:7] = U2
    V[0:2, :, 0:7] = V2
    W[:, :, 0:7] = W2
    
    # j=2, k=0: rank-1 term
    U[:, 2, 7] = 1.0
    V[2, 0, 7] = 1.0
    W[0, :, 7] = 1.0
    
    # j=2, k=1: rank-1 term
    U[:, 2, 8] = 1.0
    V[2, 1, 8] = 1.0
    W[1, :, 8] = 1.0
    
    return U, V, W

def _build_strassen_2x3x8():
    """Build rank-36 decomposition for 2x3x8 using 4 blocks of 2x3x2.
    Each block uses rank 9, total rank = 4*9 = 36.
    """
    n, m, p = 2, 3, 8
    num_blocks = p // 2  # 4 blocks
    block_rank = 9
    total_rank = num_blocks * block_rank  # 36
    
    U = np.zeros((n, m, total_rank))
    V = np.zeros((m, p, total_rank))
    W = np.zeros((p, n, total_rank))
    
    U_blk, V_blk, W_blk = _build_strassen_2x3x2()
    
    for b in range(num_blocks):
        offset = b * block_rank
        k_start = b * 2
        # U doesn't depend on k block
        U[:, :, offset:offset+block_rank] = U_blk
        # V: j x (2k) for this block
        V[:, k_start:k_start+2, offset:offset+block_rank] = V_blk
        # W: (2k) x i for this block
        W[k_start:k_start+2, :, offset:offset+block_rank] = W_blk
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Build explicit Strassen-based rank-36 decomposition
    U3, V3, W3 = _build_strassen_2x3x8()
    
    # Verify
    if _verify(U3.reshape(n*m, -1).tolist(), V3.reshape(m*p, -1).tolist(), 
               W3.reshape(p*n, -1).tolist(), n, m, p, tol=1e-9):
        U_list = U3.reshape(n*m, -1).tolist()
        V_list = V3.reshape(m*p, -1).tolist()
        W_list = W3.reshape(p*n, -1).tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 48
    best_rank = n * m * p
    U_sb = [[0] * best_rank for _ in range(n * m)]
    V_sb = [[0] * best_rank for _ in range(m * p)]
    W_sb = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
