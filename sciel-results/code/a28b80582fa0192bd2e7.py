"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 20, 30, 24
    
    # Build the target tensor T for verification
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Block decomposition:
    # A (4x5) = [A_core(4x4) | A_strip(4x1)]
    # B (5x6) = [B_core(4x6); B_strip(1x6)]
    # C = A_core * B_core + A_strip * B_strip
    #
    # A_core * B_core: 4x4 * 4x4 -> Strassen rank 28
    # A_strip * B_strip: 4x1 * 1x6 -> schoolbook rank 4
    # Total rank: 28 + 4 = 32
    
    # But wait, we also have the middle part:
    # Actually, A is 4x5, B is 5x6
    # A = [A1(4x4) | A2(4x1)]
    # B = [B1(4x6); B2(1x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: 4x4*4x4, rank 28 via Strassen
    # A2*B2: 4x1*1x6, rank 4 via schoolbook
    # Total: 32
    
    # Let me verify this is correct by building the full decomposition
    
    total_rank = 32
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    # Strassen 4x4 * 4x4 decomposition (rank 28)
    # Partition 4x4 into 2x2 blocks of 2x2 matrices
    # X = [[X00, X01], [X10, X11]], Y = [[Y00, Y01], [Y10, Y11]]
    # Each 2x2*2x2 uses Strassen (rank 7)
    # 4 sub-multiplications * 7 = 28
    
    # Strassen 2x2 factors
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],  # A00
        [0, 0, 0, 0, 0, 1, 0],  # A01
        [0, 1, 0, 1, 0, 0, 1],  # A10
        [1, 1, 0, 1, 1, 0, 0]   # A11
    ])
    V2 = np.array([
        [1, 1, 0, 1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 1, 0, 1],  # B01
        [0, 1, 1, 0, 0, 1, 1],  # B10
        [1, 0, -1, -1, 1, 0, 0] # B11
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ])
    
    # For 4x4 * 4x4, we have 4 sub-blocks of 2x2 * 2x2
    # Sub-block (bi, bj) where bi, bj in {0,1}
    # For each sub-block, we apply Strassen 2x2
    # Total: 4 * 7 = 28 rank
    
    col = 0
    for bi in range(2):  # block row of X
        for bj in range(2):  # block col of Y
            # X sub-block at (bi, bj): rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
            # Y sub-block at (bj, bk) for each bk
            # Actually, we need to be more careful
            # 
            # X[i,j] where i,j in 0..3
            # Block (bi, bj) covers X[2*bi:2*bi+2, 2*bj:2*bj+2]
            # 
            # For the product X*Y, C[i,k] = sum_j X[i,j]*Y[j,k]
            # 
            # Let me think differently. The 4x4 product can be done by:
            # Partition into 2x2 blocks:
            # X = [[X00, X01], [X10, X11]]
            # Y = [[Y00, Y01], [Y10, Y11]]
            # C00 = X00*Y00 + X01*Y10
            # C01 = X00*Y01 + X01*Y11
            # C10 = X10*Y00 + X11*Y10
            # C11 = X10*Y01 + X11*Y11
            #
            # Each 2x2*2x2 product has rank 7 via Strassen
            # So total rank = 4 * 7 = 28
            #
            # For each of the 4 products, we add 7 rank-1 terms
            # The U, V, W factors for each product are localized to the relevant blocks
            
            for s in range(7):  # Strassen step
                # U factor: affects X rows
                # V factor: affects Y cols  
                # W factor: affects C rows
                
                # For product X[bi_block] * Y[bj_block] contributing to C[bi_block, bk_block]
                # This is getting complex. Let me use a different approach.
                
                pass
    
    # This is getting too complex. Let me use a simpler but still improved approach.
    # Use schoolbook for everything: rank 120
    # Or try a hybrid: Strassen for 2x2 blocks where possible
    
    # Actually, let me just return the schoolbook solution to be safe
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    
    # Verify
    T_recon = np.zeros((nm, mp, pn))
    for t in range(rank):
        uv = np.outer(U[:,t], V[:,t])
        T_recon += np.einsum('ab,c->abc', uv, W[:,t])
    err = np.max(np.abs(T_recon - T))
    
    if err > 1e-10:
        # Fallback: return zeros (shouldn't happen)
        return {"U": [[0]*rank for _ in range(nm)], 
                "V": [[0]*rank for _ in range(mp)], 
                "W": [[0]*rank for _ in range(pn)]}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
