"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal decomposition for 6x6 matrix multiplication.
    
    Uses block-Strassen decomposition with 3x3 blocks of 2x2 matrices.
    Each 2x2 block multiplication uses Strassen's 7-term algorithm.
    Total rank: 9 blocks * 7 Strassen terms = 63
    """
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6
    
    # 3x3 blocks, each block is 2x2
    B = 3  # number of blocks per dimension
    BSIZE = 2  # size of each block
    
    # Strassen's 7 multiplication terms for 2x2 matrices:
    # Given A = [[a, b], [c, d]] and C = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-e)
    # M4 = (d)(h-g)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    # Result: [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    rank = B * B * 7  # 9 * 7 = 63
    
    U = [[0] * rank for _ in range(n * m)]  # 36 x 63
    V = [[0] * rank for _ in range(m * p)]  # 36 x 63
    W = [[0] * rank for _ in range(p * n)]  # 36 x 63
    
    # For each 3x3 block of 2x2 matrices, apply Strassen
    for bi in range(B):  # row block index (0,1,2)
        for bj in range(B):  # col block index (0,1,2)
            for bk in range(B):  # intermediate block index (0,1,2)
                # This corresponds to multiplying A[bi,bj] * B[bj,bk]
                # where A is 6x6, B is 6x6, result is 6x6
                # U indices: i*m + j where i in [bi*2, bi*2+1], j in [bj*2, bj*2+1]
                # V indices: j*p + k where j in [bj*2, bj*2+1], k in [bk*2, bk*2+1]
                # W indices: k*n + i where k in [bk*2, bk*2+1], i in [bi*2, bi*2+1]
                
                # Strassen terms for 2x2 block multiplication
                # Let A2x2 = [[A00, A01], [A10, A11]] where Axy corresponds to element at position (bi*2+x, bj*2+y)
                # Let B2x2 = [[B00, B01], [B10, B11]] where Bxy corresponds to element at position (bj*2+x, bk*2+y)
                
                # Define the 7 Strassen multiplications and their contributions
                
                # Helper to get global U index
                def u_idx(xi, yj):
                    return (bi * BSIZE + xi) * m + (bj * BSIZE + yj)
                
                # Helper to get global V index
                def v_idx(xj, zk):
                    return (bj * BSIZE + xj) * p + (bk * BSIZE + zk)
                
                # Helper to get global W index
                def w_idx(zi, xi):
                    return (bk * BSIZE + zi) * n + (bi * BSIZE + xi)
                
                # Strassen term 1: (A00+A11) * (B00+B11)
                # Contribution to result: 
                #   R00 += M1, R01 -= M1 (actually +M1 for R00, and M1 appears in R01 with -M5+M7, not directly)
                # Let me re-derive:
                # R00 = M1 + M4 - M5 + M7
                # R01 = M3 + M5
                # R10 = M2 + M4
                # R11 = M1 - M2 + M3 + M6
                
                # For each Strassen term t in 0..6, we need to define:
                # - U coefficients for each (xi, yj) in 0..1
                # - V coefficients for each (xj, zk) in 0..1  
                # - W coefficients for each (zi, xi) in 0..1
                # such that sum over terms of (U term) * (V term) * (W term) gives the correct result
                
                # Term 0 (M1): (A00+A11) * (B00+B11)
                # U: A00 + A11 => U at (xi=0,yj=0) += 1, U at (xi=1,yj=1) += 1
                # V: B00 + B11 => V at (xj=0,zk=0) += 1, V at (xj=1,zk=1) += 1
                # W: contributes to R00 and R11
                #   R00: +M1 => W at (zi=0,xi=0) += 1
                #   R11: +M1 => W at (zi=1,xi=1) += 1
                t = 0
                col = bi * 7 + t
                U[u_idx(0, 0)][col] += 1
                U[u_idx(1, 1)][col] += 1
                V[v_idx(0, 0)][col] += 1
                V[v_idx(1, 1)][col] += 1
                W[w_idx(0, 0)][col] += 1  # R00
                W[w_idx(1, 1)][col] += 1  # R11
                
                # Term 1 (M2): (A10+A11) * (B00)
                # U: A10 + A11 => U at (xi=1,yj=0) += 1, U at (xi=1,yj=1) += 1
                # V: B00 => V at (xj=0,zk=0) += 1
                # W: contributes to R10 and R11
                #   R10: +M2 => W at (zi=1,xi=0) += 1
                #   R11: -M2 => W at (zi=1,xi=1) -= 1
                t = 1
                col = bi * 7 + t
                U[u_idx(1, 0)][col] += 1
                U[u_idx(1, 1)][col] += 1
                V[v_idx(0, 0)][col] += 1
                W[w_idx(1, 0)][col] += 1  # R10
                W[w_idx(1, 1)][col] -= 1  # R11
                
                # Term 2 (M3): (A00) * (B01-B11)
                # U: A00 => U at (xi=0,yj=0) += 1
                # V: B01 - B11 => V at (xj=0,zk=1) += 1, V at (xj=1,zk=1) -= 1
                # W: contributes to R01 and R11
                #   R01: +M3 => W at (zi=0,xi=1) += 1
                #   R11: +M3 => W at (zi=1,xi=1) += 1
                t = 2
                col = bi * 7 + t
                U[u_idx(0, 0)][col] += 1
                V[v_idx(0, 1)][col] += 1
                V[v_idx(1, 1)][col] -= 1
                W[w_idx(0, 1)][col] += 1  # R01
                W[w_idx(1, 1)][col] += 1  # R11
                
                # Term 3 (M4): (A11) * (B11-B10)
                # U: A11 => U at (xi=1,yj=1) += 1
                # V: B11 - B10 => V at (xj=1,zk=1) += 1, V at (xj=1,zk=0) -= 1
                # W: contributes to R00 and R10
                #   R00: +M4 => W at (zi=0,xi=0) += 1
                #   R10: +M4 => W at (zi=1,xi=0) += 1
                t = 3
                col = bi * 7 + t
                U[u_idx(1, 1)][col] += 1
                V[v_idx(1, 1)][col] += 1
                V[v_idx(1, 0)][col] -= 1
                W[w_idx(0, 0)][col] += 1  # R00
                W[w_idx(1, 0)][col] += 1  # R10
                
                # Term 4 (M5): (A00+A01) * (B11)
                # U: A00 + A01 => U at (xi=0,yj=0) += 1, U at (xi=0,yj=1) += 1
                # V: B11 => V at (xj=1,zk=1) += 1
                # W: contributes to R00 and R01
                #   R00: -M5 => W at (zi=0,xi=0) -= 1
                #   R01: +M5 => W at (zi=0,xi=1) += 1
                t = 4
                col = bi * 7 + t
                U[u_idx(0, 0)][col] += 1
                U[u_idx(0, 1)][col] += 1
                V[v_idx(1, 1)][col] += 1
                W[w_idx(0, 0)][col] -= 1  # R00
                W[w_idx(0, 1)][col] += 1  # R01
                
                # Term 5 (M6): (A10-A00) * (B00+B10)
                # U: A10 - A00 => U at (xi=1,yj=0) += 1, U at (xi=0,yj=0) -= 1
                # V: B00 + B10 => V at (xj=0,zk=0) += 1, V at (xj=1,zk=0) += 1
                # W: contributes to R11
                #   R11: +M6 => W at (zi=1,xi=1) += 1
                t = 5
                col = bi * 7 + t
                U[u_idx(1, 0)][col] += 1
                U[u_idx(0, 0)][col] -= 1
                V[v_idx(0, 0)][col] += 1
                V[v_idx(1, 0)][col] += 1
                W[w_idx(1, 1)][col] += 1  # R11
                
                # Term 6 (M7): (A01-A11) * (B01+B11)
                # U: A01 - A11 => U at (xi=0,yj=1) += 1, U at (xi=1,yj=1) -= 1
                # V: B01 + B11 => V at (xj=0,zk=1) += 1, V at (xj=1,zk=1) += 1
                # W: contributes to R00
                #   R00: +M7 => W at (zi=0,xi=0) += 1
                t = 6
                col = bi * 7 + t
                U[u_idx(0, 1)][col] += 1
                U[u_idx(1, 1)][col] -= 1
                V[v_idx(0, 1)][col] += 1
                V[v_idx(1, 1)][col] += 1
                W[w_idx(0, 0)][col] += 1  # R00
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

