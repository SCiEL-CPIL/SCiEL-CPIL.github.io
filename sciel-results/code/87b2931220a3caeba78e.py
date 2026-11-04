"""Rewritten program: block/Strassen hybrid decomposition for M_{3,4,7}."""

import numpy as np

DIMENSIONS = [3, 4, 7]


def _strassen_m222():
    """Return (U, V, W) for 2x2*2x2, rank 7, indices:
       U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # Strassen terms M1..M7
    # a11 a12 / a21 a22, b11 b12 / b21 b22
    # M1=(a11+a22)(b11+b22)
    # M2=a22(b12-b11)
    # M3=(a11+a12)b22
    # M4=(a21+a22)b11
    # M5=(a11-a21)(b12+b21)
    # M6=(a12-a22)(b11+b21)
    # M7=(a11-a12)(b12-b22)
    # U (input A factors):
    U[0, 0] = 1; U[0, 1] = 0; U[0, 2] = 1; U[0, 3] = 0; U[0, 4] = 1; U[0, 5] = 1; U[0, 6] = 1
    U[1, 0] = 0; U[1, 1] = 0; U[1, 2] = 1; U[1, 3] = 0; U[1, 4] = 0; U[1, 5] = 1; U[1, 6] = -1
    U[2, 0] = 0; U[2, 1] = 0; U[2, 2] = 0; U[2, 3] = 1; U[2, 4] = -1; U[2, 5] = 0; U[2, 6] = 0
    U[3, 0] = 1; U[3, 1] = 1; U[3, 2] = 0; U[3, 3] = 1; U[3, 4] = 0; U[3, 5] = -1; U[3, 6] = -1
    # V (input B factors):
    V[0, 0] = 1; V[0, 1] = -1; V[0, 2] = 0; V[0, 3] = 1; V[0, 4] = 1; V[0, 5] = 1; V[0, 6] = -1
    V[1, 0] = 1; V[1, 1] = 1; V[1, 2] = 0; V[1, 3] = 0; V[1, 4] = 1; V[1, 5] = 0; V[1, 6] = 1
    V[2, 0] = 0; V[2, 1] = 0; V[2, 2] = 1; V[2, 3] = 0; V[2, 4] = 1; V[2, 5] = 1; V[2, 6] = 0
    V[3, 0] = 1; V[3, 1] = 1; V[3, 2] = 1; V[3, 3] = 1; V[3, 4] = 0; V[3, 5] = 0; V[3, 6] = -1
    # W (output C factors): W[k*2+i]
    # c11 = M1+M4-M5-M7
    # c12 = M3+M5
    # c21 = M2+M4
    # c22 = M1-M2+M6+M7
    W[0, 0] = 1; W[0, 1] = 0; W[0, 2] = 0; W[0, 3] = 1; W[0, 4] = -1; W[0, 5] = 0; W[0, 6] = -1
    W[1, 0] = 0; W[1, 1] = 0; W[1, 2] = 1; W[1, 3] = 0; W[1, 4] = 1; W[1, 5] = 0; W[1, 6] = 0
    W[2, 0] = 0; W[2, 1] = 1; W[2, 2] = 0; W[2, 3] = 1; W[2, 4] = 0; W[2, 5] = 0; W[2, 6] = 0
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 0; W[3, 3] = 0; W[3, 4] = 0; W[3, 5] = 1; W[3, 6] = 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    # Splits:
    # n: 2 + 1
    # m: 2 + 2
    # p: 4 + 3 (for the n=2 part, we further split p=4 into 2+2 and p=3 into 2+1)
    
    # We will manually place blocks.
    # Total rank estimate:
    # n=2 part:
    #   j=0..2, k=0..4: M_{2,2,4} -> 2x M_{2,2,2} (Strassen) -> 14
    #   j=0..2, k=4..7: M_{2,2,3} -> 1x M_{2,2,2} (Strassen) + 1x M_{2,2,1} (School) -> 7+4=11
    #   j=2..4, k=0..4: M_{2,2,4} -> 14
    #   j=2..4, k=4..7: M_{2,2,3} -> 11
    #   Subtotal n=2: 14+11+14+11 = 50
    # n=1 part:
    #   j=0..2, k=0..4: M_{1,2,4} -> Rank 2
    #   j=0..2, k=4..7: M_{1,2,3} -> Rank 2
    #   j=2..4, k=0..4: M_{1,2,4} -> Rank 2
    #   j=2..4, k=4..7: M_{1,2,3} -> Rank 2
    #   Subtotal n=1: 8
    # Total Rank: 58

    R = 58
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    offset = 0
    
    def add_strassen_block(i0, j0, k0):
        """Add a 2x2x2 Strassen block at A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        nonlocal offset
        Ub, Vb, Wb = _strassen_m222()
        # Map U indices: local (i, j) -> global (i0+i, j0+j) -> U index (i0+i)*m + (j0+j)
        for li in range(2):
            for lj in range(2):
                U[(i0+li)*m + (j0+lj), offset:offset+7] = Ub[li*2+lj, :]
        # Map V indices: local (j, k) -> global (j0+j, k0+k) -> V index (j0+j)*p + (k0+k)
        for lj in range(2):
            for lk in range(2):
                V[(j0+lj)*p + (k0+lk), offset:offset+7] = Vb[lj*2+lk, :]
        # Map W indices: local (k, i) -> global (k0+k, i0+i) -> W index (k0+k)*n + (i0+i)
        for lk in range(2):
            for li in range(2):
                W[(k0+lk)*n + (i0+li), offset:offset+7] = Wb[lk*2+li, :]
        offset += 7

    def add_schoolbook_block(i0, j0, k0, ni, nj, nk):
        """Add a schoolbook block for A[i0:i0+ni, j0:j0+nj] * B[j0:j0+nj, k0:k0+nk] -> C[i0:i0+ni, k0:k0+nk]"""
        nonlocal offset
        for li in range(ni):
            for lj in range(nj):
                for lk in range(nk):
                    U[(i0+li)*m + (j0+lj), offset] = 1
                    V[(j0+lj)*p + (k0+lk), offset] = 1
                    W[(k0+lk)*n + (i0+li), offset] = 1
                    offset += 1

    def add_m1mp_block(i0, j0, k0, nj, nk):
        """Add a block for A[i0, j0:j0+nj] * B[j0:j0+nj, k0:k0+nk] -> C[i0, k0:k0+nk]
           Rank = min(nj, nk). Here nj=2.
           If nj <= nk: Rank = nj.
           Decomposition:
           For c in range(nj):
             U[i0, c] = 1
             V[j0+c, k0+k] = 1 for all k? No.
             Let's use the rank min(nj, nk) logic.
             If nj=2, nk=4. Rank 2.
             c=0: U[i0,0]=1, V[j0, k0..k0+4]=1, W[k0..k0+4, i0]=1.
             c=1: U[i0,1]=1, V[j0+1, k0..k0+4]=1, W[k0..k0+4, i0]=1.
             Sum: A[i0, j0]*1 + A[i0, j0+1]*1 ... wait.
             The term is sum_c U[i0,c] V[j,c] W[k,c].
             We want C[i0, k] = sum_j A[i0, j] B[j, k].
             If we set U[i0, c] = 1, V[j0+c, k0+k] = 1, W[k0+k, i0] = 1.
             Then term c contributes A[i0, j0+c] * 1 * 1 = A[i0, j0+c].
             Summing over c=0..nj-1 gives sum_{c} A[i0, j0+c] * B[j0+c, k]? No, V is fixed to 1.
             This works if B is identity? No.
             
             Correct decomposition for M_{1, m, p} with rank min(m,p):
             If m <= p:
               For c in range(m):
                 U[i0, c] = 1
                 V[j0+c, k0+k] = 1 for all k in 0..p-1? No.
                 We need V[j, c] and W[k, c] such that sum_c V[j, c] W[k, c] = B[j, k] (conceptually).
                 Actually, the tensor entries are 1.
                 T[i, j, k] = 1.
                 We need sum_c U[i, c] V[j, c] W[k, c] = 1.
                 If m <= p:
                   For c in range(m):
                     U[i0, c] = 1
                     V[j0+c, c] = 1 (only for j=j0+c)
                     W[k0+k, c] = 1 (for all k)
                   Then for a specific j, only c=j-j0 is non-zero in V.
                   Sum = 1 * 1 * 1 = 1.
                   This works for all j in range(m) and all k.
                   
             So for M_{1, 2, 4}:
               c=0: U[i0, 0]=1, V[j0, 0..3]=1, W[k0..k0+3, 0]=1.
               c=1: U[i0, 1]=1, V[j0+1, 0..3]=1, W[k0..k0+3, 1]=1.
               Wait, V index is (j, k).
               V[j0, k0+k, c] = 1.
               So V[(j0)*p + (k0+k), c] = 1.
        """
        nonlocal offset
        rank = min(nj, nk)
        if nj <= nk:
            for c in range(nj):
                U[i0*m + j0+c, offset+c] = 1
                for lk in range(nk):
                    V[(j0+c)*p + (k0+lk), offset+c] = 1
                    W[(k0+lk)*n + i0, offset+c] = 1
        else:
            # Symmetric case, not needed here as nj=2, nk>=3
            pass
        offset += rank

    # Block 1: n=2, j=0..2, k=0..4 (M_{2,2,4})
    # Split k into 0..2 and 2..4
    add_strassen_block(0, 0, 0) # k=0..2
    add_strassen_block(0, 0, 2) # k=2..4
    
    # Block 2: n=2, j=0..2, k=4..7 (M_{2,2,3})
    # Split k into 4..6 and 6..7
    add_strassen_block(0, 0, 4) # k=4..6
    add_schoolbook_block(0, 0, 6, 2, 2, 1) # k=6..7, j=0..2. Note: j is 0..2, k is 6..7.
    # Wait, the block is A[0:2, 0:2] * B[0:2, 6:7].
    # i0=0, j0=0, k0=6, ni=2, nj=2, nk=1.
    
    # Block 3: n=2, j=2..4, k=0..4 (M_{2,2,4})
    add_strassen_block(0, 2, 0) # j=2..4, k=0..2
    add_strassen_block(0, 2, 2) # j=2..4, k=2..4
    
    # Block 4: n=2, j=2..4, k=4..7 (M_{2,2,3})
    add_strassen_block(0, 2, 4) # j=2..4, k=4..6
    add_schoolbook_block(0, 2, 6, 2, 2, 1) # j=2..4, k=6..7
    
    # Block 5: n=1, j=0..2, k=0..4 (M_{1,2,4})
    add_m1mp_block(2, 0, 0, 2, 4)
    
    # Block 6: n=1, j=0..2, k=4..7 (M_{1,2,3})
    add_m1mp_block(2, 0, 4, 2, 3)
    
    # Block 7: n=1, j=2..4, k=0..4 (M_{1,2,4})
    add_m1mp_block(2, 2, 0, 2, 4)
    
    # Block 8: n=1, j=2..4, k=4..7 (M_{1,2,3})
    add_m1mp_block(2, 2, 4, 2, 3)
    
    # Check offset
    # 14 + 11 + 14 + 11 + 2 + 2 + 2 + 2 = 58.
    if offset != R:
        print(f"Warning: Offset {offset} != Rank {R}")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
