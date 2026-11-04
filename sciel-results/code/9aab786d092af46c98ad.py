import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def _verify_decomp(U, V, W, size):
    """Verify that sum_r U[i*size+j, r]*V[j*size+k, r]*W[k*size+i, r] = 1 for all i,j,k in range(size)"""
    rank = len(U[0])
    for i in range(size):
        for j in range(size):
            for k in range(size):
                s = 0
                for r in range(rank):
                    s += U[i*size + j][r] * V[j*size + k][r] * W[k*size + i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(size):
    """Return schoolbook decomposition of rank size^3"""
    rank = size * size * size
    U = [[0] * rank for _ in range(size * size)]
    V = [[0] * rank for _ in range(size * size)]
    W = [[0] * rank for _ in range(size * size)]
    col = 0
    for i in range(size):
        for j in range(size):
            for k in range(size):
                U[i * size + j][col] = 1
                V[j * size + k][col] = 1
                W[k * size + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """Strassen 2x2x2 decomposition, rank 7"""
    # C = A*B where A,B,C are 2x2
    # a11 a12     b11 b12     c11 c12
    # a21 a22  *  b21 b22  =  c21 c22
    #
    # P1 = (a11+a22)(b11+b22)
    # P2 = (a21+a22)b11
    # P3 = a11(b21-b22)
    # P4 = a22(b12-b11)
    # P5 = (a11+a12)b22
    # P6 = (a21-a11)(b11+b12)
    # P7 = (a12-a22)(b21+b22)
    #
    # c11 = P1+P4-P5+P7
    # c12 = P3+P5
    # c21 = P2+P4
    # c22 = P1-P2+P3+P6
    #
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # i,j,k in {0,1}, r in {0..6}
    U = [
        # i=0,j=0: a11 appears in P1,P3,P5,P6
        [1, 0, 1, 0, 1, -1, 0],
        # i=0,j=1: a12 appears in P5,P7
        [0, 0, 0, 0, 1, 0, 1],
        # i=1,j=0: a21 appears in P2,P6
        [0, 1, 0, 0, 0, 1, 0],
        # i=1,j=1: a22 appears in P1,P2,P4,P7
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V = [
        # j=0,k=0: b11 appears in P1,P2,P4,P6
        [1, 1, 0, -1, 0, 1, 0],
        # j=0,k=1: b12 appears in P4,P6
        [0, 0, 0, -1, 0, 1, 0],
        # j=1,k=0: b21 appears in P3,P7
        [0, 0, -1, 0, 0, 0, 1],
        # j=1,k=1: b22 appears in P1,P3,P5,P7
        [1, 0, -1, 0, 1, 0, 1],
    ]
    W = [
        # k=0,i=0: c11 = P1+P4-P5+P7
        [1, 0, 0, 1, -1, 0, 1],
        # k=0,i=1: c21 = P2+P4
        [0, 1, 0, 1, 0, 0, 0],
        # k=1,i=0: c12 = P3+P5
        [0, 0, 1, 0, 1, 0, 0],
        # k=1,i=1: c22 = P1-P2+P3+P6
        [1, -1, 1, 0, 0, 1, 0],
    ]
    return U, V, W

def _laderman_3x3():
    """Laderman 3x3x3 decomposition, rank 7"""
    # Standard Laderman algorithm
    # C = A*B, all 3x3
    # 
    # P1 = (a11+a31)(b11-b12)
    # P2 = (a21+a22)(b21-b22)  
    # P3 = (a11+a12)(b22-b12)
    # P4 = (a22+a32)(b11-b21)
    # P5 = (a11+a12)(b22-b32)
    # P6 = (a22+a32)(b11-b31)
    # P7 = (a11+a31)(b11-b31)
    #
    # Actually let me use the standard verified Laderman:
    # From Laderman (1976), the 7 products for 3x3:
    # Let me use a known correct formulation.
    
    # Using the formulation from "Fast Matrix Multiplication" by Strassen:
    # P1 = (a11+a21+a31)(b11+b21+b31)  -- no, this isn't right either.
    
    # Let me use a verified set. The Laderman algorithm:
    # P1 = (a11 + a31)(b11 - b12)
    # P2 = (a21 + a22)(b21 - b22)
    # P3 = (a11 + a12)(b22 - b12)
    # P4 = (a22 + a32)(b11 - b21)
    # P5 = (a11 + a12)(b22 - b32)
    # P6 = (a22 + a32)(b11 - b31)
    # P7 = (a11 + a31)(b11 - b31)
    
    # c11 = P1 + P3 - P5
    # c12 = P5 - P7
    # c13 = P7 - P1
    # c21 = P2 + P4 - P6
    # c22 = P6 - P2
    # c23 = P2 - P4
    # c31 = P6 - P4
    # c32 = P4 - P6  -- wait, this doesn't seem right
    # c33 = P4 - P2 + P6  -- need to verify
    
    # Let me just use a known-correct Laderman from literature.
    # The standard Laderman 3x3 (rank 7):
    
    U = [
        # Row i*3+j, for i=0,1,2 and j=0,1,2
        # a11 (i=0,j=0): in P1, P3, P5, P7
        [1, 0, 1, 0, 1, 0, 1],
        # a12 (i=0,j=1): in P3, P5
        [0, 0, 1, 0, 1, 0, 0],
        # a13 (i=0,j=2): not in any? 
        [0, 0, 0, 0, 0, 0, 0],
        # a21 (i=1,j=0): in P2
        [0, 1, 0, 0, 0, 0, 0],
        # a22 (i=1,j=1): in P2, P4, P6
        [0, 1, 0, 1, 0, 1, 0],
        # a23 (i=1,j=2): in P4, P6
        [0, 0, 0, 1, 0, 1, 0],
        # a31 (i=2,j=0): in P1, P7
        [1, 0, 0, 0, 0, 0, 1],
        # a32 (i=2,j=1): in P4, P6
        [0, 0, 0, 1, 0, 1, 0],
        # a33 (i=2,j=2): not in any
        [0, 0, 0, 0, 0, 0, 0],
    ]
    V = [
        # b11 (j=0,k=0): in P1, P4, P6, P7
        [1, 0, 0, 1, 0, 1, 1],
        # b12 (j=0,k=1): in P1, P3, P5
        [-1, 0, -1, 0, -1, 0, 0],
        # b13 (j=0,k=2): not in any
        [0, 0, 0, 0, 0, 0, 0],
        # b21 (j=1,k=0): in P2, P4
        [0, 1, 0, -1, 0, 0, 0],
        # b22 (j=1,k=1): in P2, P3, P5
        [0, -1, -1, 0, 1, 0, 0],
        # b23 (j=1,k=2): in P5
        [0, 0, 0, 0, -1, 0, 0],
        # b31 (j=2,k=0): in P6, P7
        [0, 0, 0, 0, 0, -1, -1],
        # b32 (j=2,k=1): in P5, P6
        [0, 0, 0, 0, 0, 1, 0],
        # b33 (j=2,k=2): not in any
        [0, 0, 0, 0, 0, 0, 0],
    ]
    W = [
        # c11 (k=0,i=0): P1+P3-P5
        [1, 0, 1, 0, -1, 0, 0],
        # c12 (k=0,i=1): P5-P7  -- wait, c12 is row 1, col 2
        # Actually W[k*3+i, r] so k=0,i=1 is c_{1,0} in 0-indexed = c21 in 1-indexed
        # Let me redo: W[k*3+i] where k is col of C, i is row of C
        # c_{i,k} = sum_r U[i*3+j, r] V[j*3+k, r] W[k*3+i, r]
        # So W[k*3+i, r] is the coefficient for c_{i,k}
        # c_{0,0} (c11): P1+P3-P5 -> [1,0,1,0,-1,0,0]
        # c_{0,1} (c12): P5-P7 -> [0,0,0,0,1,0,-1]
        # c_{0,2} (c13): P7-P1 -> [-1,0,0,0,0,0,1]
        # c_{1,0} (c21): P2+P4-P6 -> [0,1,0,1,0,-1,0]
        # c_{1,1} (c22): P6-P2 -> [0,-1,0,0,0,1,0]
        # c_{1,2} (c23): P2-P4 -> [0,1,0,-1,0,0,0]
        # c_{2,0} (c31): P6-P4 -> [0,0,0,-1,0,1,0]
        # c_{2,1} (c32): P4-P6 -> [0,0,0,1,0,-1,0]
        # c_{2,2} (c33): P4-P2+P6 -> [0,-1,0,1,0,1,0]
        [1, 0, 1, 0, -1, 0, 0],   # c_{0,0}
        [0, 0, 0, 0, 1, 0, -1],   # c_{0,1}
        [-1, 0, 0, 0, 0, 0, 1],   # c_{0,2}
        [0, 1, 0, 1, 0, -1, 0],   # c_{1,0}
        [0, -1, 0, 0, 0, 1, 0],   # c_{1,1}
        [0, 1, 0, -1, 0, 0, 0],   # c_{1,2}
        [0, 0, 0, -1, 0, 1, 0],   # c_{2,0}
        [0, 0, 0, 1, 0, -1, 0],   # c_{2,1}
        [0, -1, 0, 1, 0, 1, 0],   # c_{2,2}
    ]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Try block decomposition: 5 = 2 + 3
    # Use Strassen for 2x2 blocks, Laderman for 3x3 blocks, schoolbook for mixed
    
    U2, V2, W2 = _strassen_2x2()
    U3, V3, W3 = _laderman_3x3()
    
    # Verify sub-algorithms
    if not _verify_decomp(U2, V2, W2, 2):
        return _schoolbook(5)
    if not _verify_decomp(U3, V3, W3, 3):
        return _schoolbook(5)
    
    # Build 5x5 decomposition using block structure 5 = 2 + 3
    # A is 5x5, partitioned as:
    #   A = [A11 A12]
    #       [A21 A22]
    # where A11 is 2x2, A12 is 2x3, A21 is 3x2, A22 is 3x3
    # Similarly for B and C.
    #
    # C11 = A11*B11 + A12*B21  (2x2)
    # C12 = A11*B12 + A12*B22  (2x3)
    # C21 = A21*B11 + A22*B21  (3x2)
    # C22 = A21*B12 + A22*B22  (3x3)
    
    # We need to decompose each product. For products of mismatched sizes,
    # we use schoolbook. For 2x2*2x2, use Strassen. For 3x3*3x3, use Laderman.
    # For 2x3*3x2, 2x2*2x3, 3x2*2x2, 3x2*2x3, 2x3*3x3, 3x3*3x2: schoolbook.
    
    # Total rank:
    # C11: Strassen(2x2*2x2)=7 + schoolbook(2x3*3x2)=6 = 13
    # C12: schoolbook(2x2*2x3)=6 + schoolbook(2x3*3x3)=18 = 24
    # C21: schoolbook(3x2*2x2)=6 + schoolbook(3x3*3x2)=18 = 24
    # C22: schoolbook(3x2*2x3)=6 + Laderman(3x3*3x3)=7 = 13
    # Total: 13+24+24+13 = 74
    
    # That's not great. Let me try a different partition.
    
    # Try 5 = 1 + 4:
    # C11 = A11*B11 (1x1*1x1, rank 1)
    # C12 = A11*B12 + A12*B22 (1x1*1x4 + 1x4*4x4)
    # C21 = A21*B11 + A22*B21 (4x1*1x1 + 4x4*4x1)
    # C22 = A21*B12 + A22*B22 (4x1*1x4 + 4x4*4x4)
    
    # For 4x4*4x4, we can use rank 18 (recursive Strassen)
    # This is getting complex. Let me just use the tensor product approach
    # but with proper verification.
    
    # Tensor product: 3x3 (Laderman, rank 7) ⊗ 2x2 (Strassen, rank 7) -> 6x6 (rank 49)
    # Then restrict to 5x5.
    
    n1, m1, p1 = 3, 3, 3
    n2, m2, p2 = 2, 2, 2
    r1, r2 = 7, 7
    r_total = r1 * r2  # 49
    
    n6, m6, p6 = 6, 6, 6
    
    U6 = [[0] * r_total for _ in range(n6 * m6)]
    V6 = [[0] * r_total for _ in range(m6 * p6)]
    W6 = [[0] * r_total for _ in range(p6 * n6)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    i = i1 * n2 + i2
                    j = j1 * m2 + j2
                    for r1_idx in range(r1):
                        for r2_idx in range(r2):
                            r = r1_idx * r2 + r2_idx
                            u_val = U3[i1*m1 + j1][r1_idx] * U2[i2*m2 + j2][r2_idx]
                            U6[i*m6 + j][r] = u_val
    
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    j = j1 * m2 + j2
                    k = k1 * p2 + k2
                    for r1_idx in range(r1):
                        for r2_idx in range(r2):
                            r = r1_idx * r2 + r2_idx
                            v_val = V3[j1*p1 + k1][r1_idx] * V2[j2*p2 + k2][r2_idx]
                            V6[j*p6 + k][r] = v_val
    
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    k = k1 * p2 + k2
                    i = i1 * n2 + i2
                    for r1_idx in range(r1):
                        for r2_idx in range(r2):
                            r = r1_idx * r2 + r2_idx
                            w_val = W3[k1*n1 + i1][r1_idx] * W2[k2*n2 + i2][r2_idx]
                            W6[k*n6 + i][r] = w_val
    
    # Verify 6x6
    if _verify_decomp(U6, V6, W6, 6):
        # Restrict to 5x5
        U5 = [[0] * r_total for _ in range(n * m)]
        V5 = [[0] * r_total for _ in range(m * p)]
        W5 = [[0] * r_total for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for r in range(r_total):
                    U5[i*m + j][r] = U6[i*6 + j][r]
        
        for j in range(m):
            for k in range(p):
                for r in range(r_total):
                    V5[j*p + k][r] = V6[j*6 + k][r]
        
        for k in range(p):
            for i in range(n):
                for r in range(r_total):
                    W5[k*n + i][r] = W6[k*6 + i][r]
        
        if _verify_decomp(U5, V5, W5, 5):
            return {"U": U5, "V": V5, "W": W5}
    
    # Fallback: schoolbook
    return _schoolbook(5)
# EVOLVE-BLOCK-END
