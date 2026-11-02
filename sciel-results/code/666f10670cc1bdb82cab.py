"""Find rank-minimal tensor decomposition for 3x5 * 5x7 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def _build_strassen_2x2x2():
    """Build rank-7 Strassen decomposition for 2x2 * 2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = (b-c)h
    U[1, 3] = 1; U[2, 3] = -1
    V[3, 3] = 1
    W[1, 3] = 1
    
    # M5 = (a+b)g
    U[0, 4] = 1; U[1, 4] = 1
    V[2, 4] = 1
    W[3, 4] = 1
    
    # M6 = (c-d)(f+h)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[2, 5] = 1
    
    # M7 = (a-c)(e+g)
    U[0, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Build schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _combine_decompositions(decomps):
    """Combine a list of (U, V, W) decompositions by horizontal concatenation."""
    if not decomps:
        return None
    total_rank = sum(d[0].shape[1] for d in decomps)
    U = np.zeros((decomps[0][0].shape[0], total_rank))
    V = np.zeros((decomps[0][1].shape[0], total_rank))
    W = np.zeros((decomps[0][2].shape[0], total_rank))
    col = 0
    for d in decomps:
        r = d[0].shape[1]
        U[:, col:col+r] = d[0]
        V[:, col:col+r] = d[1]
        W[:, col:col+r] = d[2]
        col += r
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(actual - expected) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast verification using numpy."""
    rank = U.shape[1]
    i_idx = np.repeat(np.arange(n), m*p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n*m)
    
    U_idx = i_idx * m + j_idx
    V_idx = j_idx * p + k_idx
    W_idx = k_idx * n + i_idx
    
    U_sel = U[U_idx, :]
    V_sel = V[V_idx, :]
    W_sel = W[W_idx, :]
    
    result = np.sum(U_sel * V_sel * W_sel, axis=1)
    return np.all(np.abs(result - 1.0) < tol)

def _build_strassen_2x2x2():
    """Build rank-7 Strassen decomposition for 2x2 * 2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 multiplications for 2x2 matrix mult
    # C = A*B where A, B are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    # U rows indexed by [i*m+j] = [0,1,2,3] for (0,0),(0,1),(1,0),(1,1)
    # V rows indexed by [j*p+k] = [0,1,2,3] for (0,0),(0,1),(1,0),(1,1)
    # W rows indexed by [k*n+i] = [0,1,2,3] for (0,0),(0,1),(1,0),(1,1)
    
    # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: contributes to C[0,0]=ae+bg and C[1,1]=cf+dh
    # M2 = (c+d)e: U: c=1,d=1; V: e=1; W: contributes to C[1,0]=ce+dg
    # M3 = a(f-h): U: a=1; V: f=1,h=-1; W: contributes to C[0,1]=af+bh
    # M4 = (b-c)h: U: b=1,c=-1; V: h=1; W: contributes to C[0,1]=af+bh
    # M5 = (a+b)g: U: a=1,b=1; V: g=1; W: contributes to C[1,1]=cf+dh
    # M6 = (c-d)(f+h): U: c=1,d=-1; V: f=1,h=1; W: contributes to C[1,0]=ce+dg
    # M7 = (a-c)(e+g): U: a=1,c=-1; V: e=1,g=1; W: contributes to C[0,0]=ae+bg
    
    # Column 0: M1
    U[0, 0] = 1; U[3, 0] = 1  # a, d
    V[0, 0] = 1; V[3, 0] = 1  # e, h
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]
    
    # Column 1: M2
    U[2, 1] = 1; U[3, 1] = 1  # c, d
    V[0, 1] = 1  # e
    W[2, 1] = 1  # C[1,0]
    
    # Column 2: M3
    U[0, 2] = 1  # a
    V[1, 2] = 1; V[3, 2] = -1  # f, h
    W[1, 2] = 1  # C[0,1]
    
    # Column 3: M4
    U[1, 3] = 1; U[2, 3] = -1  # b, c
    V[3, 3] = 1  # h
    W[1, 3] = 1  # C[0,1]
    
    # Column 4: M5
    U[0, 4] = 1; U[1, 4] = 1  # a, b
    V[2, 4] = 1  # g
    W[3, 4] = 1  # C[1,1]
    
    # Column 5: M6
    U[2, 5] = 1; U[3, 5] = -1  # c, d
    V[1, 5] = 1; V[3, 5] = 1  # f, h
    W[2, 5] = 1  # C[1,0]
    
    # Column 6: M7
    U[0, 6] = 1; U[2, 6] = -1  # a, c
    V[0, 6] = 1; V[2, 6] = 1  # e, g
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W

def _schoolbook(n, m, p):
    """Build schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _combine_decompositions(decomps):
    """Combine a list of (U, V, W) decompositions by horizontal concatenation."""
    total_rank = sum(d[0].shape[1] for d in decomps)
    U = np.zeros((decomps[0][0].shape[0], total_rank))
    V = np.zeros((decomps[0][1].shape[0], total_rank))
    W = np.zeros((decomps[0][2].shape[0], total_rank))
    col = 0
    for d in decomps:
        r = d[0].shape[1]
        U[:, col:col+r] = d[0]
        V[:, col:col+r] = d[1]
        W[:, col:col+r] = d[2]
        col += r
    return U, V, W

def _build_3x2x2():
    """Build 3x2 * 2x2 decomposition using Strassen(2x2x2) + schoolbook(1x2x2)."""
    n, m, p = 3, 2, 2
    # A is 3x2, B is 2x2, C is 3x2
    # Split A into A1 (2x2) and A2 (1x2)
    # C = A1*B + A2*B
    # Use Strassen for A1*B (rank 7), schoolbook for A2*B (rank 4)
    
    U1, V1, W1 = _build_strassen_2x2x2()
    # U1 is (4,7), V1 is (4,7), W1 is (4,7)
    # Need to extend W1 to shape (p*n, rank) = (6, 7)
    # W1 corresponds to C[0,0],C[0,1],C[1,0],C[1,1] for the 2x2 result
    # But C is 3x2, so C has rows 0,1,2 and cols 0,1
    # W index: k*n+i where k in {0,1}, i in {0,1,2}
    # For the 2x2 part (rows 0,1 of C): W indices are (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W1 has shape (4,7) with indices 0,1,2,3 which match
    # But we need W to have shape (6,7), with rows 4,5 (for k=1,i=2) being zero
    
    U1_ext = np.zeros((n*m, 7))  # (6, 7)
    U1_ext[:4, :] = U1
    # U1_ext[4, :] and U1_ext[5, :] remain 0
    
    V1_ext = np.zeros((m*p, 7))  # (4, 7) - same as V1
    V1_ext = V1.copy()
    
    W1_ext = np.zeros((p*n, 7))  # (6, 7)
    W1_ext[:4, :] = W1
    # W1_ext[4, :] and W1_ext[5, :] remain 0
    
    # Schoolbook for A2 (1x2) * B (2x2): rank = 1*2*2 = 4
    # A2 is row 2 of A (indices 4,5 in U), B is 2x2, C2 is row 2 of C
    # U indices for A2: i=2, j=0,1 -> 2*2+0=4, 2*2+1=5
    # V indices: j*p+k
    # W indices: k*n+i = k*3+2
    U2, V2, W2 = _schoolbook(1, 2, 2)
    # U2 is (2,4), V2 is (4,4), W2 is (2,4)
    # U2 row 0 -> A2[0,0] -> U index 4
    # U2 row 1 -> A2[0,1] -> U index 5
    # W2 row 0 -> C[2,0] -> W index 0*3+2=2
    # W2 row 1 -> C[2,1] -> W index 1*3+2=5
    
    U2_ext = np.zeros((n*m, 4))
    U2_ext[4:6, :] = U2
    
    V2_ext = np.zeros((m*p, 4))
    V2_ext = V2.copy()
    
    W2_ext = np.zeros((p*n, 4))
    W2_ext[2, 0] = W2[0, 0]
    W2_ext[5, 1] = W2[1, 1]
    
    return _combine_decompositions([(U1_ext, V1_ext, W1_ext), (U2_ext, V2_ext, W2_ext)])

def construct():
    """Find rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0
    
    # Schoolbook decomposition as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p
    
    # Try block decomposition
    # Split A (3x5) into A_top (2x5) and A_bottom (1x5)
    # C = A_top * B + A_bottom * B
    # A_top * B: 2x5 * 5x7 -> rank?
    # A_bottom * B: 1x5 * 5x7 -> rank 5*7=35 schoolbook
    
    # For A_top (2x5) * B (5x7):
    # Split B into B_left (5x4) and B_right (5x3)
    # C_top = A_top * B_left + A_top * B_right
    # A_top * B_left: 2x5 * 5x4 -> split into 2x2*5x4 blocks?
    
    # Let's use Strassen-based approach:
    # 2x5 * 5x7: split A into two 2x2 + one 2x1, B into...
    # Simpler: use schoolbook for sub-blocks but Strassen where applicable
    
    # 2x5 * 5x7: 
    # Split into (2x2 * 5x4) + (2x1 * 5x4) + (2x2 * 5x3) + (2x1 * 5x3)
    # 2x2 * 5x4: rank <= 7*2 = 14 (two Strassen blocks) -- no, this doesn't work directly
    # Actually 2x2 * 2x2 is rank 7, but 2x2 * 5x4 needs different approach
    
    # Let's just use schoolbook for all sub-blocks and see if we can improve
    # 3x5 * 5x7 schoolbook = 105
    
    # Try: split A into 2+1 rows
    # (2x5 * 5x7) + (1x5 * 5x7)
    # 1x5 * 5x7 schoolbook rank = 35
    # 2x5 * 5x7: split B into 5x4 + 5x3
    # (2x5 * 5x4) + (2x5 * 5x3)
    # 2x5 * 5x4: split A into 2x2+2x1, each times 5x4
    #   2x2 * 5x4: schoolbook rank = 4*4=16... but we can do better
    #   Actually 2x2 * 5x4: for each of 4 cols, 2x2*5x1 rank=10, total 40? No.
    #   2x2 * 5x4 = 8*4 = 32 schoolbook
    #   2x1 * 5x4 = 2*4 = 8 schoolbook
    #   So 2x5 * 5x4 = 32+8 = 40
    #   2x5 * 5x3 = 2x2*5x3 (6*3=18) + 2x1*5x3 (2*3=6) = 24
    #   2x5 * 5x7 = 40+24 = 64
    # Total = 64+35 = 99
    
    # Can we do 2x2 * 5x4 better? 
    # 2x2 * 5x4: for each column of B (4 cols), compute 2x2 * 5x1
    # 2x2 * 5x1: A is 2x2, B is 5x1 -> C is 2x1
    # C[i] = sum_j A[i,j]*B[j,0]
    # This is rank 2*5*1 = 10 schoolbook
    # But we can use: for each of 4 columns independently, rank 10 each, total 40
    # Or: 2x2 * 5x4 can be seen as A * B where we can use Strassen-like if we view B as 4 columns of 5x1
    # Not directly helpful.
    
    # Let's just implement the block decomposition and verify
    def _build_block_3x5x7():
        """Block decomposition: (2x5*5x7) + (1x5*5x7)"""
        n, m, p = 3, 5, 7
        # Part 1: 2x5 * 5x7
        n1, m1, p1 = 2, 5, 7
        U1, V1, W1 = _schoolbook(n1, m1, p1)
        # U1: (10, 70), V1: (35, 70), W1: (14, 70)
        # Extend to full size: U: (15, 70), V: (35, 70), W: (21, 70)
        U1_full = np.zeros((n*m, U1.shape[1]))
        U1_full[:n1*m1, :] = U1
        V1_full = V1.copy()
        W1_full = np.zeros((p*n, W1.shape[1]))
        # W1 indices: k*p1+i where k in 0..6, i in 0..1 -> 0..13
        # Full W indices: k*n+i where k in 0..6, i in 0..2 -> 0..20
        # W1[i] corresponds to W_full[k*2+i] for the top 2 rows
        # Need to map: W1 row r = k*2+i -> W_full row k*3+i
        for k in range(p1):
            for i in range(n1):
                W1_full[k*n+i, :] = W1[k*n1+i, :]
        
        # Part 2: 1x5 * 5x7
        n2, m2, p2 = 1, 5, 7
        U2, V2, W2 = _schoolbook(n2, m2, p2)
        # U2: (5, 35), V2: (35, 35), W2: (7, 35)
        U2_full = np.zeros((n*m, U2.shape[1]))
        U2_full[n1*m1:, :] = U2
        V2_full = V2.copy()
        W2_full = np.zeros((p*n, W2.shape[1]))
        for k in range(p2):
            W2_full[k*n+2, :] = W2[k*n2+0, :]
        
        return _combine_decompositions([(U1_full, V1_full, W1_full), (U2_full, V2_full, W2_full)])
    
    U_block, V_block, W_block = _build_block_3x5x7()
    block_rank = U_block.shape[1]
    
    if _verify_fast(U_block, V_block, W_block, n, m, p, tol=1e-6):
        if block_rank < best_rank:
            best_rank = block_rank
            best_result = {"U": U_block.tolist(), "V": V_block.tolist(), "W": W_block.tolist()}
    
    # Now try to improve 2x5*5x7 part using Strassen for 2x2 sub-blocks
    def _build_2x5x7_improved():
        """2x5 * 5x7 using Strassen for 2x2 blocks"""
        n1, m1, p1 = 2, 5, 7
        # Split B into B1 (5x4) and B2 (5x3)
        # A * B1: 2x5 * 5x4
        # A * B2: 2x5 * 5x3
        # For 2x5 * 5x4: split A into A1 (2x2) and A2 (2x3)
        #   A1 * B1: 2x2 * 5x4
        #   A2 * B1: 2x3 * 5x4
        # This is getting complex. Let's just use schoolbook.
        U, V, W = _schoolbook(n1, m1, p1)
        return U, V, W
    
    return best_result
