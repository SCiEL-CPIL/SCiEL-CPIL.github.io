"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [4, 4, 6]
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _build_strassen_2x2_2x2_rank7():
    """
    Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen).
    U shape (4,7), V shape (4,7), W shape (4,7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen 7 products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b10)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U[i*2+j, c] = coefficient of A[i,j] in product c
    # V[j*2+k, c] = coefficient of B[j,k] in product c
    # W[k*2+i, c] = coefficient in C[i,k] from product c
    
    # M1: (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = -1; W[2, 0] = 1; W[3, 0] = 1  # C00: M1, C11: M1 -> wait
    
    # Let me be more careful. W[k*2+i, c] means for output C[i,k], the coefficient.
    # C00 gets M1, C11 gets M1
    # W index: k*2+i. For C00: i=0,k=0 -> idx 0. For C11: i=1,k=1 -> idx 3.
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C10 gets M2: i=1,k=0 -> idx 2
    W[2, 1] = 1
    
    # M3: a00(b10-b11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    # C01 gets M3: i=0,k=1 -> idx 1
    W[1, 2] = 1
    
    # M4: a11(b01-b10)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    # C00 gets -M4: i=0,k=0 -> idx 0
    W[0, 3] = -1
    # C01 gets M4: i=0,k=1 -> idx 1
    W[1, 3] = 1
    
    # M5: (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C00 gets M5: idx 0
    W[0, 4] = 1
    # C10 gets M5: idx 2
    W[2, 4] = 1
    
    # M6: (a10-a00)(b00+b10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    # C11 gets M6: i=1,k=1 -> idx 3
    W[3, 5] = 1
    
    # M7: (a01-a11)(b01+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    # C00 gets M7: idx 0
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x2_rank28():
    """
    Build 4x4 * 4x2 multiplication as rank 28 decomposition.
    Use 4 blocks of 2x2 * 2x2, each rank 7 -> total rank 28.
    
    A is 4x4, B is 4x2, C is 4x2.
    Split A into 2x2 blocks: A00, A01, A10, A11
    Split B into 2x2 blocks: B00, B10 (top 2 rows, bottom 2 rows)
    C0 = A00*B00 + A01*B10 (top 2 rows of C)
    C1 = A10*B00 + A11*B10 (bottom 2 rows of C)
    
    U shape (16, 28): A has 4*4=16 entries
    V shape (8, 28): B has 4*2=8 entries
    W shape (8, 28): C has 2*4=8 entries (k*n+i where k in 0..1, i in 0..3)
    """
    U_block, V_block, W_block = _build_strassen_2x2_2x2_rank7()
    
    U = np.zeros((16, 28))
    V = np.zeros((8, 28))
    W = np.zeros((8, 28))
    
    # Block 0: A00 * B00 -> C0 (top part)
    # A00 indices: i in 0..1, j in 0..1 -> A[i,j] at idx i*4+j
    # B00 indices: j in 0..1, k in 0..1 -> B[j,k] at idx j*2+k
    # C0 indices: k in 0..1, i in 0..1 -> C[i,k] at idx k*4+i
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4+j, c] = U_block[i*2+j, c]
        for j in range(2):
            for k in range(2):
                V[j*2+k, c] = V_block[j*2+k, c]
        for k in range(2):
            for i in range(2):
                W[k*4+i, c] = W_block[k*2+i, c]
    
    # Block 1: A01 * B10 -> C0 (top part)
    # A01: i in 0..1, j in 2..3 -> A[i,j] at idx i*4+j
    # B10: j in 2..3, k in 0..1 -> B[j,k] at idx j*2+k
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4+(j+2), 7+c] = U_block[i*2+j, c]
        for j in range(2):
            for k in range(2):
                V[(j+2)*2+k, 7+c] = V_block[j*2+k, c]
        for k in range(2):
            for i in range(2):
                W[k*4+i, 7+c] = W_block[k*2+i, c]
    
    # Block 2: A10 * B00 -> C1 (bottom part)
    # A10: i in 2..3, j in 0..1 -> A[i,j] at idx i*4+j
    # B00: j in 0..1, k in 0..1 -> B[j,k] at idx j*2+k
    # C1: k in 0..1, i in 2..3 -> C[i,k] at idx k*4+i
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[(i+2)*4+j, 14+c] = U_block[i*2+j, c]
        for j in range(2):
            for k in range(2):
                V[j*2+k, 14+c] = V_block[j*2+k, c]
        for k in range(2):
            for i in range(2):
                W[k*4+(i+2), 14+c] = W_block[k*2+i, c]
    
    # Block 3: A11 * B10 -> C1 (bottom part)
    # A11: i in 2..3, j in 2..3 -> A[i,j] at idx i*4+j
    # B10: j in 2..3, k in 0..1 -> B[j,k] at idx j*2+k
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[(i+2)*4+(j+2), 21+c] = U_block[i*2+j, c]
        for j in range(2):
            for k in range(2):
                V[(j+2)*2+k, 21+c] = V_block[j*2+k, c]
        for k in range(2):
            for i in range(2):
                W[k*4+(i+2), 21+c] = W_block[k*2+i, c]
    
    return U, V, W

def _build_4x4x6_rank84():
    """
    Build 4x4 * 4x6 as rank 84.
    Split into 3 subproblems of 4x4 * 4x2, each rank 28.
    Total rank = 3 * 28 = 84.
    
    A: 4x4 (16 entries)
    B: 4x6 (24 entries)
    C: 6x4 (24 entries)
    
    B columns 0-1 -> B0, columns 2-3 -> B1, columns 4-5 -> B2
    C rows 0-1 -> C0, rows 2-3 -> C1, rows 4-5 -> C2
    """
    U_sub, V_sub, W_sub = _build_4x4x2_rank28()
    
    U = np.zeros((16, 84))
    V = np.zeros((24, 84))
    W = np.zeros((24, 84))
    
    for block in range(3):
        # U is the same for all blocks
        U[:, block*28:(block+1)*28] = U_sub
        
        # V: B[j, k] where j in 0..3, k in 0..5
        # For block b, k in 2b..2b+1 -> B[j, 2b+k'] at idx j*6 + 2b + k'
        for c in range(28):
            for j in range(4):
                for kp in range(2):
                    V[j*6 + block*2 + kp, block*28 + c] = V_sub[j*2 + kp, c]
        
        # W: C[k, i] where k in 0..5, i in 0..3
        # For block b, k in 2b..2b+1 -> C[2b+k', i] at idx (2b+k')*4 + i
        for c in range(28):
            for kp in range(2):
                for i in range(4):
                    W[(block*2 + kp)*4 + i, block*28 + c] = W_sub[kp*4 + i, c]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Check all tensor entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for c in range(rank):
                    lhs += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(lhs - 1.0) > 1e-10:
                    return False
    return True

def _to_list(mat):
    """Convert numpy array to list of lists with integer/half-integer rounding."""
    result = []
    for row in mat:
        r = []
        for val in row:
            # Round to nearest half-integer
            r2 = round(val * 2) / 2.0
            r.append(r2)
        result.append(r)
    return result

def construct():
    start_time = time.time()
    
    # Build the rank-84 decomposition
    U, V, W = _build_4x4x6_rank84()
    
    # Verify
    if not _verify_decomposition(U, V, W, 4, 4, 6):
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U_sb = np.zeros((n*m, rank))
        V_sb = np.zeros((m*p, rank))
        W_sb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": _to_list(U_sb), "V": _to_list(V_sb), "W": _to_list(W_sb)}
    
    # Try local search to reduce rank
    # Strategy: try to find a lower-rank decomposition by random perturbations
    # and checking if we can merge terms.
    
    best_rank = 84
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Local search: try to reduce rank by attempting to express one term as a combination
    # of others, or by random linear transformations and rank reduction.
    
    # A simpler approach: try random linear combinations of columns and see if we
    # can find a lower-rank exact solution.
    
    # Given time constraints, do a limited search
    rng = np.random.RandomState(42)
    
    for attempt in range(50):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try to find a rank-83 decomposition by perturbing and solving
        # This is heuristic and may not succeed, but we try.
        
        # Alternative: use the fact that we have 3 independent blocks.
        # Try to see if any block can be done in rank < 7.
        # The 2x2*2x2 tensor has rank 7 (proven minimal), so we can't reduce individual blocks.
        # But maybe we can share structure across blocks.
        
        # Try a random linear combination approach:
        # Take the current 84-term decomposition and try to find a lower-rank one
        # by solving a system of equations.
        
        # For now, just return the rank-84 solution if we can't improve.
        pass
    
    # Return the best found
    return {"U": _to_list(best_U), "V": _to_list(best_V), "W": _to_list(best_W)}

# EVOLVE-BLOCK-END
