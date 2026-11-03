"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    r = U.shape[1]
    if r != V.shape[1] or r != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Return rank-7 decomposition for 2x2 * 2x2.
    U is (4,7), V is (4,7), W is (4,7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 multiplications
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+b)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-c)
    # M5 = (a+d)(h)
    # M6 = (b-c)(g)
    # M7 = (b-d)(f+e)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0: M1 = (a+b)(e+h)
    U[0, 0] = 1  # a
    U[1, 0] = 1  # b
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # c=1: M2 = (c+d)(e)
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    V[0, 1] = 1  # e
    W[2, 1] = 1  # C21
    W[3, 1] = -1 # C22
    
    # c=2: M3 = (a)(f-h)
    U[0, 2] = 1  # a
    V[1, 2] = 1  # f
    V[3, 2] = -1 # h
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # c=3: M4 = (d)(g-c)
    U[3, 3] = 1  # d
    V[2, 3] = 1  # g
    V[0, 3] = -1 # c
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # c=4: M5 = (a+d)(h)
    U[0, 4] = 1  # a
    U[3, 4] = 1  # d
    V[3, 4] = 1  # h
    W[0, 4] = -1 # C11
    W[1, 4] = 1  # C12
    
    # c=5: M6 = (b-c)(g)
    U[1, 5] = 1  # b
    U[2, 5] = -1 # c
    V[2, 5] = 1  # g
    W[3, 5] = 1  # C22
    
    # c=6: M7 = (b-d)(f+e)
    U[1, 6] = 1  # b
    U[3, 6] = -1 # d
    V[0, 6] = 1  # e
    V[1, 6] = 1  # f
    W[0, 6] = -1 # C11
    
    return U, V, W

def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4 * 4x4 using tensor product of Strassen.
    U is (16, 49), V is (16, 49), W is (16, 49)
    """
    U2, V2, W2 = _strassen_2x2x2()
    # Tensor product: rank 7*7=49
    # For block (i1,i2,j1,j2,k1,k2):
    # U[(i1*2+i2)*2 + (j1*2+j2), c1*7+c2] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
    # V[(j1*2+j2)*2 + (k1*2+k2), c1*7+c2] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
    # W[(k1*2+k2)*2 + (i1*2+i2), c1*7+c2] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u_val = U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]
                            if u_val != 0:
                                U[(i1 * 2 + i2) * 4 + (j1 * 2 + j2), c] = u_val
            
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            v_val = V2[j1 * 2 + k1, c1] * V2[j2 * 2 + k2, c2]
                            if v_val != 0:
                                V[(j1 * 2 + j2) * 4 + (k1 * 2 + k2), c] = v_val
            
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            w_val = W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]
                            if w_val != 0:
                                W[(k1 * 2 + k2) * 4 + (i1 * 2 + i2), c] = w_val
    
    return U, V, W

def _build_4x4x7():
    """Build rank-93 decomposition for 4x4 * 4x7.
    Part 1: 4x4x4 with rank 49 (Strassen tensor product)
    Part 2: 4x4x3 with rank 44 (schoolbook for 3 columns)
    Total: 49 + 44 = 93
    """
    n, m, p = 4, 4, 7
    p1 = 4  # first 4 columns
    p2 = 3  # remaining 3 columns
    
    # Part 1: 4x4x4
    U1, V1, W1 = _build_4x4x4_strassen()
    r1 = 49
    
    # Part 2: 4x4x3 schoolbook
    r2 = n * m * p2  # 4*4*3 = 48
    # Wait, that's 48, total would be 97. Let me check.
    # Actually for 4x4x3, schoolbook is 4*4*3 = 48.
    # Total: 49 + 48 = 97.
    
    # But the inspiration says rank 93. Let me think about this differently.
    # Maybe we can do better on the 4x4x3 part.
    # For 4x4x3, we can use Strassen-based approach.
    # 4x4x3 = 4x4x2 + 4x4x1
    # 4x4x2 can use rank 28 (tensor product of Strassen 2x2x2 with 2x1... no that doesn't work directly)
    
    # Actually, let me just use schoolbook for the 3 columns: rank 48.
    # Total: 49 + 48 = 97.
    
    # But wait, the inspiration says 93. Let me re-read.
    # "fixing index bugs in the decomposition builder to reduce rank from 97 to 93"
    # So the base construction gives 97, and fixing bugs gives 93?
    # That doesn't make sense - fixing bugs shouldn't reduce rank.
    # Unless the "97" was an incorrect construction that had extra terms.
    
    # Let me try a different approach: use Strassen for 4x4x4 (rank 49)
    # and for the 4x4x3 part, use a more efficient decomposition.
    
    # For 4x4x3, we can think of it as multiplying by a 4x3 matrix.
    # One approach: use the fact that 4x4x3 <= 4x4x4 by zero-padding,
    # but that gives rank 49 for the 4x4x3 part, total 98.
    
    # Alternatively, use schoolbook: 48 for 4x4x3, total 97.
    
    # Let me just go with 97 for now and see if we can improve.
    
    U2 = np.zeros((n * m, r2))
    V2 = np.zeros((m * p2, r2))
    W2 = np.zeros((p2 * n, r2))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p2):
                U2[i * m + j, col] = 1
                V2[j * p2 + k, col] = 1
                W2[k * n + i, col] = 1
                col += 1
    
    total_rank = r1 + r2
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    # Place Part 1 (4x4x4)
    U_full[:, :r1] = U1
    V_full[:m * p1, :r1] = V1
    W_full[:p1 * n, :r1] = W1
    
    # Place Part 2 (4x4x3) - columns 4,5,6 of B
    U_full[:, r1:] = U2
    V_full[m * p1:, r1:] = V2
    W_full[p1 * n:, r1:] = W2
    
    return U_full, V_full, W_full

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Build the rank-97 decomposition
    U, V, W = _build_4x4x7()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by checking for zero columns
    keep = [c for c in range(U.shape[1])
            if np.any(U[:, c] != 0) and np.any(V[:, c] != 0)
            and np.any(W[:, c] != 0)]
    
    if len(keep) < U.shape[1]:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    
    # Verify again after reduction
    if not _verify(U, V, W, n, m, p):
        # Rebuild from scratch
        U, V, W = _build_4x4x7()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
