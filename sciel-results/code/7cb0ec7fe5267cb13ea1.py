import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor entry T[i, j, k] should be 1 for all i, j, k.
    T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1
    
    return U, V, Sw

def _build_strassen_based(n, m, p):
    """
    Build a decomposition for n x m times m x p by applying Strassen to 2x2 blocks.
    For 2x3x10:
    - Use Strassen for 2x2 subblocks of B (columns 0-1, 2-3, 4-5, 6-7, 8-9)
    - Each 2x2x2 block uses rank 7
    - But A is 2x3, so for each pair of columns k1, k2 of B:
      - We multiply A (2x3) by B_sub (3x2) to get C_sub (2x2)
      - This can be done by treating A as [A_left (2x2) | A_right (2x1)]
      - B_sub = [B_left (2x2) | B_right (2x1)]
      - C_sub = A_left * B_left + A_right * B_right
      - A_left * B_left: Strassen (rank 7)
      - A_right * B_right: schoolbook (rank 2*1*2 = 4? No, A_right is 2x1, B_right is 2x2... wait)
      
    Actually, let me think more carefully.
    A is 2x3, B is 3x10.
    For a pair of columns (k1, k2) of B:
    B_sub is 3x2.
    A * B_sub is 2x2.
    
    We can decompose this as:
    A * B_sub = [A[:, :2] * B_sub[:, :2]] + [A[:, 2:3] * B_sub[:, 2:3]]
    
    A[:, :2] is 2x2, B_sub[:, :2] is 3x2... no that's not right.
    
    Let me re-index. B is 3x10. Columns are indexed by k=0..9.
    For columns k1, k2:
    B_sub = B[:, [k1, k2]] which is 3x2.
    A is 2x3.
    C_sub = A @ B_sub which is 2x2.
    
    Now, A @ B_sub = sum_{j=0}^{2} A[:, j] * B_sub[j, :]
    
    We can split j=0,1 and j=2:
    = A[:, :2] @ B_sub[:2, :] + A[:, 2:3] @ B_sub[2:3, :]
    
    A[:, :2] is 2x2, B_sub[:2, :] is 2x2. So A[:, :2] @ B_sub[:2, :] is a 2x2x2 multiplication.
    A[:, 2:3] is 2x1, B_sub[2:3, :] is 1x2. So A[:, 2:3] @ B_sub[2:3, :] is outer product, rank 4? No.
    A[:, 2:3] @ B_sub[2:3, :] = (A[:, 2:3]) * (B_sub[2, :]) which is a 2x2 matrix that is a rank-1 outer product.
    So this part has rank 1? No, it's 4 separate multiplications in schoolbook, but the tensor for 2x1 times 1x2 has rank 4? 
    Actually, the tensor T[i, j, k] for multiplying a 2x1 by 1x2 matrix:
    T[i, j, k] where i=0..1, j=0..0, k=0..1.
    T[i, 0, k] = 1 for all i, k.
    This is a 2x1x2 tensor. Its rank is 2 (since it's a sum of 2 rank-1 tensors).
    
    So total rank per pair = 7 (Strassen for 2x2x2) + 2 (for 2x1x2) = 9.
    With 5 pairs, total rank = 5 * 9 = 45.
    
    Wait, but the current program says 55. Let me check the current implementation.
    The current program uses 7 Strassen terms + 4 schoolbook terms = 11 per pair.
    Why 4 schoolbook? Because they're treating the 2x1x2 as 4 separate terms? That's not optimal.
    
    Let me verify: for 2x1x2, the tensor is:
    T[0,0,0] = 1, T[0,0,1] = 1, T[1,0,0] = 1, T[1,0,1] = 1
    This can be decomposed as:
    e_0^A * e_0^B * e_0^C + e_0^A * e_0^B * e_1^C + e_1^A * e_0^B * e_0^C + e_1^A * e_0^B * e_1^C
    = (e_0^A + e_1^A) * e_0^B * e_0^C + e_0^A * e_0^B * e_1^C
    Wait, that's not right.
    
    Let U, V, W for 2x1x2:
    U: (2*1, R) = (2, R)
    V: (1*2, R) = (2, R)
    W: (2*2, R) = (4, R)
    
    We need: sum_r U[i, r] * V[k, r] * W[k*2+i, r] = 1 for all i=0,1, k=0,1.
    
    Try R=2:
    Term 0: U[0,0]=1, U[1,0]=1, V[0,0]=1, V[1,0]=0, W[0,0]=1, W[1,0]=0, W[2,0]=0, W[3,0]=0
    Term 1: U[0,1]=0, U[1,1]=0, V[0,1]=0, V[1,1]=1, W[0,1]=0, W[1,1]=1, W[2,1]=0, W[3,1]=1
    
    Check:
    i=0, k=0: U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] = 1*1*1 + 0 = 1 OK
    i=0, k=1: U[0,0]*V[1,0]*W[1,0] + U[0,1]*V[1,1]*W[1,1] = 1*0*0 + 0*1*1 = 0 FAIL
    
    Let me try again.
    The tensor is T[i, j, k] = 1 for i=0,1, j=0, k=0,1.
    This is T = sum_{i,k} e_i^A * e_0^B * e_k^C
    = (e_0^A + e_1^A) * e_0^B * e_0^C + e_0^A * e_0^B * e_1^C
    
    So:
    Term 0: U[0,0]=1, U[1,0]=1, V[0,0]=1, W[0,0]=1 (W[k=0, i=0] = W[0,0]=1)
    Term 1: U[0,1]=1, U[1,1]=0, V[0,1]=1, W[1,1]=1 (W[k=1, i=0] = W[1,1]=1)
    
    Check:
    i=0, k=0: U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] = 1*1*1 + 1*1*0 = 1 OK
    i=0, k=1: U[0,0]*V[1,0]*W[1,0] + U[0,1]*V[1,1]*W[1,1] = 1*0*0 + 1*1*1 = 1 OK
    i=1, k=0: U[1,0]*V[0,0]*W[2,0] + U[1,1]*V[0,1]*W[2,1] = 1*1*0 + 0*1*0 = 0 FAIL
    
    I'm making an error. W[k*n+i] for n=2, i=0,1; k=0,1.
    W[0*2+0] = W[0] for i=0, k=0
    W[0*2+1] = W[1] for i=1, k=0
    W[1*2+0] = W[2] for i=0, k=1
    W[1*2+1] = W[3] for i=1, k=1
    
    So W index is k*n + i = 2*k + i.
    
    Term 0: U[0,0]=1, U[1,0]=1, V[0,0]=1, W[0,0]=1, W[1,0]=0, W[2,0]=0, W[3,0]=0
    Term 1: U[0,1]=1, U[1,1]=0, V[0,1]=1, W[0,1]=0, W[1,1]=0, W[2,1]=1, W[3,1]=0
    
    Check:
    i=0, k=0: U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] = 1*1*1 + 1*1*0 = 1 OK
    i=0, k=1: U[0,0]*V[1,0]*W[2,0] + U[0,1]*V[1,1]*W[2,1] = 1*0*0 + 1*0*1 = 0 FAIL
    
    I think the issue is that V is (m*p, R) = (1*2, R) = (2, R), indexed by j*p+k where j=0, k=0,1.
    So V[0] for j=0, k=0, V[1] for j=0, k=1.
    
    Let me redo:
    T[i, j, k] = 1 for i=0,1, j=0, k=0,1.
    Decomposition:
    Term 0: U[0,0]=1, U[1,0]=1, V[0,0]=1, W[0,0]=1, W[1,0]=1, W[2,0]=0, W[3,0]=0
    Term 1: U[0,1]=0, U[1,1]=0, V[1,1]=1, W[2,1]=1, W[3,1]=1
    
    Check:
    i=0, k=0: U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] = 1*1*1 + 0*0*0 = 1 OK
    i=0, k=1: U[0,0]*V[1,0]*W[2,0] + U[0,1]*V[1,1]*W[2,1] = 1*0*0 + 0*1*1 = 0 FAIL
    
    I'm struggling. Let me just use rank 4 for the 2x1x2 part and move on. The total would be 5*(7+4) = 55, same as current.
    
    Actually, the 2x1x2 tensor has rank 2. Let me find it.
    T = sum_{i=0}^1 sum_{k=0}^1 e_i^A e_0^B e_k^C
    = e_0^A e_0^B e_0^C + e_0^A e_0^B e_1^C + e_1^A e_0^B e_0^C + e_1^A e_0^B e_1^C
    
    We can write:
    = (e_0^A + e_1^A) e_0^B e_0^C + e_0^A e_0^B e_1^C
    
    So:
    Term 0: U[0,0]=1, U[1,0]=1, V[0,0]=1, W[0,0]=1, W[1,0]=1, W[2,0]=0, W[3,0]=0
    Term 1: U[0,1]=1, U[1,1]=0, V[0,1]=1, W[0,1]=0, W[1,1]=0, W[2,1]=1, W[3,1]=0
    
    Check:
    i=0, k=0: U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] = 1*1*1 + 1*1*0 = 1 OK
    i=0, k=1: U[0,0]*V[1,0]*W[2,0] + U[0,1]*V[1,1]*W[2,1] = 1*0*0 + 1*1*1 = 1 OK
    i=1, k=0: U[1,0]*V[0,0]*W[1,0] + U[1,1]*V[0,1]*W[1,1] = 1*1*1 + 0*1*0 = 1 OK
    i=1, k=1: U[1,0]*V[1,0]*W[3,0] + U[1,1]*V[1,1]*W[3,1] = 1*0*0 + 0*1*0 = 0 FAIL
    
    I see the issue. W[3] corresponds to k=1, i=1. But in Term 1, I set W[3,1]=0.
    I need W[3,1]=1 for i=1, k=1.
    
    Let me try:
    Term 0: U[0,0]=1, U[1,0]=1, V[0,0]=1, W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1
    Term 1: all zeros
    
    Then:
    i=0, k=0: 1*1*1 = 1 OK
    i=0, k=1: 1*1*1 = 1 OK
    i=1, k=0: 1*1*1 = 1 OK
    i=1, k=1: 1*1*1 = 1 OK
    
    So rank 1 works? No, that's not a valid decomposition because U, V, W must be such that the sum equals T.
    With one term: U[i,0]*V[k,0]*W[2k+i,0] = 1 for all i,k.
    If U[i,0]=1, V[k,0]=1, W[2k+i,0]=1, then it works.
    
    So the 2x1x2 tensor has rank 1? That seems wrong.
    
    Wait, the tensor is T[i,j,k] where j is the index of the middle dimension.
    For 2x1x2: n=2, m=1, p=2.
    U: (n*m, R) = (2*1, R) = (2, R)
    V: (m*p, R) = (1*2, R) = (2, R)
    W: (p*n, R) = (2*2, R) = (4, R)
    
    T[i, j, k] = 1 for i=0,1, j=0, k=0,1.
    
    With one term: U[0,0]=1, U[1,0]=1, V[0,0]=1, V[1,0]=1, W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1.
    Then sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = U[i*1+0, 0] * V[0*2+k, 0] * W[k*2+i, 0]
    = U[i, 0] * V[k, 0] * W[2k+i, 0] = 1 * 1 * 1 = 1.
    
    So yes, rank 1 works! The 2x1x2 tensor has rank 1.
    
    So total rank per pair = 7 + 1 = 8.
    With 5 pairs, total rank = 5 * 8 = 40.
    
    This is a significant improvement over 55.
    
    Let me implement this.
    """
    n, m, p = 2, 3, 10
    num_pairs = p // 2  # 5 pairs
    
    terms_per_pair = 7 + 1  # Strassen + 2x1x2
    total_rank = num_pairs * terms_per_pair
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    Su, Sv, Sw = _strassen_2x2x2()
    
    for pair in range(num_pairs):
        col_offset = pair * terms_per_pair
        k0 = pair * 2
        
        # Strassen part for A[:, :2] * B_sub[:2, :]
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    val = Su[i * 2 + j, r]
                    if val != 0:
                        U[i * m + j, col_offset + r] = val
            for j in range(2):
                for k in range(2):
                    val = Sv[j * 2 + k, r]
                    if val != 0:
                        V[j * p + (k0 + k), col_offset + r] = val
            for k in range(2):
                for i in range(2):
                    val = Sw[k * 2 + i, r]
                    if val != 0:
                        W[(k0 + k) * n + i, col_offset + r] = val
        
        # 2x1x2 part for A[:, 2:3] * B_sub[2:3, :]
        sb_offset = 7
        idx = col_offset + sb_offset
        # U: i=0,1; j=2
        U[0 * m + 2, idx] = 1
        U[1 * m + 2, idx] = 1
        # V: j=2; k=k0, k0+1
        V[2 * p + k0, idx] = 1
        V[2 * p + (k0 + 1), idx] = 1
        # W: k=k0, k0+1; i=0,1
        W[(k0 + 0) * n + 0, idx] = 1
        W[(k0 + 0) * n + 1, idx] = 1
        W[(k0 + 1) * n + 0, idx] = 1
        W[(k0 + 1) * n + 1, idx] = 1
    
    return U, V, W

def construct():
    n, m, p = 2, 3, 10
    deadline = time.time() + 140
    
    # Build the improved Strassen-based decomposition
    U, V, W = _build_strassen_based(n, m, p)
    rank = U.shape[1]
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
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
