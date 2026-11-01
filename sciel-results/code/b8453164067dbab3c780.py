"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 140  # 140s safety margin (evaluator kills at 150s)

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a flat array.
    T[a, b, d] = 1 if a=i*m+j, b=j*p+k, d=k*n+i for some i,j,k, else 0.
    Returns T as a 3D array of shape (n*m, m*p, p*n).
    """
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-6):
    """Check if U, V, W exactly decompose T.
    Returns True if sum_t U[a,t]*V[b,t]*W[d,t] == T[a,b,d] for all a,b,d.
    """
    # Compute the full tensor from the decomposition
    # T_decomp[a,b,d] = sum_t U[a,t]*V[b,t]*W[d,t]
    # This is a tensor contraction: einsum('at,bt,dt->abd', U, V, W)
    T_decomp = np.einsum('at,bt,dt->abd', U, V, W, optimize=True)
    # Check all entries
    if not np.allclose(T_decomp, T, atol=tol, rtol=0):
        return False
    return True

def _alternating_minimization(T, r, n, m, p, max_iter=500, time_budget=20.0):
    """Try to find a rank-r decomposition via alternating least squares.
    Returns (U, V, W) if found, else None.
    """
    nm, mp, pn = n * m, m * p, p * n
    T_flat = T.reshape(nm, mp, pn)
    
    # Precompute the "slices" of T for efficient solving
    # For solving W[d,:]: we need to solve M @ W[d,:]^T = T[:, :, d].ravel()
    # where M = U @ V.T (nm x mp)
    # For solving U[a,:]: we need to solve M2 @ U[a,:]^T = T[a, :, :].ravel()
    # where M2 = V @ W.T (mp x pn)
    # For solving V[b,:]: we need to solve M3 @ V[b,:]^T = T[:, b, :].ravel()
    # where M3 = U @ W.T (nm x pn)
    
    # Precompute T slices
    T_w = T_flat.transpose(2, 0, 1).reshape(pn, nm * mp)  # T_w[d, a*mp+b] = T[a,b,d]
    T_u = T_flat.transpose(0, 1, 2).reshape(nm, mp * pn)  # T_u[a, b*pn+d] = T[a,b,d]
    T_v = T_flat.transpose(1, 0, 2).reshape(mp, nm * pn)  # T_v[b, a*pn+d] = T[a,b,d]
    
    start = time.time()
    
    # Random initialization
    np.random.seed(42)
    U = np.random.randn(nm, r) * 0.5
    V = np.random.randn(mp, r) * 0.5
    W = np.random.randn(pn, r) * 0.5
    
    for iteration in range(max_iter):
        if time.time() - start > time_budget:
            return None
        
        # Solve for W given U, V
        M = U @ V.T  # (nm, mp)
        M_flat = M.reshape(nm * mp)  # not needed, M is already 2D
        # For each d: solve M @ x = T_w[d, :] where x = W[d, :]
        # This is an overdetermined system: (nm*mp) equations, r unknowns
        # Use least squares
        W_new = np.zeros((pn, r))
        for d in range(pn):
            try:
                W_new[d, :], _, _, _ = np.linalg.lstsq(M, T_w[d, :], rcond=None)
            except:
                return None
        W = W_new
        
        # Solve for U given V, W
        M2 = V @ W.T  # (mp, pn)
        U_new = np.zeros((nm, r))
        for a in range(nm):
            try:
                U_new[a, :], _, _, _ = np.linalg.lstsq(M2, T_u[a, :], rcond=None)
            except:
                return None
        U = U_new
        
        # Solve for V given U, W
        M3 = U @ W.T  # (nm, pn)
        V_new = np.zeros((mp, r))
        for b in range(mp):
            try:
                V_new[b, :], _, _, _ = np.linalg.lstsq(M3, T_v[b, :], rcond=None)
            except:
                return None
        V = V_new
        
        # Check convergence
        if _check_decomposition(U, V, W, T, tol=1e-8):
            return U, V, W
    
    # Final check
    if _check_decomposition(U, V, W, T, tol=1e-8):
        return U, V, W
    
    return None

def _strassen_decomp_2x2x2():
    """Return U, V, W for the Strassen decomposition of 2x2 * 2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    The tensor T[a,b,d] = 1 if C[i,j] = sum_{k} A[i,k] * B[k,j] for a=i*2+j, b=k*2+j', d=j'*2+i'
    Wait, the convention is:
    a = i*m + j = i*2 + j
    b = j*p + k = j*2 + k
    d = k*n + i = k*2 + i
    So for 2x2*2x2:
    a = i*2+j (i in {0,1}, j in {0,1})
    b = j*2+k (j in {0,1}, k in {0,1})
    d = k*2+i (k in {0,1}, i in {0,1})
    
    Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A10)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Let's map indices:
    A[i,j] -> a = i*2+j
    B[j,k] -> b = j*2+k
    C[i,j] -> d = j*2+i  (wait, d = k*n + i = k*2 + i, but C is indexed by (i,j) where i is row, j is col)
    Actually, the output C[i,j] is at position d = j*2 + i? No.
    The tensor is T[a, b, d] where:
    a = i*m + j (input A index)
    b = j*p + k (input B index)
    d = k*n + i (output C index)
    
    So for C[i,j], the output index is d = j*2 + i? No, d = k*n + i where k is the column index of C?
    Wait, C is n x p = 2 x 2. C[i,j] where i in {0,1}, j in {0,1}.
    The formula says d = k*n + i. Here k is the column index of C (since B is j x k, C is i x k).
    So d = j*2 + i.
    
    Let's verify: C[0,0] -> d = 0*2+0 = 0
    C[0,1] -> d = 1*2+0 = 2
    C[1,0] -> d = 0*2+1 = 1
    C[1,1] -> d = 1*2+1 = 3
    
    So:
    C00 is at d=0
    C01 is at d=2
    C10 is at d=1
    C11 is at d=3
    
    Strassen formulas:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Now, U[a, t] * V[b, t] * W[d, t] = sum over t of U[a,t]*V[b,t]*W[d,t]
    
    For M1 = (A00+A11)(B00+B11):
    A00 is at a=0, A11 is at a=3
    B00 is at b=0, B11 is at b=3
    
    So M1 contributes to:
    a=0, b=0: U[0,0]*V[0,0]*W[d,0]
    a=0, b=3: U[0,0]*V[3,0]*W[d,0]
    a=3, b=0: U[3,0]*V[0,0]*W[d,0]
    a=3, b=3: U[3,0]*V[3,0]*W[d,0]
    
    And M1 appears in C00 (d=0) with coeff +1, and C11 (d=3) with coeff +1.
    So W[0,0] = 1, W[3,0] = 1, other W[*,0] = 0.
    
    Similarly for M2 = (A10+A11)B00:
    A10 is at a=2, A11 is at a=3
    B00 is at b=0
    
    M2 appears in C10 (d=1) with coeff +1, and C11 (d=3) with coeff -1.
    So W[1,1] = 1, W[3,1] = -1.
    
    M3 = A00(B01-B11):
    A00 is at a=0
    B01 is at b=1, B11 is at b=3
    
    M3 appears in C01 (d=2) with coeff +1, and C11 (d=3) with coeff +1.
    So W[2,2] = 1, W[3,2] = 1.
    
    M4 = A11(B10-B00):
    A11 is at a=3
    B10 is at b=2, B00 is at b=0
    
    M4 appears in C00 (d=0) with coeff +1, and C10 (d=1) with coeff +1.
    So W[0,3] = 1, W[1,3] = 1.
    
    M5 = (A00+A10)B11:
    A00 is at a=0, A10 is at a=2
    B11 is at b=3
    
    M5 appears in C00 (d=0) with coeff -1, and C01 (d=2) with coeff +1.
    So W[0,4] = -1, W[2,4] = 1.
    
    M6 = (A10-A00)(B00+B01):
    A10 is at a=2, A00 is at a=0
    B00 is at b=0, B01 is at b=1
    
    M6 appears in C11 (d=3) with coeff +1.
    So W[3,5] = 1.
    
    M7 = (A01-A11)(B10+B11):
    A01 is at a=1, A11 is at a=3
    B10 is at b=2, B11 is at b=3
    
    M7 appears in C00 (d=0) with coeff +1.
    So W[0,6] = 1.
    
    Now let's construct U, V, W:
    
    U[0,0] = 1, U[3,0] = 1  (M1: A00, A11)
    U[2,1] = 1, U[3,1] = 1  (M2: A10, A11)
    U[0,2] = 1               (M3: A00)
    U[3,3] = 1               (M4: A11)
    U[0,4] = 1, U[2,4] = 1  (M5: A00, A10)
    U[2,5] = 1, U[0,5] = -1 (M6: A10, -A00)
    U[1,6] = 1, U[3,6] = -1 (M7: A01, -A11)
    
    V[0,0] = 1, V[3,0] = 1  (M1: B00, B11)
    V[0,1] = 1               (M2: B00)
    V[1,2] = 1, V[3,2] = -1 (M3: B01, -B11)
    V[2,3] = 1, V[0,3] = -1 (M4: B10, -B00)
    V[3,4] = 1               (M5: B11)
    V[0,5] = 1, V[1,5] = 1  (M6: B00, B01)
    V[2,6] = 1, V[3,6] = 1  (M7: B10, B11)
    
    W[0,0] = 1, W[3,0] = 1  (M1: C00, C11)
    W[1,1] = 1, W[3,1] = -1 (M2: C10, -C11)
    W[2,2] = 1, W[3,2] = 1  (M3: C01, C11)
    W[0,3] = 1, W[1,3] = 1  (M4: C00, C10)
    W[0,4] = -1, W[2,4] = 1 (M5: -C00, C01)
    W[3,5] = 1               (M6: C11)
    W[0,6] = 1               (M7: C00)
    
    Let's double check M6: (A10-A00)(B00+B01)
    A10 is at a=2, A00 is at a=0
    So U[2,5] = 1, U[0,5] = -1. Correct.
    B00 is at b=0, B01 is at b=1
    So V[0,5] = 1, V[1,5] = 1. Correct.
    M6 appears in C11 (d=3) with coeff +1.
    So W[3,5] = 1. Correct.
    
    Let's double check M7: (A01-A11)(B10+B11)
    A01 is at a=1, A11 is at a=3
    So U[1,6] = 1, U[3,6] = -1. Correct.
    B10 is at b=2, B11 is at b=3
    So V[2,6] = 1, V[3,6] = 1. Correct.
    M7 appears in C00 (d=0) with coeff +1.
    So W[0,6] = 1. Correct.
    
    Let's verify the full tensor.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1
    
    # M3
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1
    
    # M4
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1
    
    # M5
    U[0,4] = 1; U[2,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1
    
    # M6
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _identity_decomp_2x1x4():
    """Return U, V, W for the identity decomposition of 2x1 * 1x4 matrix multiplication.
    This is just the outer product. Rank = 2*1*4 = 8.
    U has shape (2, 8), V has shape (4, 8), W has shape (8, 8).
    Wait, for 2x1 * 1x4:
    n=2, m=1, p=4
    nm=2, mp=4, pn=8
    T[a,b,d] = 1 if a=i*1+j=i, b=j*4+k=k, d=k*2+i
    i in {0,1}, j=0, k in {0,1,2,3}
    a = i
    b = k
    d = k*2 + i
    
    So T[i, k, k*2+i] = 1.
    This is a permutation tensor. Rank = 8.
    U[i, t] * V[k, t] * W[k*2+i, t] = 1
    We can set U[i, i*4+k] = 1, V[k, i*4+k] = 1, W[k*2+i, i*4+k] = 1.
    """
    U = np.zeros((2, 8))
    V = np.zeros((4, 8))
    W = np.zeros((8, 8))
    for i in range(2):
        for k in range(4):
            t = i*4 + k
            U[i, t] = 1
            V[k, t] = 1
            W[k*2+i, t] = 1
    return U, V, W

def _build_2x3x4_decomp():
    """Build a rank-22 decomposition for 2x3 * 3x4.
    Split k into two parts: k in {0,1} and k in {2,3}.
    For k in {0,1}: This is a 2x3 * 3x2 problem.
    We can decompose this into a 2x2 * 2x2 (Strassen, rank 7) and a 2x1 * 1x2 (identity, rank 4).
    Wait, 2x3 * 3x2:
    A is 2x3, B is 3x2.
    Split A into A_left (2x2) and A_right (2x1).
    Split B into B_top (2x2) and B_bottom (1x2).
    C = A_left * B_top + A_right * B_bottom.
    A_left * B_top is 2x2 * 2x2, rank 7 (Strassen).
    A_right * B_bottom is 2x1 * 1x2, rank 4 (identity).
    Total rank = 11.
    
    Similarly for k in {2,3}: another 11.
    Total rank = 22.
    
    Let's construct this.
    For the first block (k=0,1):
    A_left: A[0,0], A[0,1], A[1,0], A[1,1] -> a=0,1,3,4 (since a=i*3+j)
    Wait, for 2x3: a = i*3+j.
    A[0,0]->0, A[0,1]->1, A[0,2]->2
    A[1,0]->3, A[1,1]->4, A[1,2]->5
    
    B is 3x2: b = j*2+k.
    B[0,0]->0, B[0,1]->1
    B[1,0]->2, B[1,1]->3
    B[2,0]->4, B[2,1]->5
    
    C is 2x2: d = k*2+i.
    C[0,0]->0, C[0,1]->2
    C[1,0]->1, C[1,1]->3
    
    For k=0,1:
    A_left involves A[0,0], A[0,1], A[1,0], A[1,1] -> a=0,1,3,4
    A_right involves A[0,2], A[1,2] -> a=2,5
    B_top involves B[0,0], B[0,1], B[1,0], B[1,1] -> b=0,1,2,3
    B_bottom involves B[2,0], B[2,1] -> b=4,5
    
    We need to map the Strassen decomposition (which assumes 2x2 inputs) to this.
    The Strassen U, V, W are for a=0,1,2,3 (i*2+j).
    Here, A_left has a=0,1,3,4.
    We need to permute the rows of U.
    Strassen a=0 -> A[0,0] -> our a=0
    Strassen a=1 -> A[0,1] -> our a=1
    Strassen a=2 -> A[1,0] -> our a=3
    Strassen a=3 -> A[1,1] -> our a=4
    
    So we need a permutation P_A such that P_A[0]=0, P_A[1]=1, P_A[2]=3, P_A[3]=4.
    Similarly for B:
    Strassen b=0 -> B[0,0] -> our b=0
    Strassen b=1 -> B[0,1] -> our b=1
    Strassen b=2 -> B[1,0] -> our b=2
    Strassen b=3 -> B[1,1] -> our b=3
    So B_top uses b=0,1,2,3 directly. No permutation needed.
    
    For W:
    Strassen d=0 -> C[0,0] -> our d=0
    Strassen d=1 -> C[1,0] -> our d=1
    Strassen d=2 -> C[0,1] -> our d=2
    Strassen d=3 -> C[1,1] -> our d=3
    So W uses d=0,1,2,3 directly. No permutation needed.
    
    For A_right * B_bottom (2x1 * 1x2):
    A_right: a=2,5
    B_bottom: b=4,5
    C: d=0,1,2,3
    
    T[a,b,d] = 1 if a=i*3+2, b=2*2+k=4+k, d=k*2+i
    i in {0,1}, k in {0,1}
    a = 2+3*i
    b = 4+k
    d = 2*k+i
    
    This is a 2x1 * 1x2 problem.
    We can use the identity decomposition logic.
    Rank = 4.
    
    Let's construct the full 2x3x4 decomposition.
    """
    # First block: k=0,1
    U_str, V_str, W_str = _strassen_decomp_2x2x2()
    # Permute U_str rows: 0->0, 1->1, 2->3, 3->4
    U_str_perm = np.zeros((6, 7))
    U_str_perm[0, :] = U_str[0, :]
    U_str_perm[1, :] = U_str[1, :]
    U_str_perm[3, :] = U_str[2, :]
    U_str_perm[4, :] = U_str[3, :]
    
    # V_str uses b=0,1,2,3 directly
    V_str_block = np.zeros((6, 7))
    V_str_block[0:4, :] = V_str
    
    # W_str uses d=0,1,2,3 directly
    W_str_block = np.zeros((4, 7))
    W_str_block[:] = W_str
    
    # Identity part for A_right * B_bottom
    U_id = np.zeros((6, 4))
    V_id = np.zeros((6, 4))
    W_id = np.zeros((4, 4))
    for i in range(2):
        for k in range(2):
            t = i*2 + k
            U_id[2+3*i, t] = 1  # a = 2+3*i
            V_id[4+k, t] = 1     # b = 4+k
            W_id[2*k+i, t] = 1   # d = 2*k+i
    
    # Combine
    U1 = np.hstack([U_str_perm, U_id])
    V1 = np.hstack([V_str_block, V_id])
    W1 = np.hstack([W_str_block, W_id])
    
    # Second block: k=2,3
    # This is similar, but b indices are shifted by 4 (since b = j*4+k, and now k=2,3)
    # Wait, for 2x3 * 3x4: b = j*4+k.
    # For k=2,3:
    # B_top: B[0,2], B[0,3], B[1,2], B[1,3] -> b=2,3,6,7
    # B_bottom: B[2,2], B[2,3] -> b=10,11
    
    U_str2 = np.zeros((6, 7))
    U_str2[0, :] = U_str[0, :]
    U_str2[1, :] = U_str[1, :]
    U_str2[3, :] = U_str[2, :]
    U_str2[4, :] = U_str[3, :]
    
    V_str2 = np.zeros((12, 7))
    V_str2[2, :] = V_str[0, :]
    V_str2[3, :] = V_str[1, :]
    V_str2[6, :] = V_str[2, :]
    V_str2[7, :] = V_str[3, :]
    
    W_str2 = np.zeros((8, 7))
    # C indices for k=2,3:
    # d = k*2+i
    # k=2: d=4+i -> 4,5
    # k=3: d=6+i -> 6,7
    # Strassen d=0 -> C[0,0] -> our d=4
    # Strassen d=1 -> C[1,0] -> our d=5
    # Strassen d=2 -> C[0,1] -> our d=6
    # Strassen d=3 -> C[1,1] -> our d=7
    W_str2[4, :] = W_str[0, :]
    W_str2[5, :] = W_str[1, :]
    W_str2[6, :] = W_str[2, :]
    W_str2[7, :] = W_str[3, :]
    
    U_id2 = np.zeros((6, 4))
    V_id2 = np.zeros((12, 4))
    W_id2 = np.zeros((8, 4))
    for i in range(2):
        for k in range(2):
            t = i*2 + k
            U_id2[2+3*i, t] = 1
            V_id2[10+k, t] = 1  # b = 2*4+k = 8+k? No, B is 3x4. b = j*4+k.
                                 # j=2, k=2,3 -> b = 8+2=10, 8+3=11. Correct.
            W_id2[2*(2+k)+i, t] = 1 # d = k*2+i, but here k is 2,3.
                                     # Wait, the formula is d = k*n + i = k*2 + i.
                                     # For the second block, the local k is 0,1, but global k is 2,3.
                                     # So d = (2+k_local)*2 + i = 4 + 2*k_local + i.
                                     # k_local=0 -> d=4+i
                                     # k_local=1 -> d=6+i
                                     # So d = 4 + 2*t%2 + t//2?
                                     # t = i*2 + k_local
                                     # i = t//2, k_local = t%2
                                     # d = 4 + 2*(t%2) + (t//2)
            W_id2[4 + 2*(t%2) + (t//2), t] = 1
    
    U2 = np.hstack([U_str2, U_id2])
    V2 = np.hstack([V_str2, V_id2])
    W2 = np.hstack([W_str2, W_id2])
    
    # Combine both blocks
    U_full = np.hstack([U1, U2])
    V_full = np.hstack([V1, V2])
    W_full = np.hstack([W1, W2])
    
    return U_full, V_full, W_full

def _build_2x3x8_decomp():
    """Build a rank-44 decomposition for 2x3 * 3x8.
    Split k into two parts: k in {0,1,2,3} and k in {4,5,6,7}.
    Each part is a 2x3 * 3x4 problem, which has rank 22.
    Total rank = 44.
    """
    U1, V1, W1 = _build_2x3x4_decomp()
    # U1 is (6, 22), V1 is (12, 22), W1 is (8, 22)
    # For the second block (k=4,5,6,7):
    # V indices are shifted by 12 (since b = j*8+k, and now k=4,7)
    # Wait, for 2x3 * 3x8: b = j*8+k.
    # For k=0,3: b = j*8+k
    # For k=4,7: b = j*8+k
    
    U2 = U1.copy()
    V2 = np.zeros((24, 22))
    # V1 has non-zero rows at 0,1,2,3,4,5 (for first block)
    # For second block, we need to shift these by 16?
    # No, V1 was for 2x3 * 3x4, where b = j*4+k.
    # For 2x3 * 3x8, b = j*8+k.
    # The mapping is different.
    
    # Let's re-derive for 2x3 * 3x8 directly.
    # It's easier to just use the 2x3x4 decomposition and adjust indices.
    # For 2x3 * 3x4:
    # a = i*3+j (i=0,1; j=0,1,2)
    # b = j*4+k (j=0,1,2; k=0,1,2,3)
    # d = k*2+i (k=0,1,2,3; i=0,1)
    
    # For 2x3 * 3x8:
    # a = i*3+j
    # b = j*8+k
    # d = k*2+i
    
    # The first block (k=0,3) uses b = j*8+k.
    # In the 2x3x4 decomp, b = j*4+k.
    # So we need to remap b indices.
    # j=0: b=0,1,2,3 -> in 2x3x8: b=0,1,2,3
    # j=1: b=4,5,6,7 -> in 2x3x8: b=8,9,10,11
    # j=2: b=8,9,10,11 -> in 2x3x8: b=16,17,18,19
    
    # So V1 rows 0,1,2,3 stay the same.
    # V1 rows 4,5,6,7 become 8,9,10,11.
    # V1 rows 8,9,10,11 become 16,17,18,19.
    
    V1_remapped = np.zeros((24, 22))
    V1_remapped[0:4, :] = V1[0:4, :]
    V1_remapped[8:12, :] = V1[4:8, :]
    V1_remapped[16:20, :] = V1[8:12, :]
    
    # W1 uses d = k*2+i. For k=0,3, d=0,1,2,3,4,5,6,7.
    # This is the same for 2x3x8.
    W1_remapped = W1
    
    U1_remapped = U1
    
    # Second block (k=4,7):
    # b = j*8+k
    # j=0: b=4,5,6,7
    # j=1: b=12,13,14,15
    # j=2: b=20,21,22,23
    
    V2_remapped = np.zeros((24, 22))
    V2_remapped[4:8, :] = V1[0:4, :]
    V2_remapped[12:16, :] = V1[4:8, :]
    V2_remapped[20:24, :] = V1[8:12, :]
    
    # W2 uses d = k*2+i. For k=4,7, d=8,9,10,11,12,13,14,15.
    W2_remapped = np.zeros((16, 22))
    W2_remapped[8:16, :] = W1
    
    U2_remapped = U1
    
    U_full = np.hstack([U1_remapped, U2_remapped])
    V_full = np.hstack([V1_remapped, V2_remapped])
    W_full = np.hstack([W1_remapped, W2_remapped])
    
    return U_full, V_full, W_full

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 6, 24, 16
    
    # Build the tensor
    T = _build_tensor(n, m, p)
    
    # Try the rank-44 decomposition
    U, V, W = _build_2x3x8_decomp()
    
    # Verify
    if _check_decomposition(U, V, W, T, tol=1e-8):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank-48
    rank = n * m * p  # 48
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
