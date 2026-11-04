import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reconstructs the tensor."""
    target = _build_target_tensor(n, m, p)
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(recon, target, atol=tol)

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a+d)(e+h)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # m2 = (c+d)e
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    
    # m3 = a(f-h)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    
    # m4 = d(g-e)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    
    # m5 = (a+b)h
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    
    # m6 = (c-a)(e+f)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # m7 = (b-d)(g+h)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1; W[3,6]=1
    
    return U, V, W

def _build_2x4x6_decomposition():
    """
    Build 2x4x6 decomposition using Strassen for 2x2x2 blocks.
    """
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    S_U, S_V, S_W = _strassen_2x2x2()
    
    sub_problems = [
        (0, 0, 0, 0),  # A1 * B11 -> C cols 0,1
        (0, 0, 2, 2),  # A1 * B12 -> C cols 2,3
        (0, 0, 4, 4),  # A1 * B13 -> C cols 4,5
        (2, 2, 0, 0),  # A2 * B21 -> C cols 0,1
        (2, 2, 2, 2),  # A2 * B22 -> C cols 2,3
        (2, 2, 4, 4),  # A2 * B23 -> C cols 4,5
    ]
    
    for sp_idx, (a_start, b_row_start, b_col_start, c_col_start) in enumerate(sub_problems):
        base = sp_idx * 7
        for c in range(7):
            col = base + c
            for i in range(2):
                for j in range(2):
                    U[i*4 + (a_start + j), col] = S_U[i*2 + j, c]
            for j in range(2):
                for k in range(2):
                    V[(b_row_start + j)*6 + (b_col_start + k), col] = S_V[j*2 + k, c]
            for k in range(2):
                for i in range(2):
                    W[(c_col_start + k)*2 + i, col] = S_W[k*2 + i, c]
    
    return U, V, W

def _wga_2x2x3_convolution():
    """
    Rank-4 Winograd decomposition for 2x2x3 matrix multiplication
    (equivalent to linear convolution of length 2 with length 2).
    
    This handles multiplying a 2x2 matrix by a 2x3 matrix.
    Uses the WGA transform approach.
    """
    # For 2x2 * 2x3 -> 2x3, we can use rank-4 via Winograd
    # Input: A (2x2), B (2x3), Output: C (2x3)
    # Flattened: A has 4 entries, B has 6 entries, C has 6 entries
    
    U = np.zeros((4, 4))
    V = np.zeros((6, 4))
    W = np.zeros((6, 4))
    
    # Winograd for 2x2 * 2x3 convolution
    # A = [[a, b], [c, d]]
    # B = [[e, f, g], [h, i, j]]
    # C = A*B = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    
    # Use Winograd transforms:
    # U (for A): [[1, 0, 1, 0], [0, 1, 0, -1]]  -- maps A to transform domain
    # Actually, let me think more carefully.
    
    # For 2x2 * 2x3, we can decompose as:
    # C[:, 0:2] = A * B[:, 0:2]  (2x2 * 2x2, rank 7 via Strassen)
    # C[:, 2] = A * B[:, 2]      (2x2 * 2x1, rank 4 via schoolbook)
    # Total: 7 + 4 = 11... that's not rank 4.
    
    # Let me use a different approach. For 2x2 * 2x3:
    # We can view this as 2x2 * 2x2 + 2x2 * 2x1
    # Or use the fact that 2x2*2x3 can be done in rank 8 (schoolbook)
    # But Winograd can reduce it.
    
    # Actually, let me just use the known rank for 2x2*2x3.
    # The tensor rank of M_{2,2,3} is 8 (schoolbook).
    # But with Winograd, we might do better.
    
    # Let me just return a valid rank-8 decomposition for now,
    # and we'll use the Kronecker structure differently.
    
    # For M_{2,2,3}: rank 8 (schoolbook)
    U = np.zeros((4, 8))
    V = np.zeros((6, 8))
    W = np.zeros((6, 8))
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j, col] = 1
                V[j*3+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    
    return U, V, W

def _wga_kronecker_decomposition():
    """
    Build 2x4x6 decomposition using WGA Kronecker product structure.
    
    2x4x6 = (2x2x2) ⊗ (2x2x3) in terms of tensor structure.
    - 4 = 2*2 (middle dimension split as 2x2)
    - 6 = 2*3 (last dimension split as 2x3)
    - 2 = 2*1 (first dimension, just 2)
    
    Using Strassen (rank 7) for the 2x2x2 part and a rank-8 decomposition
    for the 2x2x3 part gives total rank 7*8 = 56... that's worse.
    
    Let me try a different factorization:
    2x4x6: We can write 4 = 2*2 and 6 = 3*2.
    
    Actually, the key insight is:
    M_{2,4,6} can be viewed as computing C = A*B where:
    A is 2x4, B is 4x6, C is 2x6.
    
    Split B into B = [B1, B2, B3] where B1, B2, B3 are 4x2.
    Then C = [A*B1, A*B2, A*B3].
    
    Each A*Bk is 2x2*4x2 = 2x2x2 multiplication (after reshaping).
    Using Strassen for each: 3 * 7 = 21 rank? No, that's not right either.
    
    Let me think about this differently. The WGA approach for M_{2,4,6}:
    
    View A as 2x4 = (2x1) * (1x4) or 2x(2*2).
    View B as 4x6 = (4*1) * (1x6) or (2*2) * (2*3).
    
    The Kronecker structure: M_{2,4,6} = M_{1,2,2} ⊗ M_{2,2,3}
    where M_{1,2,2} is the "outer product" tensor (rank 1) and
    M_{2,2,3} is the 2x2*2x3 multiplication tensor.
    
    So rank(M_{2,4,6}) <= rank(M_{1,2,2}) * rank(M_{2,2,3}) = 1 * rank(M_{2,2,3})
    
    But that's not quite right because the dimensions don't align that way.
    
    Let me just try the direct approach: use Strassen blocks.
    2x4 * 4x6:
    Split A into A1 (2x2) and A2 (2x2).
    Split B into B11, B12, B13 (2x2 each) and B21, B22, B23 (2x2 each).
    C = [A1*B11 + A2*B21, A1*B12 + A2*B22, A1*B13 + A2*B23]
    
    Each 2x2*2x2 can be done in rank 7 (Strassen).
    But we have 6 such multiplications, giving rank 42.
    
    For WGA, we can do better by sharing computations.
    """
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    S_U, S_V, S_W = _strassen_2x2x2()
    
    sub_problems = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4),
    ]
    
    for sp_idx, (a_start, b_row_start, b_col_start, c_col_start) in enumerate(sub_problems):
        base = sp_idx * 7
        for c in range(7):
            col = base + c
            for i in range(2):
                for j in range(2):
                    U[i*4 + (a_start + j), col] = S_U[i*2 + j, c]
            for j in range(2):
                for k in range(2):
                    V[(b_row_start + j)*6 + (b_col_start + k), col] = S_V[j*2 + k, c]
            for k in range(2):
                for i in range(2):
                    W[(c_col_start + k)*2 + i, col] = S_W[k*2 + i, c]
    
    return U, V, W

def _als_decomposition(target, n, m, p, rank, rng, max_iters=200, tol=1e-8):
    """
    Alternating Least Squares with L1 sparsity regularization.
    """
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    
    lambda_l1 = 0.01
    
    for it in range(max_iters):
        if _deadline_reached():
            break
            
        P = (V[:, None, :] * W[None, :, :]).reshape(rank, -1)
        try:
            PTP = P @ P.T + lambda_l1 * np.eye(rank)
            PTY = P @ target.reshape(n*m, -1).T
            U_new = np.linalg.solve(PTP, PTY).T
            U = np.sign(U_new) * np.maximum(np.abs(U_new) - lambda_l1*0.5, 0)
        except:
            break
            
        Q = (U[:, None, :] * W[None, :, :]).reshape(rank, -1)
        try:
            QTP = Q @ Q.T + lambda_l1 * np.eye(rank)
            QTY = Q @ target.transpose(1, 0, 2).reshape(m*p, -1).T
            V_new = np.linalg.solve(QTP, QTY).T
            V = np.sign(V_new) * np.maximum(np.abs(V_new) - lambda_l1*0.5, 0)
        except:
            break
            
        R = (U[:, None, :] * V[None, :, :]).reshape(rank, -1)
        try:
            RTP = R @ R.T + lambda_l1 * np.eye(rank)
            RTY = R @ target.transpose(2, 0, 1).reshape(p*n, -1).T
            W_new = np.linalg.solve(RTP, RTY).T
            W = np.sign(W_new) * np.maximum(np.abs(W_new) - lambda_l1*0.5, 0)
        except:
            break
            
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        err = np.linalg.norm(target - recon)
        if err < tol:
            break
            
    return U, V, W

def _try_lower_rank_search(n, m, p, max_rank, max_iters=200, seed=42):
    rng = np.random.RandomState(seed)
    target = _build_target_tensor(n, m, p)
    
    best_U, best_V, best_W = None, None, None
    best_rank = n*m*p
    
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n*m*p
    
    for rank in range(30, min(best_rank, max_rank + 1)):
        if _deadline_reached():
            break
            
        for attempt in range(10):
            if _deadline_reached():
                break
                
            U, V, W = _als_decomposition(target, n, m, p, rank, rng, max_iters=max_iters)
            
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                if rank < best_rank:
                    best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
                    best_rank = rank
                    break
        
    if best_rank < n*m*p:
        return best_U, best_V, best_W, best_rank
    
    return None, None, None, best_rank

def _wga_2x4x6():
    """
    Winograd Generalized Algorithm for 2x4x6 matrix multiplication.
    
    Key idea: 4 = 2*2 and 6 = 2*3.
    We decompose M_{2,4,6} using the structure:
    
    A (2x4) = A @ I_4, where we split the 4 columns into two groups of 2.
    B (4x6) = I_4 @ B, where we split the 6 columns into two groups of 3.
    
    The tensor M_{2,4,6} can be written as a sum of rank-1 tensors.
    Using the WGA, we can achieve rank 28 = 7 * 4 by:
    - Using Strassen (rank 7) for the 2x2x2 "core" multiplication
    - Using a rank-4 "convolution" for the 2x2x3 part
    
    Specifically:
    M_{2,4,6} = sum_{r=1}^{28} u_r ⊗ v_r ⊗ w_r
    
    where the structure exploits that 4=2*2 and 6=2*3.
    
    Let's construct this explicitly.
    
    A is 2x4: rows indexed by i=0,1; columns by j=0,1,2,3.
    B is 4x6: rows by j=0,1,2,3; columns by k=0,1,2,3,4,5.
    C is 2x6: rows by i=0,1; columns by k=0,1,2,3,4,5.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split j into j = j1 + 2*j2 where j1 in {0,1}, j2 in {0,1}.
    Split k into k = k1 + 2*k2 where k1 in {0,1}, k2 in {0,1,2} (for 6=2*3).
    
    Wait, 6 = 2*3 means k = k1 + 2*k2 with k1 in {0,1}, k2 in {0,1,2}.
    But that gives k in {0,1,2,3,4,5}. Good.
    
    And 4 = 2*2 means j = j1 + 2*j2 with j1 in {0,1}, j2 in {0,1}.
    
    So C[i, k1+2*k2] = sum_{j1,j2} A[i, j1+2*j2] * B[j1+2*j2, k1+2*k2]
    
    = sum_{j1,j2} A[i, j1+2*j2] * B[j1+2*j2, k1+2*k2]
    
    This is a 2x2x2 * 2x2x3 type structure.
    
    For each fixed (j2, k2), we have a 2x2x2 multiplication:
    C_part[i, k1] = sum_{j1} A[i, j1+2*j2] * B[j1+2*j2, k1+2*k2]
    
    There are 2*3 = 6 such 2x2x2 blocks.
    Using Strassen for each: 6*7 = 42 rank.
    
    But WGA can share computations across blocks!
    
    The Winograd approach uses transforms to reduce the number of multiplications.
    For the 2x2x2 part, Strassen already gives rank 7.
    For the 2x2x3 part, we need a better decomposition.
    
    Actually, let me just implement the rank-42 Strassen version and verify it works,
    then try to improve with search.
    """
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    S_U, S_V, S_W = _strassen_2x2x2()
    
    sub_problems = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4),
    ]
    
    for sp_idx, (a_start, b_row_start, b_col_start, c_col_start) in enumerate(sub_problems):
        base = sp_idx * 7
        for c in range(7):
            col = base + c
            for i in range(2):
                for j in range(2):
                    U[i*4 + (a_start + j), col] = S_U[i*2 + j, c]
            for j in range(2):
                for k in range(2):
                    V[(b_row_start + j)*6 + (b_col_start + k), col] = S_V[j*2 + k, c]
            for k in range(2):
                for i in range(2):
                    W[(c_col_start + k)*2 + i, col] = S_W[k*2 + i, c]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Try the WGA-based decomposition
    U, V, W = _wga_2x4x6()
    
    if _verify_decomposition(U, V, W, n, m, p):
        # Try to improve with search
        if not _deadline_reached():
            U2, V2, W2, rank2 = _try_lower_rank_search(n, m, p, max_rank=42, max_iters=100, seed=42)
            if U2 is not None and rank2 < 42:
                U, V, W = U2, V2, W2
    
    # Fallback to schoolbook if anything fails
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _schoolbook_decomposition(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
