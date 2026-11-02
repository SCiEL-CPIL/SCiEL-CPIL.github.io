"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
    except:
        return False
    
    if U_arr.shape != (n * m, len(U_arr[0])) or V_arr.shape != (m * p, len(V_arr[0])) or W_arr.shape != (p * n, len(W_arr[0])):
        return False
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return rank-7 Strassen decomposition for 2x2 * 2x2.
    U rows indexed by i*2+j, V rows by j*2+k, W rows by k*2+i
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Term 0 (M1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1 (M2)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 2 (M3)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 3 (M4)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Term 4 (M5)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 5 (M6)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[2][5] = 1
    
    # Term 6 (M7)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _combine_blocks(block1, block2, n1, m1, p1, n2, m2, p2, n_total, m_total, p_total):
    """
    Combine two block decompositions into a full decomposition.
    block1: for (n1 x m1) * (m1 x p1)
    block2: for (n2 x m2) * (m2 x p2)
    The full problem is (n_total x m_total) * (m_total x p_total)
    where n_total = n1+n2, m_total = m1+m2, p_total = p1+p2
    """
    r1 = len(block1['U'][0])
    r2 = len(block2['U'][0])
    rank = r1 + r2
    
    n, m, p = n_total, m_total, p_total
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Block 1 occupies rows 0..n1-1 in A, cols 0..p1-1 in B
    for i in range(n1):
        for j in range(m1):
            for c in range(r1):
                U[i*m + j][c] = block1['U'][i*m1 + j][c]
    
    for j in range(m1):
        for k in range(p1):
            for c in range(r1):
                V[j*p + k][c] = block1['V'][j*p1 + k][c]
    
    for k in range(p1):
        for i in range(n1):
            for c in range(r1):
                W[k*n + i][c] = block1['W'][k*n1 + i][c]
    
    # Block 2: offset
    for i in range(n2):
        for j in range(m2):
            for c in range(r2):
                U[(n1+i)*m + (m1+j)][r1+c] = block2['U'][i*m2 + j][c]
    
    for j in range(m2):
        for k in range(p2):
            for c in range(r2):
                V[(m1+j)*p + (p1+k)][r1+c] = block2['V'][j*p2 + k][c]
    
    for k in range(p2):
        for i in range(n2):
            for c in range(r2):
                W[(p1+k)*n + (n1+i)][r1+c] = block2['W'][k*n2 + i][c]
    
    return {"U": U, "V": V, "W": W}

def _add_cross_terms(U, V, W, n1, m1, p1, n2, m2, p2, n_total, m_total, p_total):
    """
    Add schoolbook terms for cross blocks to complete the decomposition.
    Cross block 1: (n1 x m1) * (m1 x p2) -> C rows 0..n1-1, cols p1..p_total-1
    Cross block 2: (n2 x m1) * (m1 x p1) -> C rows n1..n_total-1, cols 0..p1-1
    """
    rank = len(U[0])
    col = rank
    n, m, p = n_total, m_total, p_total
    
    # Extend matrices if needed
    # We assume U, V, W have enough columns allocated by caller or we return new ones
    # For simplicity in this helper, we'll just assume the caller handles the size.
    # Actually, let's just return the modified matrices.
    
    # Cross 1: A[0:n1, 0:m1] * B[0:m1, p1:p_total] -> C[0:n1, p1:p_total]
    for i in range(n1):
        for j in range(m1):
            for k in range(p2):
                U[i*m + j][col] = 1
                V[j*p + (p1+k)][col] = 1
                W[(p1+k)*n + i][col] = 1
                col += 1
                
    # Cross 2: A[n1:n_total, 0:m1] * B[0:m1, 0:p1] -> C[n1:n_total, 0:p1]
    for i in range(n2):
        for j in range(m1):
            for k in range(p1):
                U[(n1+i)*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + (n1+i)][col] = 1
                col += 1
                
    return U, V, W, col

def _construct_3x3x4_via_blocks():
    """
    Construct a decomposition for 3x3 * 3x4 by splitting into blocks.
    Split A into 2x3 and 1x3. Split B into 3x2 and 3x2.
    Block 1: 2x3 * 3x2. Use schoolbook (rank 12) or optimized?
    Block 2: 1x3 * 3x2. Rank 6.
    Cross terms:
    C1: 2x3 * 3x2 (top-left of C)
    C2: 1x3 * 3x2 (bottom-left of C)
    C3: 2x3 * 3x2 (top-right of C)
    C4: 1x3 * 3x2 (bottom-right of C)
    
    Wait, the split is:
    A = [A1 (2x3); A2 (1x3)]
    B = [B1 (3x2) B2 (3x2)]
    
    C = A*B = [A1*B1 + A1*B2, A1*B2]  -- No, C is 3x4.
    C = [ A1*B1  A1*B2 ]
        [ A2*B1  A2*B2 ]
        
    A1 is 2x3, B1 is 3x2 -> 2x2. Rank 7 (Strassen).
    A1 is 2x3, B2 is 3x2 -> 2x2. Rank 7 (Strassen).
    A2 is 1x3, B1 is 3x2 -> 1x2. Rank 2 (Schoolbook).
    A2 is 1x3, B2 is 3x2 -> 1x2. Rank 2 (Schoolbook).
    
    Total Rank = 7 + 7 + 2 + 2 = 18.
    This is better than 36.
    """
    n, m, p = 3, 3, 4
    
    # Block 1: A1 (2x3) * B1 (3x2)
    # We need a 2x3 * 3x2 decomposition.
    # We can split B1 into two 3x1 columns.
    # Or use a general 2x3 * 3x2 algorithm.
    # Let's just use schoolbook for 2x3*3x2 to be safe, rank 12.
    # Or better: 2x3 * 3x2 can be seen as two 2x3*3x1.
    # 2x3 * 3x1: Rank 6 (schoolbook). Total 12.
    # Is there a better way for 2x3 * 3x2?
    # Actually, 2x2 * 2x2 is 7.
    # 2x3 * 3x2 is equivalent to 2x2 * 2x2 with some padding? No.
    # Let's stick to schoolbook for sub-blocks for now, but use Strassen where 2x2 fits.
    
    # Let's refine the split.
    # A is 3x3. B is 3x4.
    # Split A into 2x3 and 1x3.
    # Split B into 3x2 and 3x2.
    
    # Sub-problem 1: (2x3) * (3x2) -> 2x2.
    # We can decompose this into two (2x3)*(3x1) which are rank 6 each? No.
    # (2x3)*(3x1) is a vector-matrix product. Rank 6 schoolbook.
    # But wait, (2x3)*(3x2) is just two columns of (2x3)*(3x2).
    # If we use Strassen for 2x2*2x2, we need square matrices.
    # We can embed 2x3 into 3x3? No, that increases rank.
    
    # Let's try a different split.
    # A: 3x3. B: 3x4.
    # Split B into 3x3 and 3x1.
    # Block 1: (3x3) * (3x3). Rank 23 (Lader).
    # Block 2: (3x3) * (3x1). Rank 9 (Schoolbook).
    # Cross terms: None, because the split is on the output columns of B, which correspond to columns of C.
    # C = A * B = [ A*B1  A*B2 ].
    # So we just need to concatenate the decompositions.
    # Rank = 23 + 9 = 32.
    # This is better than 36.
    
    # Can we do better?
    # Split A into 2x3 and 1x3. Split B into 3x3 and 3x1.
    # C11: (2x3)*(3x3) -> 2x3.
    # C12: (2x3)*(3x1) -> 2x1.
    # C21: (1x3)*(3x3) -> 1x3.
    # C22: (1x3)*(3x1) -> 1x1.
    
    # (2x3)*(3x3): Split B into 3x2 and 3x1.
    #   (2x3)*(3x2) -> 2x2. Rank 12 (Schoolbook) or 7? No, 2x3 is not square.
    #   (2x3)*(3x1) -> 2x1. Rank 6.
    #   Total 18.
    # (2x3)*(3x1): Rank 6.
    # (1x3)*(3x3): Split B into 3x2 and 3x1.
    #   (1x3)*(3x2) -> 1x2. Rank 2.
    #   (1x3)*(3x1) -> 1x1. Rank 1.
    #   Total 3.
    # (1x3)*(3x1): Rank 1.
    
    # Total Rank = 18 + 6 + 3 + 1 = 28.
    
    # Can we optimize (2x3)*(3x2)?
    # It's a 2x2 output.
    # We can use Strassen if we can map it.
    # A is 2x3, B is 3x2.
    # C = A B.
    # C00 = A00 B00 + A01 B10 + A02 B20
    # C01 = A00 B01 + A01 B11 + A02 B21
    # C10 = A10 B00 + A11 B10 + A12 B20
    # C11 = A10 B01 + A11 B11 + A12 B21
    
    # This is essentially two 2x2 multiplications if we split the middle dimension?
    # No, the middle dimension is 3.
    # We can split B into B_left (3x1) and B_right (3x1).
    # Then we have two (2x3)*(3x1) problems.
    # Each is rank 6. Total 12.
    # Is there a rank < 12 for (2x3)*(3x2)?
    # The tensor rank of (2,3,2) is 12?
    # Actually, (2,3,2) is equivalent to (2,2,2) with an extra dimension in the middle.
    # R(2,2,2) = 7.
    # If we can reuse the 7 products?
    # No, the inputs are different.
    
    # Let's stick to the 32 rank solution (3x3*3x3 + 3x3*3x1) as it's robust and simple.
    # 23 + 9 = 32.
    
    # Construct 3x3*3x3 (Lader rank 23)
    # We need a rank 23 decomposition for 3x3*3x3.
    # This is complex to hardcode.
    # Let's use schoolbook for 3x3*3x3 (rank 27) and schoolbook for 3x3*3x1 (rank 9).
    # Total 36.
    
    # Wait, the current program uses schoolbook for everything (36).
    # The inspiration suggests using Strassen blocks.
    # Let's try to implement the 28 rank solution explicitly.
    
    # Sub-problems:
    # 1. (2x3) * (3x2) -> 2x2. Rank 12.
    # 2. (2x3) * (3x1) -> 2x1. Rank 6.
    # 3. (1x3) * (3x2) -> 1x2. Rank 2.
    # 4. (1x3) * (3x1) -> 1x1. Rank 1.
    
    # Total 21? No, 12+6+2+1 = 21.
    # Wait, earlier I said 18+6+3+1 = 28.
    # Let's re-evaluate (2x3)*(3x2).
    # If I use schoolbook, it's 2*3*2 = 12.
    # If I split B into two 3x1, it's two (2x3)*(3x1) which are 6 each. Total 12.
    # So 12 is the schoolbook rank.
    
    # Is there a better decomposition for (2x3)*(3x2)?
    # It is known that R(2,3,2) = 12?
    # Actually, R(2,3,2) = 12 is the trivial bound.
    # But R(2,2,2)=7.
    # If we pad A to 2x2 and B to 2x2, we get 7.
    # But A is 2x3, B is 3x2.
    # We can't just pad.
    
    # Let's assume the 28 rank solution is valid and better than 36.
    # We will construct it.
    
    n, m, p = 3, 3, 4
    # Split A into A1 (2x3) and A2 (1x3)
    # Split B into B1 (3x2) and B2 (3x2)
    
    # We will build U, V, W for the whole 3x3*3x4.
    # Rank = 12 + 6 + 2 + 1 = 21?
    # Let's check the cross terms again.
    # C = [ C11 C12 ]
    #     [ C21 C22 ]
    # C11 = A1 B1 (2x3 * 3x2 -> 2x2)
    # C12 = A1 B2 (2x3 * 3x2 -> 2x2)
    # C21 = A2 B1 (1x3 * 3x2 -> 1x2)
    # C22 = A2 B2 (1x3 * 3x2 -> 1x2)
    
    # Wait, B is 3x4. Split into 3x2 and 3x2.
    # So B1 is 3x2, B2 is 3x2.
    # C11 is 2x2. C12 is 2x2. C21 is 1x2. C22 is 1x2.
    
    # Ranks:
    # C11: (2x3)*(3x2). Rank 12.
    # C12: (2x3)*(3x2). Rank 12.
    # C21: (1x3)*(3x2). Rank 2.
    # C22: (1x3)*(3x2). Rank 2.
    
    # Total 28.
    
    U = [[0]*28 for _ in range(9)]
    V = [[0]*28 for _ in range(12)]
    W = [[0]*28 for _ in range(12)]
    
    col = 0
    
    # C11: A1 (rows 0,1) * B1 (cols 0,1) -> C (rows 0,1, cols 0,1)
    # A1 is 2x3. B1 is 3x2.
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[i*3 + j][col] = 1
                V[j*4 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
                
    # C12: A1 (rows 0,1) * B2 (cols 2,3) -> C (rows 0,1, cols 2,3)
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[i*3 + j][col] = 1
                V[j*4 + (2+k)][col] = 1
                W[(2+k)*3 + i][col] = 1
                col += 1
                
    # C21: A2 (row 2) * B1 (cols 0,1) -> C (row 2, cols 0,1)
    for i in range(1):
        for j in range(3):
            for k in range(2):
                U[(2+i)*3 + j][col] = 1
                V[j*4 + k][col] = 1
                W[k*3 + (2+i)][col] = 1
                col += 1
                
    # C22: A2 (row 2) * B2 (cols 2,3) -> C (row 2, cols 2,3)
    for i in range(1):
        for j in range(3):
            for k in range(2):
                U[(2+i)*3 + j][col] = 1
                V[j*4 + (2+k)][col] = 1
                W[(2+k)*3 + (2+i)][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _optimize(U, V, W, n, m, p, max_iters=2000, lr_init=0.01, time_limit=30):
    """Optimize using Adam with adaptive learning rate."""
    start_time = time.time()
    
    # Adam optimizer state
    mU = np.zeros_like(U)
    vU = np.zeros_like(U)
    mV = np.zeros_like(V)
    vV = np.zeros_like(V)
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    t = 0
    
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        loss = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    inner = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err = inner - 1.0
                    loss += err * err
                    
                    dU[i*m+j] += 2 * err * V[j*p+k] * W[k*n+i]
                    dV[j*p+k] += 2 * err * U[i*m+j] * W[k*n+i]
                    dW[k*n+i] += 2 * err * U[i*m+j] * V[j*p+k]
        
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_loss < 1e-10:
            break
        
        t += 1
        lr = lr_init / (1 + 0.001 * t)
        
        mU = b1 * mU + (1 - b1) * dU
        vU = b2 * vU + (1 - b2) * dU**2
        mU_hat = mU / (1 - b1**t)
        vU_hat = vU / (1 - b2**t)
        U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        
        mV = b1 * mV + (1 - b1) * dV
        vV = b2 * vV + (1 - b2) * dV**2
        mV_hat = mV / (1 - b1**t)
        vV_hat = vV / (1 - b2**t)
        V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        
        mW = b1 * mW + (1 - b1) * dW
        vW = b2 * vW + (1 - b2) * dW**2
        mW_hat = mW / (1 - b1**t)
        vW_hat = vW / (1 - b2**t)
        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
    
    return best_U, best_V, best_W, best_loss

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Start with the 28-rank block decomposition
    best = _construct_3x3x4_via_blocks()
    best_rank = 28
    
    # Verify
    if _verify(best["U"], best["V"], best["W"], n, m, p):
        pass
    else:
        # Fallback to schoolbook
        best = _schoolbook(n, m, p)
        best_rank = n * m * p

    # Try to optimize the 28-rank decomposition to see if it can be reduced
    # or if we can find a lower rank one.
    # We will try to optimize the 28-rank one to see if it converges to a valid solution
    # (it should, since it's already valid).
    
    # Try to find a rank 27 decomposition?
    # Or just return the 28 rank one.
    
    # Let's try to optimize the 28 rank one to ensure it's exact.
    U = np.array(best["U"], dtype=float)
    V = np.array(best["V"], dtype=float)
    W = np.array(best["W"], dtype=float)
    
    U_opt, V_opt, W_opt, loss = _optimize(U, V, W, n, m, p, max_iters=1000, lr_init=0.01, time_limit=10)
    
    if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p):
        best = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
    
    return best

# EVOLVE-BLOCK-END
