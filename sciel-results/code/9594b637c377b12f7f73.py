"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
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
    return U, V, W

def _strassen_2x2():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication with rank 7 (Strassen).
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # Products:
    # p1 = (a11 + a22)(b11 + b22)
    # p2 = (a21 + a22)b11
    # p3 = a11(b12 - b22)
    # p4 = a22(b21 - b11)
    # p5 = (a11 + a12)b22
    # p6 = (a21 - a11)(b11 + b12)
    # p7 = (a12 - a22)(b21 + b22)
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    # We need T[i,j,k] = 1 for all i,j,k in {0,1}
    # The tensor decomposition: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    
    # Let's construct this carefully.
    # For each product p_r, we have a rank-1 term.
    # p1 = (a11 + a22)(b11 + b22)
    #   U contribution: for (i=0,j=0): +1, (i=1,j=1): +1
    #   V contribution: for (j=0,k=0): +1, (j=1,k=1): +1
    #   W contribution: for (k=0,i=0): +1, (k=1,i=1): +1
    #   But wait, the product gives (a11+a22)(b11+b22) = a11b11 + a11b22 + a22b11 + a22b22
    #   This contributes to C11, C12, C21, C22.
    
    # Let me think differently. The tensor M_{2,2,2} has entries M[i,j,k] = 1 for all i,j,k.
    # We want to decompose it as sum of 7 rank-1 tensors.
    
    # Actually, let me use a known decomposition.
    # The Strassen decomposition for 2x2x2:
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Product 1: (a11+a22)(b11+b22)
    # U[0,0]=1, U[3,0]=1 (i=0,j=0 and i=1,j=1)
    # V[0,0]=1, V[3,0]=1 (j=0,k=0 and j=1,k=1)
    # W[0,0]=1, W[3,0]=1 (k=0,i=0 and k=1,i=1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Product 2: (a21+a22)b11
    # U[2,1]=1, U[3,1]=1 (i=1,j=0 and i=1,j=1)
    # V[0,1]=1 (j=0,k=0)
    # W[0,1]=1 (k=0,i=0)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][1] = 1
    
    # Product 3: a11(b12-b22)
    # U[0,2]=1 (i=0,j=0)
    # V[1,2]=1, V[3,2]=-1 (j=0,k=1 and j=1,k=1)
    # W[1,2]=1 (k=1,i=0)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Product 4: a22(b21-b11)
    # U[3,3]=1 (i=1,j=1)
    # V[2,3]=1, V[0,3]=-1 (j=1,k=0 and j=0,k=0)
    # W[2,3]=1 (k=0,i=1)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1
    
    # Product 5: (a11+a12)b22
    # U[0,4]=1, U[1,4]=1 (i=0,j=0 and i=0,j=1)
    # V[3,4]=1 (j=1,k=1)
    # W[3,4]=1 (k=1,i=1)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[3][4] = 1
    
    # Product 6: (a21-a11)(b11+b12)
    # U[2,5]=1, U[0,5]=-1 (i=1,j=0 and i=0,j=0)
    # V[0,5]=1, V[1,5]=1 (j=0,k=0 and j=0,k=1)
    # W[0,5]=1 (k=0,i=0)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1
    
    # Product 7: (a12-a22)(b21+b22)
    # U[1,6]=1, U[3,6]=-1 (i=0,j=1 and i=1,j=1)
    # V[2,6]=1, V[3,6]=1 (j=1,k=0 and j=1,k=1)
    # W[2,6]=1 (k=0,i=1)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[2][6] = 1
    
    return U, V, W

def _build_3x4x6_from_strassen():
    """
    Build a decomposition of 3x4x6 using Strassen for 2x2 sub-blocks.
    
    Strategy:
    - Split the 4x4 middle into 2x2 blocks: positions (0,0), (0,1), (1,0), (1,1)
    - For the 3x3 output, we can use a 3x3 decomposition or combine 2x2 Strassen blocks.
    
    Actually, let's use a different approach:
    - Use the fact that 3x4x6 can be decomposed using known results.
    - Rank of 3x4x6 is known to be at most 66 (or similar).
    
    Let me try a constructive approach:
    - Split n=3 into 2+1, m=4 into 2+2, p=6 into 2+2+2
    - Use Strassen for the 2x2x2 parts
    - Use schoolbook for the remaining parts
    
    This is getting complex. Let me try a simpler approach:
    Use the known rank-66 decomposition or try to find one via optimization.
    """
    pass

def _try_rank(n, m, p, rank, max_iter=1000, tol=1e-8, seed=0):
    """Try to find a decomposition of given rank using alternating minimization."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    np.random.seed(seed)
    
    # Initialize randomly
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Alternating minimization
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            return None
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                try:
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        B[i, r] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    x = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        C[j, r] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Check convergence
        if it % 5 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        err = max(err, abs(s - 1.0))
            if err < tol:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                # Round to nearest half-integer if close
                for row in U_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                for row in V_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                for row in W_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                return U_list, V_list, W_list
    
    return None

def _construct_strassen_based():
    """
    Construct a decomposition of 3x4x6 using Strassen for 2x2x2 sub-blocks.
    
    Approach:
    - Decompose the 3x4x6 tensor into blocks.
    - Use Strassen (rank 7) for 2x2x2 sub-tensors.
    - Use schoolbook for remaining parts.
    
    Let's split:
    n=3: rows 0,1,2
    m=4: cols 0,1,2,3
    p=6: cols 0,1,2,3,4,5
    
    We can think of this as:
    - A 2x2x2 block for (n:0,1; m:0,1; p:0,1) -> Strassen rank 7
    - A 2x2x2 block for (n:0,1; m:2,3; p:0,1) -> Strassen rank 7
    - A 2x2x2 block for (n:0,1; m:0,1; p:2,3) -> Strassen rank 7
    - A 2x2x2 block for (n:0,1; m:2,3; p:2,3) -> Strassen rank 7
    - A 2x2x2 block for (n:0,1; m:0,1; p:4,5) -> Strassen rank 7
    - A 2x2x2 block for (n:0,1; m:2,3; p:4,5) -> Strassen rank 7
    - For n=2 (row 2), we need 1x4x6 -> schoolbook rank 24
    
    Total: 6*7 + 24 = 42 + 24 = 66
    
    But wait, this isn't quite right because the Strassen blocks don't directly compose this way.
    The tensor is not simply a direct sum of 2x2x2 blocks.
    
    Let me reconsider. The matrix multiplication tensor M_{3,4,6} has entries:
    M[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    
    We want to write this as a sum of rank-1 tensors.
    
    Actually, I think the key insight is that we can use a block structure.
    Let me try a different approach: use known decompositions.
    
    For now, let's return a decomposition of rank 66 using a combination of Strassen and schoolbook.
    """
    n, m, p = 3, 4, 6
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Use Strassen for the 2x2x2 blocks where i in {0,1}, j in {0,1} or {2,3}, k in {0,1}, {2,3}, or {4,5}
    # There are 2 (j-groups) * 3 (k-groups) = 6 such blocks, each with rank 7.
    # For i=2, use schoolbook: 4*6 = 24 terms.
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2()
    
    for j_group in range(2):  # j in {0,1} or {2,3}
        j_base = j_group * 2
        for k_group in range(3):  # k in {0,1}, {2,3}, or {4,5}
            k_base = k_group * 2
            
            for r in range(7):
                # U: i in {0,1}, j in {j_base, j_base+1}
                for i in range(2):
                    for jj in range(2):
                        U[i * m + j_base + jj][col] = strassen_U[i * 2 + jj][r]
                
                # V: j in {j_base, j_base+1}, k in {k_base, k_base+1}
                for jj in range(2):
                    for kk in range(2):
                        V[(j_base + jj) * p + k_base + kk][col] = strassen_V[jj * 2 + kk][r]
                
                # W: k in {k_base, k_base+1}, i in {0,1}
                for kk in range(2):
                    for i in range(2):
                        W[(k_base + kk) * n + i][col] = strassen_W[kk * 2 + i][r]
                
                col += 1
    
    # For i=2, use schoolbook
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = n * m * p
    
    # Strategy 1: Try the Strassen-based construction of rank 66
    try:
        U, V, W = _construct_strassen_based()
        if _verify(U, V, W, n, m, p):
            best_rank = 66
            best_result = {"U": U, "V": V, "W": W}
    except:
        pass
    
    # Strategy 2: Try to find a better decomposition via optimization
    if time.time() < _DEADLINE - 10.0:
        for rank in range(55, min(66, best_rank)):
            if time.time() > _DEADLINE - 5.0:
                break
            for seed in range(5):
                if time.time() > _DEADLINE - 5.0:
                    break
                result = _try_rank(n, m, p, rank, max_iter=200, seed=seed)
                if result is not None:
                    U, V, W = result
                    if _verify(U, V, W, n, m, p):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {"U": U, "V": V, "W": W}
                        break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
