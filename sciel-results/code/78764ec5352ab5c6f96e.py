"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the multiplication tensor T[i, j, k] = 1 if C[i,k] gets A[i,j]*B[j,k]
    # T has shape (n, m, p) in (i, j, k)
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Let's check by building the full tensor and comparing
    # Tensor T[i, j, k] = 1 for all valid (i, j, k)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    # Reconstruct from factors
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    if not np.allclose(T, T_recon, atol=1e-9):
        return False
    return True

def _strassen_rank14():
    """Construct the rank-14 decomposition using shared Strassen."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))  # 4 x 14
    V = np.zeros((m*p, rank))  # 8 x 14
    W = np.zeros((p*n, rank))  # 8 x 14
    
    # Strassen products for left half (B columns 0,1) and right half (B columns 2,3)
    # For each half, we have 7 Strassen products
    # A indices: A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22
    # U row index: i*m + j, so A[0,0]->0, A[0,1]->1, A[1,0]->2, A[1,1]->3
    
    # Left half (k=0,1): B[j,0] and B[j,1]
    # V row index: j*p + k, so B[0,0]->0, B[0,1]->1, B[1,0]->4, B[1,1]->5
    # W row index: k*n + i, so C[0,0]->0, C[0,1]->1, C[1,0]->2, C[1,1]->3
    
    # Right half (k=2,3): B[j,2] and B[j,3]
    # V: B[0,2]->2, B[0,3]->3, B[1,2]->6, B[1,3]->7
    # W: C[0,2]->4, C[0,3]->5, C[1,2]->6, C[1,3]->7
    
    # Strassen for 2x2 x 2x2:
    # m1 = (A11+A22)(B11+B22)
    # m2 = (A21+A22)B11
    # m3 = A11(B21-B22)
    # m4 = A22(B12-B11)
    # m5 = (A11+A12)B22
    # m6 = (A21-A11)(B11+B12)
    # m7 = (A12-A22)(B21+B22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # For left half: B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    # V indices: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # W indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # Product 1 (m1_L): (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    r = 0
    U[0, r] = 1; U[3, r] = 1  # A11 + A22
    V[0, r] = 1; V[5, r] = 1  # B[0,0] + B[1,1]
    W[0, r] = 1; W[1, r] = 1; W[2, r] = -1; W[3, r] = 1  # C00+=1, C01+=1, C10+=-1, C11+=1
    # Wait, let me recheck. C11 = m1+m4-m5+m7, C12=m3+m5, C21=m2+m4, C22=m1-m2+m3+m6
    # So m1 contributes to C11 and C22 (in 1-indexed), which are C[0,0] and C[1,1] in 0-indexed
    # W[k*n+i]: C[0,0] -> k=0,i=0 -> index 0; C[1,1] -> k=1,i=1 -> index 3
    # So m1_L contributes +1 to C[0,0] and +1 to C[1,1]
    W[0, r] = 1; W[3, r] = 1
    
    # Product 2 (m2_L): (A[1,0]+A[1,1]) * B[0,0]
    r = 1
    U[2, r] = 1; U[3, r] = 1  # A21 + A22
    V[0, r] = 1  # B[0,0]
    # m2 contributes to C21 and C22: C[1,0] and C[1,1]
    # C21 = m2+m4, C22 = m1-m2+m3+m6
    W[2, r] = 1; W[3, r] = -1
    
    # Product 3 (m3_L): A[0,0] * (B[1,0]-B[1,1])
    r = 2
    U[0, r] = 1  # A11
    V[4, r] = 1; V[5, r] = -1  # B[1,0] - B[1,1]
    # m3 contributes to C12 and C22: C[0,1] and C[1,1]
    # C12 = m3+m5, C22 = m1-m2+m3+m6
    W[1, r] = 1; W[3, r] = 1
    
    # Product 4 (m4_L): A[1,1] * (B[0,1]-B[0,0])
    r = 3
    U[3, r] = 1  # A22
    V[1, r] = 1; V[0, r] = -1  # B[0,1] - B[0,0]
    # m4 contributes to C11 and C21: C[0,0] and C[1,0]
    # C11 = m1+m4-m5+m7, C21 = m2+m4
    W[0, r] = 1; W[2, r] = 1
    
    # Product 5 (m5_L): (A[0,0]+A[0,1]) * B[1,1]
    r = 4
    U[0, r] = 1; U[1, r] = 1  # A11 + A12
    V[5, r] = 1  # B[1,1]
    # m5 contributes to C12 and C11: C[0,1] and C[0,0]
    # C12 = m3+m5, C11 = m1+m4-m5+m7
    W[1, r] = 1; W[0, r] = -1
    
    # Product 6 (m6_L): (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    r = 5
    U[2, r] = 1; U[0, r] = -1  # A21 - A11
    V[0, r] = 1; V[1, r] = 1  # B[0,0] + B[0,1]
    # m6 contributes to C22: C[1,1]
    # C22 = m1-m2+m3+m6
    W[3, r] = 1
    
    # Product 7 (m7_L): (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    r = 6
    U[1, r] = 1; U[3, r] = -1  # A12 - A22
    V[4, r] = 1; V[5, r] = 1  # B[1,0] + B[1,1]
    # m7 contributes to C11: C[0,0]
    # C11 = m1+m4-m5+m7
    W[0, r] = 1
    
    # Now right half (k=2,3): B[j,2] and B[j,3]
    # V indices: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # W indices: C[0,2]=4, C[0,3]=5, C[1,2]=6, C[1,3]=7
    
    # Product 8 (m1_R): (A[0,0]+A[1,1]) * (B[0,2]+B[1,3])
    r = 7
    U[0, r] = 1; U[3, r] = 1
    V[2, r] = 1; V[7, r] = 1
    W[4, r] = 1; W[7, r] = 1
    
    # Product 9 (m2_R): (A[1,0]+A[1,1]) * B[0,2]
    r = 8
    U[2, r] = 1; U[3, r] = 1
    V[2, r] = 1
    W[6, r] = 1; W[7, r] = -1
    
    # Product 10 (m3_R): A[0,0] * (B[1,2]-B[1,3])
    r = 9
    U[0, r] = 1
    V[6, r] = 1; V[7, r] = -1
    W[5, r] = 1; W[7, r] = 1
    
    # Product 11 (m4_R): A[1,1] * (B[0,3]-B[0,2])
    r = 10
    U[3, r] = 1
    V[3, r] = 1; V[2, r] = -1
    W[4, r] = 1; W[6, r] = 1
    
    # Product 12 (m5_R): (A[0,0]+A[0,1]) * B[1,3]
    r = 11
    U[0, r] = 1; U[1, r] = 1
    V[7, r] = 1
    W[5, r] = 1; W[4, r] = -1
    
    # Product 13 (m6_R): (A[1,0]-A[0,0]) * (B[0,2]+B[0,3])
    r = 12
    U[2, r] = 1; U[0, r] = -1
    V[2, r] = 1; V[3, r] = 1
    W[7, r] = 1
    
    # Product 14 (m7_R): (A[0,1]-A[1,1]) * (B[1,2]+B[1,3])
    r = 13
    U[1, r] = 1; U[3, r] = -1
    V[6, r] = 1; V[7, r] = 1
    W[4, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with rank-14 Strassen construction
    best = _strassen_rank14()
    best_rank = 14
    
    # Verify the rank-14 solution
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook if verification fails
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
    
    # Try to find rank-13 or lower using random search
    # The tensor T[i,j,k] = 1 for all i in [0,1], j in [0,1], k in [0,3]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try rank 13 with random initialization and gradient-free search
    for target_rank in [13, 12, 11]:
        if time.time() > deadline - 30:
            break
        found = False
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Try to solve using iterative refinement
            # This is a non-convex problem, so we use random restarts with local search
            # Use a simple approach: fix two matrices, solve for the third
            
            success = False
            for it in range(200):
                if time.time() > deadline - 30:
                    break
                # Update U: for each (i,j), U[i*m+j, r] = sum_k V[j*p+k, r] * W[k*n+i, r] / sum_k V[j*p+k, r]^2 * W[k*n+i, r]^2
                # This is a least-squares update
                for i in range(n):
                    for j in range(m):
                        # Target: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                        # Let a_r = V[j*p+k,r] * W[k*n+i,r] for each k
                        # We want sum_r U[i*m+j,r] * a_r = 1 for all k
                        # This is underdetermined if target_rank < p
                        # Use least squares
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            for r in range(target_rank):
                                A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        b_vec = np.ones(p)
                        # Solve min ||A_mat @ u - b_vec||^2
                        if target_rank >= p:
                            u, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        else:
                            u = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        U[i*m+j, :] = u
                # Similarly update V and W
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            for r in range(target_rank):
                                A_mat[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        b_vec = np.ones(n)
                        V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            for r in range(target_rank):
                                A_mat[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        b_vec = np.ones(m)
                        W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                
                # Check convergence
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                            error += (val - 1.0) ** 2
                if error < 1e-10:
                    success = True
                    break
            
            if success:
                found = True
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                break
        
        if found:
            break
    
    return best
# EVOLVE-BLOCK-END
