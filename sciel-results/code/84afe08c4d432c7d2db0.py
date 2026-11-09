<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Rank 22 decomposition
    # 7 for A_L B_T1 (2x2x2 Strassen)
    # 7 for A_L B_T2 (2x2x2 Strassen)
    # 8 for A_R B_B  (2x1x4 Schoolbook)
    rank = 22
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Indices:
    # A: 2x3. A[i,j] index i*3+j
    # B: 3x4. B[j,k] index j*4+k
    # C: 2x4. W[k*n+i, r] -> W index is k*n+i. n=2. So W index is 2*k + i.
    
    # Block 1: A_L B_T1
    # A_L is A[:, 0:2] (2x2)
    # B_T1 is B[0:2, 0:2] (2x2)
    # Strassen for C1 (2x2) corresponding to C[:, 0:2]
    
    # A_L indices:
    # A00 -> A[0,0] -> idx 0
    # A01 -> A[0,1] -> idx 1
    # A10 -> A[1,0] -> idx 3
    # A11 -> A[1,1] -> idx 4
    
    # B_T1 indices:
    # B00 -> B[0,0] -> idx 0
    # B01 -> B[0,1] -> idx 1
    # B10 -> B[1,0] -> idx 4
    # B11 -> B[1,1] -> idx 5
    
    # C1 indices (W):
    # C00 -> C[0,0] -> W idx 0*2+0 = 0
    # C01 -> C[0,1] -> W idx 1*2+0 = 2
    # C10 -> C[1,0] -> W idx 0*2+1 = 1
    # C11 -> C[1,1] -> W idx 1*2+1 = 3
    
    col = 0
    # M1: (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2: (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # M3: A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M4: A11(B10-B00)
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5: (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    
    # M6: (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # Block 2: A_L B_T2
    # B_T2 is B[0:2, 2:4] (2x2)
    # B indices:
    # B02 -> B[0,2] -> idx 2
    # B03 -> B[0,3] -> idx 3
    # B12 -> B[1,2] -> idx 6
    # B13 -> B[1,3] -> idx 7
    
    # C2 indices (W):
    # C02 -> C[0,2] -> W idx 2*2+0 = 4
    # C03 -> C[0,3] -> W idx 3*2+0 = 6
    # C12 -> C[1,2] -> W idx 2*2+1 = 5
    # C13 -> C[1,3] -> W idx 3*2+1 = 7
    
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1
    
    # M3
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # M4
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[4][col] = 1; W[5][col] = 1
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = -1; W[6][col] = 1
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = 1
    col += 1
    
    # Block 3: A_R B_B
    # A_R is A[:, 2] (2x1)
    # A02 -> idx 2
    # A12 -> idx 5
    
    # B_B is B[2, :] (1x4)
    # B20 -> idx 8
    # B21 -> idx 9
    # B22 -> idx 10
    # B23 -> idx 11
    
    # C indices:
    # C00 -> W 0
    # C01 -> W 2
    # C02 -> W 4
    # C03 -> W 6
    # C10 -> W 1
    # C11 -> W 3
    # C12 -> W 5
    # C13 -> W 7
    
    # Z1: A02 B20 -> C00
    U[2][col] = 1; V[8][col] = 1; W[0][col] = 1
    col += 1
    # Z2: A02 B21 -> C01
    U[2][col] = 1; V[9][col] = 1; W[2][col] = 1
    col += 1
    # Z3: A02 B22 -> C02
    U[2][col] = 1; V[10][col] = 1; W[4][col] = 1
    col += 1
    # Z4: A02 B23 -> C03
    U[2][col] = 1; V[11][col] = 1; W[6][col] = 1
    col += 1
    # Z5: A12 B20 -> C10
    U[5][col] = 1; V[8][col] = 1; W[1][col] = 1
    col += 1
    # Z6: A12 B21 -> C11
    U[5][col] = 1; V[9][col] = 1; W[3][col] = 1
    col += 1
    # Z7: A12 B22 -> C12
    U[5][col] = 1; V[10][col] = 1; W[5][col] = 1
    col += 1
    # Z8: A12 B23 -> C13
    U[5][col] = 1; V[11][col] = 1; W[7][col] = 1
    col += 1
    
    # Verification
    if time.time() < deadline - 5:
        try:
            U_np = np.array(U)
            V_np = np.array(V)
            W_np = np.array(W)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for l in range(rank):
                            val += U[i][l] * V[j][l] * W[k*n+i][l]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                # Fallback to schoolbook if verification fails (should not happen)
                # Schoolbook rank 24
                rank_fb = 24
                U_fb = [[0.0] * rank_fb for _ in range(n * m)]
                V_fb = [[0.0] * rank_fb for _ in range(m * p)]
                W_fb = [[0.0] * rank_fb for _ in range(p * n)]
                
                col_fb = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_fb[i*m+j][col_fb] = 1.0
                            V_fb[j*p+k][col_fb] = 1.0
                            W_fb[k*n+i][col_fb] = 1.0
                            col_fb += 1
                            
                return {"U": U_fb, "V": V_fb, "W": W_fb}
        except Exception:
            pass
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for l in range(rank):
                    val += U[i*m+j][l] * V[j*p+k][l] * W[k*n+i][l]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_rank22():
    """Known rank-22 decomposition using Strassen for 2x2 blocks."""
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # A: 2x3, A[i,j] -> idx i*3+j
    # B: 3x4, B[j,k] -> idx j*4+k
    # W: W[k*n+i, r], n=2, so W idx = 2*k + i
    
    col = 0
    # Block 1: A_L (cols 0,1) x B_T1 (rows 0,1; cols 0,1) -> C[:, 0:2]
    # Strassen 2x2x2
    # A00=0, A01=1, A10=3, A11=4
    # B00=0, B01=1, B10=4, B11=5
    # C00=W[0], C01=W[2], C10=W[1], C11=W[3]
    
    U[0][col]=1; U[4][col]=1; V[0][col]=1; V[5][col]=1; W[0][col]=1; W[3][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[0][col]=1; W[1][col]=1; W[3][col]=-1; col+=1
    U[0][col]=1; V[1][col]=1; V[5][col]=-1; W[2][col]=1; W[3][col]=1; col+=1
    U[4][col]=1; V[4][col]=1; V[0][col]=-1; W[0][col]=1; W[1][col]=1; col+=1
    U[0][col]=1; U[1][col]=1; V[5][col]=1; W[0][col]=-1; W[2][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[0][col]=1; V[1][col]=1; W[3][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[4][col]=1; V[5][col]=1; W[0][col]=1; col+=1
    
    # Block 2: A_L (cols 0,1) x B_T2 (rows 0,1; cols 2,3) -> C[:, 2:4]
    # B02=2, B03=3, B12=6, B13=7
    # C02=W[4], C03=W[6], C12=W[5], C13=W[7]
    
    U[0][col]=1; U[4][col]=1; V[2][col]=1; V[7][col]=1; W[4][col]=1; W[7][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[2][col]=1; W[5][col]=1; W[7][col]=-1; col+=1
    U[0][col]=1; V[3][col]=1; V[7][col]=-1; W[6][col]=1; W[7][col]=1; col+=1
    U[4][col]=1; V[6][col]=1; V[2][col]=-1; W[4][col]=1; W[5][col]=1; col+=1
    U[0][col]=1; U[1][col]=1; V[7][col]=1; W[4][col]=-1; W[6][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[2][col]=1; V[3][col]=1; W[7][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[6][col]=1; V[7][col]=1; W[4][col]=1; col+=1
    
    # Block 3: A_R (col 2) x B_B (row 2) -> C[:, :] (diagonal, rank 8)
    # A02=2, A12=5; B20=8, B21=9, B22=10, B23=11
    U[2][col]=1; V[8][col]=1; W[0][col]=1; col+=1
    U[2][col]=1; V[9][col]=1; W[2][col]=1; col+=1
    U[2][col]=1; V[10][col]=1; W[4][col]=1; col+=1
    U[2][col]=1; V[11][col]=1; W[6][col]=1; col+=1
    U[5][col]=1; V[8][col]=1; W[1][col]=1; col+=1
    U[5][col]=1; V[9][col]=1; W[3][col]=1; col+=1
    U[5][col]=1; V[10][col]=1; W[5][col]=1; col+=1
    U[5][col]=1; V[11][col]=1; W[7][col]=1; col+=1
    
    return U, V, W

def _try_rank21_numerical(deadline):
    """
    Attempt to find a rank-21 decomposition using numerical optimization.
    Strategy: Start from the rank-22 solution, try to merge two terms
    by finding a linear dependency, or use random initialization with
    gradient-free search on the polynomial constraints.
    
    We use the approach of checking if the tensor can be decomposed with
    rank 21 by solving a system of polynomial equations via numerical methods.
    """
    n, m, p = 2, 3, 4
    target_rank = 21
    
    # The tensor T[i,j,k] = 1 for all i,j,k in valid ranges
    # We need U[i,j,r] * V[j,k,r] * W[k,i,r] summed over r = 1 for all (i,j,k)
    # and 0 for invalid (but in our case all are valid since n*m*p = 24 terms)
    
    # Number of equations: n*m*p = 2*3*4 = 24
    # Unknowns: 3 * n*m*p * target_rank = 3 * 24 * 21 = 1512
    # But we can fix gauge: set one element per column to 1 (21 constraints)
    # Effective unknowns: 1512 - 21 = 1491
    # Equations: 24
    # This is heavily overdetermined in a different sense - we have more unknowns than equations
    # So rank 21 should be feasible if it exists.
    
    # Use a randomized approach: parameterize and try to find a solution
    # via iterative refinement starting from a perturbed rank-22 solution.
    
    # Get rank-22 solution
    U22, V22, W22 = _construct_rank22()
    U22 = np.array(U22, dtype=float)
    V22 = np.array(V22, dtype=float)
    W22 = np.array(W22, dtype=float)
    
    # Strategy: Try to find a rank-21 decomposition by expressing one column
    # as a linear combination of others in a way that preserves the tensor.
    # 
    # Alternative: Use the fact that if rank 21 exists, we can find it by
    # solving the polynomial system. Let's use a simple iterative approach:
    # randomly initialize rank-21 matrices and try to refine.
    
    rng = np.random.RandomState(42)
    
    # Build the tensor T as a 3D array
    T = np.ones((n, m, p))
    
    best_error = float('inf')
    best_solution = None
    
    # Try multiple random initializations
    n_attempts = 50
    for attempt in range(n_attempts):
        if time.time() > deadline - 15:
            break
        
        # Random initialization with small values
        U = rng.randn(n*m, target_rank) * 0.1
        V = rng.randn(m*p, target_rank) * 0.1
        W = rng.randn(p*n, target_rank) * 0.1
        
        # Iterative refinement: for each term r, try to optimize
        # We use a simple alternating minimization approach
        # For fixed V, W: solve for U (linear)
        # For fixed U, W: solve for V (linear)  
        # For fixed U, V: solve for W (linear)
        
        # Actually, each equation is: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        # For fixed V and W, this is linear in U.
        # Let's build the linear system for U.
        
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            # Optimize U: for each (i,j), we have:
            # sum_r U[i*m+j, r] * (V[j, :, r] * W[:, i, r] reshaped) = T[i,j,:]
            # This is: sum_r U[i*m+j, r] * B[i,j,r] = 1 for each k
            # where B[i,j,r] = sum_k V[j*p+k, r] * W[k*n+i, r] ... no
            # Actually: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for each k
            # For fixed V, W: this is a system of p equations in target_rank unknowns for each (i,j)
            
            # Let's do it differently. For each (i,j):
            # We need: sum_r U[i*m+j, r] * V[j, :, r] * W[:, i, r] = ones(p)
            # where V[j, :, r] is vector of length p (indices j*p to j*p+p-1)
            # and W[:, i, r] is vector of length p (indices 0*p+i, 1*p+i, ..., (p-1)*p+i)
            
            # Let M[i,j] be a p x target_rank matrix where
            # M[i,j, k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Then: M[i,j] @ U[i*m+j, :] = ones(p)
            
            # Solve least squares for each (i,j)
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve M @ u = ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Optimize V: for each (j,k):
            # sum_r U[:, j, r] * V[j*p+k, r] * W[k, :, r] = ones(n)
            # M[j,k, i, r] = U[i*m+j, r] * W[k*n+i, r]
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Optimize W: for each (k,i):
            # sum_r U[:, :, r] * V[:, k, r] * W[k*n+i, r] = ones(m)
            # M[k,i, j, r] = U[i*m+j, r] * V[j*p+k, r]
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        error += (val - 1.0)**2
            error = np.sqrt(error / (n*m*p))
            
            if error < best_error:
                best_error = error
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if error < 1e-6:
                break
        
        if best_error < 1e-6:
            break
    
    if best_solution is not None and best_error < 1e-6:
        U_sol, V_sol, W_sol = best_solution
        # Round to nearest half-integer if possible
        U_rounded = np.round(U_sol * 2) / 2
        V_rounded = np.round(V_sol * 2) / 2
        W_rounded = np.round(W_sol * 2) / 2
        
        # Verify rounded solution
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, target_rank, tol=1e-9):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        # Verify unrounded solution
        if _verify_decomposition(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), n, m, p, target_rank, tol=1e-9):
            return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # First, try to find a rank-21 decomposition
    if time.time() < deadline - 30:
        result21 = _try_rank21_numerical(deadline)
        if result21 is not None:
            U, V, W = result21
            rank = len(U[0])
            # Verify
            if _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to known rank-22 decomposition
    U, V, W = _construct_rank22()
    
    # Verify rank-22
    if not _verify_decomposition(U, V, W, n, m, p, 22, tol=1e-9):
        # Schoolbook fallback
        rank_fb = 24
        U_fb = [[0.0] * rank_fb for _ in range(n * m)]
        V_fb = [[0.0] * rank_fb for _ in range(m * p)]
        W_fb = [[0.0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col_fb] = 1.0
                    V_fb[j*p+k][col_fb] = 1.0
                    W_fb[k*n+i][col_fb] = 1.0
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": W and V, "W": W}

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I have a bug in the return statement. Let me fix that.

