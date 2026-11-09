"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook_decomp(n, m, p):
    """Return schoolbook decomposition for n x m * m x p -> n x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomp(U, V, W, n, m, p):
    """Verify that U, V, W form a valid exact decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    
    # For each (i, j, k), check sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen products:
    # M1 = (A00+A01)(B10+B11)
    # M2 = (A10+A11)(B00+B11)
    # M3 = A00(B00-B10)
    # M4 = A11(B11-B01)
    # M5 = (A00+A10)(B00+B01)
    # M6 = (A11+A01)(B10+B00)
    # M7 = (A01-A11)(B11+B10)
    
    # U entries for A00, A01, A10, A11
    # V entries for B00, B01, B10, B11
    # W entries for C00, C01, C10, C11
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: A00, A01; B10, B11; C00, C10
    U[0][0] = 1; U[1][0] = 1  # A00, A01
    V[2][0] = 1; V[3][0] = 1  # B10, B11
    W[0][0] = 1; W[2][0] = 1  # C00, C10
    
    # M2: A10, A11; B00, B11; C00, C10
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1; V[3][1] = 1  # B00, B11
    W[0][1] = -1; W[2][1] = 1 # C00, C10
    
    # M3: A00; B00, B10; C01, C11
    U[0][2] = 1  # A00
    V[0][2] = 1; V[2][2] = -1 # B00, B10
    W[1][2] = 1; W[3][2] = 1  # C01, C11
    
    # M4: A11; B11, B01; C00, C11
    U[3][3] = 1  # A11
    V[3][3] = 1; V[1][3] = -1 # B11, B01
    W[0][3] = 1; W[3][3] = -1 # C00, C11
    
    # M5: A00, A10; B00, B01; C00, C11
    U[0][4] = 1; U[2][4] = 1  # A00, A10
    V[0][4] = 1; V[1][4] = 1  # B00, B01
    W[0][4] = 1; W[3][4] = 1  # C00, C11
    
    # M6: A11, A01; B10, B00; C11
    U[3][5] = 1; U[1][5] = 1  # A11, A01
    V[2][5] = 1; V[0][5] = 1  # B10, B00
    W[3][5] = -1  # C11
    
    # M7: A01, A11; B11, B10; C01
    U[1][6] = 1; U[3][6] = -1 # A01, A11
    V[3][6] = 1; V[2][6] = 1  # B11, B10
    W[1][6] = 1  # C01
    
    return U, V, W

def _construct_3x2x2_strassen():
    """
    Construct a decomposition for 3x2 * 2x2 using Strassen-like ideas.
    A is 3x2, B is 2x2, C is 3x2.
    We can view this as computing 3 rows of A times B.
    However, the B matrix is shared.
    
    Let's try to construct a rank 12 decomposition first (schoolbook) and see if we can reduce.
    Actually, let's just use the schoolbook for 3x2x2 which is rank 12.
    Then 3x4x8 = 2 * (3x2x4) + ... no.
    
    Let's try a different approach.
    3x4x8 can be decomposed by splitting the middle dimension 4 into 2+2.
    C = A1 B1 + A2 B2.
    A1, A2 are 3x2. B1, B2 are 2x8.
    This is sum of two 3x2x8 products.
    Rank(3x2x8) = ?
    We can further split 8 into 4+4.
    B1 = [B1a, B1b], B2 = [B2a, B2b] where each is 2x4.
    This gets complicated.
    
    Let's try to use Strassen on the 2x2 blocks of the 4x4 middle part?
    No, the middle is 4.
    
    Let's try to find a decomposition for 3x2x2 with rank < 12.
    If we can get rank 10, then 3x4x8 might be 40?
    
    Actually, let's just implement a random search for 3x2x2 first to see if we can beat 12.
    Then use that to build 3x4x8.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # First, try to find a good decomposition for 3x2x2.
    # If we find one with rank R < 12, we can potentially use it.
    # But how to extend 3x2x2 to 3x4x8?
    # 3x4x8 is not a simple tensor product of 3x2x2.
    
    # Let's just do a direct random search for 3x4x8.
    # Schoolbook rank is 96.
    # Let's try to find a decomposition with rank < 96.
    
    # We will use an iterative optimization approach.
    # Initialize U, V, W randomly.
    # Update them to minimize the error ||T - sum_r u_r v_r w_r||.
    
    # T is the matrix multiplication tensor.
    # T[i, j, k] = 1.
    
    # We will try a few different ranks.
    # Start with rank 90, then 85, etc.
    
    # To make it faster, we can use a greedy approach or random restarts.
    
    # Let's try rank 80.
    target_rank = 80
    num_trials = 5
    max_iters = 50
    
    best_result = None
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        np.random.seed(trial)
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update W
            W_new = W.copy()
            for i in range(n):
                for k in range(p):
                    # We want W[k*n+i, :] such that sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                    # This is not a simple linear system for W because U and V are also variables.
                    # But if we fix U and V, we can solve for W?
                    # No, it's a tensor decomposition.
                    # Let's use a simple gradient step.
                    # Or use the ALS approach from inspiration 2.
                    
                    # ALS:
                    # Fix U, V. Solve for W.
                    # For each (i, k), we have a vector W_i_k of length rank.
                    # The equation is sum_r U_i_j(r) V_j_k(r) W_i_k(r) = 1.
                    # This is sum_r (U_i_j * V_j_k)_r W_i_k(r) = 1.
                    # Let A_ik(r) = sum_j U[i*m+j, r] V[j*p+k, r].
                    # Then sum_r A_ik(r) W_i_k(r) = 1.
                    # This is a single linear equation for the vector W_i_k.
                    # It has many solutions. We want the one that minimizes ||W_i_k||^2?
                    # Or just pick one.
                    # If A_ik is not zero, we can set W_i_k = A_ik / ||A_ik||^2.
                    
                    A_ik = np.zeros(target_rank)
                    for j in range(m):
                        A_ik += U[i*m+j, :] * V[j*p+k, :]
                    
                    if np.linalg.norm(A_ik) > 1e-12:
                        W_new[k*n+i, :] = A_ik / np.dot(A_ik, A_ik)
                    else:
                        W_new[k*n+i, :] = 0
                    
            W = W_new
            
            # Similarly update U and V.
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    A_ij = np.zeros(target_rank)
                    for k in range(p):
                        A_ij += V[j*p+k, :] * W[k*n+i, :]
                    if np.linalg.norm(A_ij) > 1e-12:
                        U_new[i*m+j, :] = A_ij / np.dot(A_ij, A_ij)
                    else:
                        U_new[i*m+j, :] = 0
            U = U_new
            
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    A_jk = np.zeros(target_rank)
                    for i in range(n):
                        A_jk += U[i*m+j, :] * W[k*n+i, :]
                    if np.linalg.norm(A_jk) > 1e-12:
                        V_new[j*p+k, :] = A_jk / np.dot(A_jk, A_jk)
                    else:
                        V_new[j*p+k, :] = 0
            V = V_new
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        error += (val - 1.0)**2
                        
            if error < 1e-8:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2.0
                V_r = np.round(V * 2) / 2.0
                W_r = np.round(W * 2) / 2.0
                
                err2 = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                            err2 += (val - 1.0)**2
                            
                if err2 < 1e-8:
                    best_result = {
                        'U': U_r.tolist(),
                        'V': V_r.tolist(),
                        'W': W_r.tolist()
                    }
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return _schoolbook_decomp(n, m, p)

# EVOLVE-BLOCK-END
