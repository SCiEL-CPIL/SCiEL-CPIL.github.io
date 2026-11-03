"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def _build_strassen_2x2x2():
    """
    Build Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W of shape (4,7), (4,7), (4,7) for 2x2x2.
    
    Strassen formulas:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    """
    # Indexing: A is 2x2, B is 2x2, C is 2x2
    # U[i*m+j, r] where m=2, so U[i*2+j, r]
    # V[j*p+k, r] where p=2, so V[j*2+k, r]
    # W[k*n+i, r] where n=2, so W[k*2+i, r]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[1, 0] = 1  # B12... wait, need to be careful
    
    # Let me use explicit indexing
    # A indices: A[0,0], A[0,1], A[1,0], A[1,1] -> flat: 0,1,2,3
    # B indices: B[0,0], B[0,1], B[1,0], B[1,1] -> flat: 0,1,2,3
    # C indices: C[0,0], C[0,1], C[1,0], C[1,1] -> flat: 0,1,2,3
    
    # M1 = (A11+A22)(B11+B22)
    # A11 = A[0,0] = flat 0, A22 = A[1,1] = flat 3
    # B11 = B[0,0] = flat 0, B22 = B[1,1] = flat 3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 += M1, C22 += M1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0] and C[1,1]
    
    # M2 = (A21+A22)B11
    # A21 = A[1,0] = flat 2, A22 = A[1,1] = flat 3
    # B11 = B[0,0] = flat 0
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 += M2, C22 -= M2
    W[2, 1] = 1  # C[1,0]
    W[3, 1] = -1  # C[1,1]
    
    # M3 = A11(B21-B22)
    # A11 = A[0,0] = flat 0
    # B21 = B[1,0] = flat 2, B22 = B[1,1] = flat 3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    # C12 += M3, C22 += M3
    W[1, 2] = 1  # C[0,1]
    W[3, 2] = 1  # C[1,1]
    
    # M4 = A22(B12-B11)
    # A22 = A[1,1] = flat 3
    # B12 = B[0,1] = flat 1, B11 = B[0,0] = flat 0
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    # C11 += M4, C21 += M4
    W[0, 3] = 1  # C[0,0]
    W[2, 3] = 1  # C[1,0]
    
    # M5 = (A11+A12)B22
    # A11 = A[0,0] = flat 0, A12 = A[0,1] = flat 1
    # B22 = B[1,1] = flat 3
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 -= M5, C12 += M5
    W[0, 4] = -1  # C[0,0]
    W[1, 4] = 1  # C[0,1]
    
    # M6 = (A21-A11)(B11+B12)
    # A21 = A[1,0] = flat 2, A11 = A[0,0] = flat 0
    # B11 = B[0,0] = flat 0, B12 = B[0,1] = flat 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 += M6
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (A12-A22)(B21+B22)
    # A12 = A[0,1] = flat 1, A22 = A[1,1] = flat 3
    # B21 = B[1,0] = flat 2, B22 = B[1,1] = flat 3
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += M7
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W

def _verify(U, V, W, tol=1e-8):
    """Verify tensor decomposition against multiple random test cases."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    
    # Test with multiple random matrices for robustness
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        U_reshaped = np.asarray(U).reshape(n, m, R)
        V_reshaped = np.asarray(V).reshape(m, p, R)
        W_reshaped = np.asarray(W).reshape(p, n, R)
        
        S_A = np.einsum('ijr,ij->ir', U_reshaped, A)
        S_B = np.einsum('jkr,jk->kr', V_reshaped, B)
        C_comp = np.einsum('kir,ir,kr->ik', W_reshaped, S_A, S_B)
        
        if not np.allclose(C_true, C_comp, atol=tol):
            return False
    
    return True

# EVOLVE-BLOCK-START
def construct():
    """
    Subproduct method: split 3x4x7 into 2x4x7 + 1x4x7.
    - 2x4x7: Use block Strassen-like decomposition
    - 1x4x7: Schoolbook (rank 7)
    Total target rank: 21 + 7 = 28 (if 2x4x7 has rank 21)
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Strategy: Build explicit decomposition
    # Part 1: 2x4x7 multiplication (rows 0,1 of A)
    # Part 2: 1x4x7 multiplication (row 2 of A)
    
    # For 2x4x7: split into 2x2x2 blocks? No, 4 and 7 aren't powers of 2.
    # Alternative: Use known Strassen for 2x2x2 and extend.
    # Actually, let's try a different approach:
    # Split 4 into 2+2, and 7 into 3+2+2
    # This gets complicated. Let me try direct optimization with better init.
    
    # Better approach: Use the structure to initialize
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Let's try rank 28 first (subproduct bound)
    target_ranks = [28, 29, 30, 31, 32, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 83, 84]
    
    for R in target_ranks:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + attempt * 1000 + R)
            
            # Initialize with structured values based on subproduct idea
            # For the 2x4x7 part: use Strassen-like patterns
            # For the 1x4x7 part: schoolbook
            
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Part 1: 2x4x7 using 7 Strassen-like terms per 2x2x2 block
            # Split columns 4 into 2+2, rows 7 into 3+2+2? 
            # Simpler: just use random init for now and optimize
            
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            def residuals(x):
                U_flat = x[:n*m*R].reshape(n*m, R)
                V_flat = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_flat = x[n*m*R + m*p*R :].reshape(p*n, R)
                
                U_r = U_flat.reshape(n, m, R)
                V_r = V_flat.reshape(m, p, R)
                W_r = W_flat.reshape(p, n, R)
                
                T_comp = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                return (T_comp - 1.0).flatten()
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = least_squares(residuals, x0, method='lm', max_nfev=500)
                if res.cost < 1e-12:
                    U_sol = res.x[:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                    
                    if _verify(U_sol, V_sol, W_sol):
                        return {
                            "U": U_sol.tolist(),
                            "V": V_sol.tolist(),
                            "W": W_sol.tolist()
                        }
            except Exception:
                pass
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
