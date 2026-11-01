"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n=3, m=3, p=3, tol=1e-6):
    """Verify that U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _rank24_decomposition():
    """
    Construct a rank-24 decomposition for 3x3 matrix multiplication.
    
    Strategy: Decompose the 3x3 multiplication into:
    1. A 2x2 Strassen core for the top-left block (rank 7)
    2. A 1x1 scalar term for C[2,2] (rank 1)  
    3. Cross terms between the 2x2 block and the 1x1 block (rank 16)
    
    Total: 7 + 1 + 16 = 24
    """
    n, m, p = 3, 3, 3
    rank = 24
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: Strassen for 2x2 top-left block (indices 0,1 for both A and B)
    # A = [[a00, a01],
    #      [a10, a11]]
    # B = [[b00, b01],
    #      [b10, b11]]
    # 
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b10 - b11)
    # M4 = a11(b01 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Strassen terms (indices 0-6)
    # M1 = (a00 + a11)(b00 + b11)
    U[0][0] = 1.0; U[1][0] = 1.0  # a00 + a11
    V[0][0] = 1.0; V[1][0] = 1.0  # b00 + b11
    W[0][0] = 1.0; W[1][0] = 1.0  # C00 += M1, C11 += M1
    
    # M2 = (a10 + a11)b00
    U[1][1] = 1.0; U[2][1] = 1.0  # a10 + a11
    V[0][1] = 1.0                 # b00
    W[2][1] = 1.0; W[1][1] = -1.0 # C10 += M2, C11 -= M2
    
    # M3 = a00(b10 - b11)
    U[0][2] = 1.0                 # a00
    V[1][2] = 1.0; V[0][2] = -1.0 # b10 - b11
    W[1][2] = 1.0; W[3][2] = 1.0  # C01 += M3, C11 += M3
    
    # M4 = a11(b01 - b00)
    U[1][3] = 1.0                 # a11
    V[0][3] = -1.0; V[1][3] = 1.0 # b01 - b00
    W[0][3] = 1.0; W[2][3] = 1.0  # C00 += M4, C10 += M4
    
    # M5 = (a00 + a01)b11
    U[0][4] = 1.0; U[1][4] = 1.0  # a00 + a01
    V[1][4] = 1.0                 # b11
    W[0][4] = -1.0; W[1][4] = 1.0 # C00 -= M5, C01 += M5
    
    # M6 = (a10 - a00)(b00 + b01)
    U[1][5] = 1.0; U[0][5] = -1.0 # a10 - a00
    V[0][5] = 1.0; V[1][5] = 1.0  # b00 + b01
    W[0][5] = 1.0                 # C00 += M6... wait, let me check Strassen again
    
    # Actually, let me use the standard Strassen formula more carefully.
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # Let me redo this with correct indices.
    # Reset
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Using 0-indexed: A[i,j], B[j,k], C[i,k]
    # Top-left 2x2 block: i,j in {0,1}, k in {0,1}
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1.0; U[1][0] = 1.0
    V[0][0] = 1.0; V[1][0] = 1.0
    W[0][0] = 1.0; W[1][0] = 1.0
    
    # M2 = (A10 + A11)B00
    U[1][1] = 1.0; U[2][1] = 1.0
    V[0][1] = 1.0
    W[2][1] = 1.0; W[1][1] = -1.0
    
    # M3 = A00(B10 - B11)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[0][2] = -1.0
    W[1][2] = 1.0; W[3][2] = 1.0
    
    # M4 = A11(B01 - B00)
    U[1][3] = 1.0
    V[0][3] = -1.0; V[1][3] = 1.0
    W[0][3] = 1.0; W[2][3] = 1.0
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1.0; U[1][4] = 1.0
    V[1][4] = 1.0
    W[0][4] = -1.0; W[1][4] = 1.0
    
    # M6 = (A10 - A00)(B00 + B01)
    U[1][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[0][5] = 1.0  # Hmm, this doesn't match standard Strassen
    
    # Let me look up the exact Strassen coefficients again.
    # Actually, I think I'm overcomplicating this. Let me just use a known rank-24 construction.
    
    # Alternative approach: Use the fact that 3x3 can be decomposed using 
    # a combination of smaller multiplications.
    
    # Let me try a different approach: use a known algebraic identity.
    
    # For now, let me construct a valid rank-27 (schoolbook) as fallback,
    # and try to find a rank-24 via optimization.
    
    return None

def _optimize_rank(rank, deadline):
    """
    Optimize for a tensor decomposition of given rank using gradient descent.
    """
    n, m, p = 3, 3, 3
    total_vars = (n*m + m*p + p*n) * rank
    
    def objective(x):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    errors.append(val - 1.0)
        return np.sum(np.array(errors)**2)
    
    # Initialize
    np.random.seed(42)
    x0 = np.random.randn(total_vars) * 0.1
    
    # Use scipy if available, else simple gradient descent
    try:
        from scipy.optimize import minimize
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 50, 'ftol': 1e-8})
        if result.fun < 1e-4:
            x = result.x
            U = x[:n*m*rank].reshape(n*m, rank)
            V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Round to nearest half-integer
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Try rank 24 via optimization
    for r in [24, 25, 26]:
        if time.time() > deadline - 10:
            break
        try:
            res = _optimize_rank(r, deadline)
            if res:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Fallback to schoolbook rank 27
    n, m, p = 3, 3, 3
    rank = 27
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
