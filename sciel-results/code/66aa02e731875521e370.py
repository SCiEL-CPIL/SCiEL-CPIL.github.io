import time
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [2, 3, 3]

def _verify_exact(U, V, W):
    """
    Verify that the decomposition exactly matches the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r] should be 1 for all i,j,k.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    # Convert to float for checking, assuming input is already clean (int or half-int)
    U_f = np.array(U, dtype=float)
    V_f = np.array(V, dtype=float)
    W_f = np.array(W, dtype=float)
    
    for r in range(rank):
        T += U_f[:, r][:, np.newaxis, np.newaxis] * V_f[:, r][np.newaxis, :, np.newaxis] * W_f[:, r][np.newaxis, np.newaxis, :]
    
    # Target is all ones
    target = np.ones((n, m, p), dtype=float)
    return np.allclose(T, target, atol=1e-9)

def _schoolbook_solution():
    """
    Generate the standard schoolbook decomposition of rank 18.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    return U, V, W

def _strassen_schoolbook_solution():
    """
    Construct a rank-17 solution using Strassen for the top-left 2x2x2 block
    and schoolbook for the remaining terms.
    
    Dimensions: 2x3 * 3x3 -> 2x3
    Block structure:
    A = [[A00, A01, A02],
         [A10, A11, A12]]
    B = [[B00, B01, B02],
         [B10, B11, B12],
         [B20, B21, B22]]
    
    C = A @ B
    
    We decompose into:
    1. Terms involving column 2 of A (A02, A12) and all B: Schoolbook (6 terms)
       A02 * B[2,:] -> C[:, 2] ... wait, A02 is scalar? No, A is 2x3.
       A[i,2] * B[2,k].
       Terms:
       r0: A02 B20 -> C00
       r1: A02 B21 -> C01
       r2: A02 B22 -> C02
       r3: A12 B20 -> C10
       r4: A12 B21 -> C11
       r5: A12 B22 -> C12
       
    2. Terms involving row 2 of B (B20, B21, B22) and columns 0,1 of A: Schoolbook (4 terms)
       A[i,j] * B[j,2] for j=0,1.
       r6: A00 B02 -> C02
       r7: A01 B12 -> C02
       r8: A10 B02 -> C12
       r9: A11 B12 -> C12
       
    3. Remaining terms involve A[:, 0:2] and B[0:2, 0:2]. This is a 2x2 * 2x2 multiplication.
       We use Strassen's algorithm for this block (7 terms).
       
    Total rank = 6 + 4 + 7 = 17.
    
    Mapping to global indices:
    U index: i*m + j = i*3 + j. (i in 0..1, j in 0..2)
    V index: j*p + k = j*3 + k. (j in 0..2, k in 0..2)
    W index: k*n + i = k*2 + i. (k in 0..2, i in 0..1)
    
    Schoolbook Part 1 (A col 2):
    r=0: A02 B20 -> C00
       U[0*3+2, 0]=1, V[2*3+0, 0]=1, W[0*2+0, 0]=1
    r=1: A02 B21 -> C01
       U[2, 1]=1, V[7, 1]=1, W[1*2+0, 1]=1 -> W[2, 1]=1
    r=2: A02 B22 -> C02
       U[2, 2]=1, V[8, 2]=1, W[2*2+0, 2]=1 -> W[4, 2]=1
    r=3: A12 B20 -> C10
       U[1*3+2, 3]=1 -> U[5, 3]=1, V[6, 3]=1, W[0, 3]=1
    r=4: A12 B21 -> C11
       U[5, 4]=1, V[7, 4]=1, W[1*2+1, 4]=1 -> W[3, 4]=1
    r=5: A12 B22 -> C12
       U[5, 5]=1, V[8, 5]=1, W[2*2+1, 5]=1 -> W[5, 5]=1
       
    Schoolbook Part 2 (B col 2, A cols 0,1):
    r=6: A00 B02 -> C02
       U[0*3+0, 6]=1 -> U[0, 6]=1, V[0*3+2, 6]=1 -> V[2, 6]=1, W[2*2+0, 6]=1 -> W[4, 6]=1
    r=7: A01 B12 -> C02
       U[0*3+1, 7]=1 -> U[1, 7]=1, V[1*3+2, 7]=1 -> V[5, 7]=1, W[4, 7]=1
    r=8: A10 B02 -> C12
       U[1*3+0, 8]=1 -> U[3, 8]=1, V[2, 8]=1, W[5, 8]=1
    r=9: A11 B12 -> C12
       U[1*3+1, 9]=1 -> U[4, 9]=1, V[5, 9]=1, W[5, 9]=1
       
    Strassen Part (A[:, 0:2] * B[0:2, 0:2]):
    Local indices: i_l in 0,1; j_l in 0,1; k_l in 0,1.
    Global U index: i_l*3 + j_l.
    Global V index: j_l*3 + k_l.
    Global W index: k_l*2 + i_l.
    
    Strassen intermediates (M1..M7) correspond to rank terms r=10..16.
    
    M1 = (A00 + A11)(B00 + B11)
       U: A00 (U[0, r]), A11 (U[4, r])
       V: B00 (V[0, r]), B11 (V[4, r])
       W: C00 (W[0, r]), C11 (W[3, r]) ... wait, W accumulates contributions to C.
       
    Let's list the Strassen terms explicitly with global indices.
    
    M1: (A00 + A11) * (B00 + B11)
       U[0, r] = 1, U[4, r] = 1
       V[0, r] = 1, V[4, r] = 1
       Contributes to C00 and C11?
       C00 += M1
       C11 += M1
       So W[0, r] = 1 (for C00, k=0, i=0 -> idx 0*2+0=0)
       W[3, r] = 1 (for C11, k=1, i=1 -> idx 1*2+1=3)
       
    M2: (A10 + A11) * B00
       U[3, r] = 1 (A10), U[4, r] = 1 (A11)
       V[0, r] = 1 (B00)
       Contributes to C10.
       C10 += M2
       W[1, r] = 1 (C10, k=0, i=1 -> idx 0*2+1=1)
       
    M3: A00 * B11
       U[0, r] = 1
       V[4, r] = 1
       Contributes to C01 and C11?
       C01 += M3
       C11 -= M3  (Wait, C11 = M1 - M3 + M2 + M5)
       So W[2, r] = 1 (C01, k=1, i=0 -> idx 1*2+0=2)
       W[3, r] = -1 (C11, k=1, i=1 -> idx 3)
       
    M4: A11 * B00
       U[4, r] = 1
       V[0, r] = 1
       Contributes to C00 and C10?
       C00 -= M4
       C10 += M4
       So W[0, r] = -1
       W[1, r] = 1
       
    M5: (A00 + A01) * B11
       U[0, r] = 1, U[1, r] = 1
       V[4, r] = 1
       Contributes to C01 and C11?
       C01 += M5
       C11 += M5
       So W[2, r] = 1
       W[3, r] = 1
       
    M6: (A10 - A00) * B01
       U[3, r] = 1, U[0, r] = -1
       V[1, r] = 1
       Contributes to C00 and C10?
       C00 += M6
       C10 += M6
       So W[0, r] = 1
       W[1, r] = 1
       
    M7: (A01 - A11) * B10
       U[1, r] = 1, U[4, r] = -1
       V[3, r] = 1
       Contributes to C00 and C01?
       C00 += M7
       C01 += M7
       So W[0, r] = 1
       W[2, r] = 1
       
    Let's assemble the matrices.
    Rank 17.
    r=0..5: Schoolbook 1
    r=6..9: Schoolbook 2
    r=10..16: Strassen
    """
    n, m, p = DIMENSIONS
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: A col 2
    # r=0: A02 B20 -> C00
    U[2, 0] = 1.0; V[6, 0] = 1.0; W[0, 0] = 1.0
    # r=1: A02 B21 -> C01
    U[2, 1] = 1.0; V[7, 1] = 1.0; W[2, 1] = 1.0
    # r=2: A02 B22 -> C02
    U[2, 2] = 1.0; V[8, 2] = 1.0; W[4, 2] = 1.0
    # r=3: A12 B20 -> C10
    U[5, 3] = 1.0; V[6, 3] = 1.0; W[1, 3] = 1.0
    # r=4: A12 B21 -> C11
    U[5, 4] = 1.0; V[7, 4] = 1.0; W[3, 4] = 1.0
    # r=5: A12 B22 -> C12
    U[5, 5] = 1.0; V[8, 5] = 1.0; W[5, 5] = 1.0
    
    # Part 2: B col 2 (from A cols 0,1)
    # r=6: A00 B02 -> C02
    U[0, 6] = 1.0; V[2, 6] = 1.0; W[4, 6] = 1.0
    # r=7: A01 B12 -> C02
    U[1, 7] = 1.0; V[5, 7] = 1.0; W[4, 7] = 1.0
    # r=8: A10 B02 -> C12
    U[3, 8] = 1.0; V[2, 8] = 1.0; W[5, 8] = 1.0
    # r=9: A11 B12 -> C12
    U[4, 9] = 1.0; V[5, 9] = 1.0; W[5, 9] = 1.0
    
    # Part 3: Strassen for A[:, 0:2] * B[0:2, 0:2]
    # r=10: M1 = (A00 + A11)(B00 + B11)
    U[0, 10] = 1.0; U[4, 10] = 1.0
    V[0, 10] = 1.0; V[4, 10] = 1.0
    W[0, 10] = 1.0; W[3, 10] = 1.0
    
    # r=11: M2 = (A10 + A11) B00
    U[3, 11] = 1.0; U[4, 11] = 1.0
    V[0, 11] = 1.0
    W[1, 11] = 1.0
    
    # r=12: M3 = A00 B11
    U[0, 12] = 1.0
    V[4, 12] = 1.0
    W[2, 12] = 1.0; W[3, 12] = -1.0
    
    # r=13: M4 = A11 B00
    U[4, 13] = 1.0
    V[0, 13] = 1.0
    W[0, 13] = -1.0; W[1, 13] = 1.0
    
    # r=14: M5 = (A00 + A01) B11
    U[0, 14] = 1.0; U[1, 14] = 1.0
    V[4, 14] = 1.0
    W[2, 14] = 1.0; W[3, 14] = 1.0
    
    # r=15: M6 = (A10 - A00) B01
    U[3, 15] = 1.0; U[0, 15] = -1.0
    V[1, 15] = 1.0
    W[0, 15] = 1.0; W[1, 15] = 1.0
    
    # r=16: M7 = (A01 - A11) B10
    U[1, 16] = 1.0; U[4, 16] = -1.0
    V[3, 16] = 1.0
    W[0, 16] = 1.0; W[2, 16] = 1.0
    
    return U, V, W

def _attempt_rank_16_optimization():
    """
    Attempt to find a rank-16 solution using local optimization.
    Returns None if not found or if time limit is approaching.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Very tight budget to ensure safety
    
    n, m, p = DIMENSIONS
    rank = 16
    
    target = np.ones((n, m, p))
    
    def objective(params):
        u_flat = params[:n*m*rank]
        v_flat = params[n*m*rank : n*m*rank + m*p*rank]
        w_flat = params[n*m*rank + m*p*rank : ]
        
        U = u_flat.reshape(n*m, rank)
        V = v_flat.reshape(m*p, rank)
        W = w_flat.reshape(p*n, rank)
        
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = T - target
        return np.sum(residual**2)
    
    # Initialize from rank-17 solution? Hard to do directly with different rank.
    # Just use random restarts.
    num_params = (n*m + m*p + p*n) * rank
    
    for _ in range(5): # 5 restarts
        if time.time() > deadline - 2:
            break
        
        x0 = np.random.uniform(-2, 2, size=num_params)
        # Use a simple gradient descent or L-BFGS-B
        from scipy.optimize import minimize
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8})
        
        if res.fun < 1e-6:
            # Verify if it can be rounded to half-integers
            u_flat = res.x[:n*m*rank]
            v_flat = res.x[n*m*rank : n*m*rank + m*p*rank]
            w_flat = res.x[n*m*rank + m*p*rank : ]
            
            U = u_flat.reshape(n*m, rank)
            V = v_flat.reshape(m*p, rank)
            W = w_flat.reshape(p*n, rank)
            
            U_r = np.round(U * 2) / 2.0
            V_r = np.round(V * 2) / 2.0
            W_r = np.round(W * 2) / 2.0
            
            if _verify_exact(U_r, V_r, W_r):
                return U_r, V_r, W_r
                
    return None

def construct():
    """
    Returns the best known decomposition.
    Currently tries rank 16 with very limited time, then falls back to rank 17.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Attempt rank 16
    res_16 = _attempt_rank_16_optimization()
    if res_16 is not None:
        U, V, W = res_16
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to rank 17
    U, V, W = _strassen_schoolbook_solution()
    
    # Verify just in case
    if not _verify_exact(U, V, W):
        # This should not happen, but if it does, try schoolbook rank 18
        U, V, W = _schoolbook_solution()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# Keep a dummy main if needed, but task says keep entry point names.
# The entry point is construct.

if __name__ == "__main__":
    res = construct()
    print("Constructed decomposition.")
