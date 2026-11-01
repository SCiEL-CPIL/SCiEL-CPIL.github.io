"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
import itertools
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k] for output (i,k)."""
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_true = _build_tensor(n, m, p)
    
    diff = np.abs(T_recon - T_true)
    return np.max(diff) < tol, np.max(diff)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W of shape (4, 7) each with integer entries.
    
    The 7 products are:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    Output:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U vectors for A = [[a00, a01], [a10, a11]]
    # Index: 0=a00, 1=a01, 2=a10, 3=a11
    U = np.zeros((4, 7), dtype=float)
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    # M3 = a00(b10-b11)
    U[0, 2] = 1
    # M4 = a11(b01-b00)
    U[3, 3] = 1
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    
    # V vectors for B = [[b00, b01], [b10, b11]]
    # Index: 0=b00, 1=b01, 2=b10, 3=b11
    V = np.zeros((4, 7), dtype=float)
    # M1 = (a00+a11)(b00+b11)
    V[0, 0] = 1; V[3, 0] = 1
    # M2 = (a10+a11)b00
    V[0, 1] = 1
    # M3 = a00(b10-b11)
    V[2, 2] = 1; V[3, 2] = -1
    # M4 = a11(b01-b00)
    V[1, 3] = 1; V[0, 3] = -1
    # M5 = (a00+a01)b11
    V[3, 4] = 1
    # M6 = (a10-a00)(b00+b01)
    V[0, 5] = 1; V[1, 5] = 1
    # M7 = (a01-a11)(b10+b11)
    V[2, 6] = 1; V[3, 6] = 1
    
    # W vectors for C = [[c00, c01], [c10, c11]]
    # Index: 0=c00, 1=c01, 2=c10, 3=c11
    W = np.zeros((4, 7), dtype=float)
    # C00 = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    # C01 = M3 + M5
    W[1, 2] = 1; W[1, 4] = 1
    # C10 = M2 + M4
    W[2, 1] = 1; W[2, 3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def _hybrid_decomposition(n, m, p):
    """
    Construct a hybrid Strassen/Schoolbook decomposition for n=2, m=3, p=7.
    
    Strategy:
    - Split the inner dimension j=0,1,2
    - For j=0,1: use Strassen-like decomposition on the 2x2x7 subproblem
      Actually, let's think differently.
    
    Better strategy:
    - The output is 2x7, input A is 2x3, input B is 3x7
    - C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    We can group the 7 columns of B into groups and use Strassen where possible.
    
    For the first 6 columns (k=0..5), we can pair them: (0,1), (2,3), (4,5)
    For each pair, we have a 2x2x2 subproblem in the output dimension.
    But the inner dimension is still 3, not 2.
    
    Alternative: Use Strassen on the 2x2 part (rows of A, columns of C) for each column of B,
    but that doesn't help since we already have only 2 rows.
    
    Actually, the known improvement for 2x3x7 is to use a specific decomposition.
    Let me try a different approach:
    
    For k=0,1,2,3,4,5,6:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    We can try to combine terms across k values.
    
    Let's use a known rank-39 construction for 2x3x7.
    One approach: use 7 rank-1 terms for the first column pair (Strassen), 
    and schoolbook for the rest.
    
    Actually, let's try:
    - For columns k=0,1: use a 2x2x2 Strassen block (7 terms)
    - For columns k=2,3: use another 2x2x2 Strassen block (7 terms)  
    - For columns k=4,5: use another 2x2x2 Strassen block (7 terms)
    - For column k=6: use 3 schoolbook terms
    
    But wait, Strassen works for 2x2x2, meaning 2 rows of A, 2 columns of B, 2x2 output.
    Here we have 2 rows of A, 3 columns of B (inner dim), and we want to produce 2 columns of output.
    This is a 2x3x2 subproblem, not 2x2x2.
    
    Let me reconsider. The hybrid approach that achieved rank 39 likely uses:
    - Some combination of Strassen and schoolbook that exploits the specific dimensions.
    
    For now, let's construct a valid decomposition and see if we can reduce the rank.
    
    Actually, let me just implement a simple approach:
    - Use schoolbook for all 42 terms as a baseline
    - Then try to find a lower rank decomposition using the structure
    """
    # For now, return the schoolbook decomposition
    return _schoolbook(n, m, p)

def _try_rank(n, m, p, rank, time_budget=10.0):
    """Try to find a decomposition of the given rank."""
    start_time = time.time()
    T = _build_tensor(n, m, p)
    
    n_schoolbook = n * m * p
    if rank > n_schoolbook:
        return None
    
    # Initialize with random vectors
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    if rank >= n_schoolbook:
        for col in range(rank):
            i = col // (m * p)
            j = (col // p) % m
            k = col % p
            U[i * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + i, col] = 1
    else:
        for col in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.5:
                            U[i * m + j, col] += random.gauss(0, 1)
                            V[j * p + k, col] += random.gauss(0, 1)
                            W[k * n + i, col] += random.gauss(0, 1)
    
    best_diff = np.inf
    for iteration in range(200):
        if time.time() - start_time > time_budget:
            break
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        diff = np.max(np.abs(T_recon - T))
        
        if diff < best_diff:
            best_diff = diff
            if best_diff < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
        
        col = random.randint(0, rank - 1)
        if random.random() < 0.33:
            idx = random.randint(0, n * m - 1)
            U[idx, col] += random.gauss(0, 0.3)
        elif random.random() < 0.5:
            idx = random.randint(0, m * p - 1)
            V[idx, col] += random.gauss(0, 0.3)
        else:
            idx = random.randint(0, p * n - 1)
            W[idx, col] += random.gauss(0, 0.3)
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    diff = np.max(np.abs(T_recon - T))
    if diff < 1e-10:
        return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for matrix multiplication of sizes 2x3 and 3x7.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Get the schoolbook solution as a fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_school, V_school, W_school)
    
    # Try the hybrid decomposition
    U_hybrid, V_hybrid, W_hybrid = _hybrid_decomposition(n, m, p)
    if U_hybrid:
        valid, diff = _verify(U_hybrid, V_hybrid, W_hybrid, n, m, p)
        if valid:
            hybrid_rank = len(U_hybrid[0])
            if hybrid_rank < best_rank:
                best_rank = hybrid_rank
                best_result = (U_hybrid, V_hybrid, W_hybrid)
    
    # Try to find even lower rank decompositions
    lower_bound = max(n * m, m * p, p * n)
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        for restart in range(3):
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, rank, time_budget=15.0)
            if result is not None:
                U, V, W = result
                valid, diff = _verify(U, V, W, n, m, p)
                if valid:
                    best_rank = rank
                    best_result = (U, V, W)
                    break
    
    U, V, W = best_result
    return {
        "U": U,
        "V": V,
        "W": W
    }
# EVOLVE-BLOCK-END
