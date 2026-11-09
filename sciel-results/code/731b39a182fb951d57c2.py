import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin within 120s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # Using the official trace convention W[k*n+i, r]
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # a-index: i*m+j, b-index: j*p+k, c-index: k*n+i
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify_decomposition(U, V, W, T_target, tol=1e-9):
        """Verify that sum_r U[:,r] V[:,r] W[:,r] == T_target"""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_recon, T_target, atol=tol, rtol=0):
            return True
        return False
    
    # Schoolbook decomposition (rank 50)
    U_school = np.zeros((n*m, schoolbook_rank), dtype=float)
    V_school = np.zeros((m*p, schoolbook_rank), dtype=float)
    W_school = np.zeros((p*n, schoolbook_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = schoolbook_rank
    
    # Verify schoolbook
    if not verify_decomposition(best_U, best_V, best_W, T):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Strategy: Use Strassen's algorithm for M_{3,3,3} (rank 7) to embed
    # M_{2,5,5} into a larger problem and project back.
    
    # Key insight: We can use the Kronecker product structure.
    # M_{2,5,5} can be seen as a sub-problem of M_{3,5,5} or similar.
    # Alternatively, we can use the fact that M_{2,5,5} has a known lower bound of 25.
    # Let's try to find a rank 25 decomposition using structured patterns.
    
    # Approach: Use the identity that M_{n,m,p} can be decomposed using
    # the Kronecker product of smaller matrix multiplications.
    
    # For M_{2,5,5}, we can try to use the rank-7 Strassen decomposition for M_{3,3,3}
    # and embed it appropriately.
    
    # Strassen's 7-term decomposition for M_{3,3,3}:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # For M_{2,5,5}, let's try a different approach.
    # We know that M_{2,5,5} >= 25 (lower bound).
    # Let's try to construct a rank 25 decomposition.
    
    # Idea: Use the fact that M_{2,5,5} can be decomposed as:
    # For each (i,j,k) with i in {0,1}, j in {0,...,4}, k in {0,...,4}
    # We need U[i*5+j, r] * V[j*5+k, r] * W[k*2+i, r] to sum to 1
    
    # Try a structured rank 25 decomposition:
    # Let r = j*2 + i for U, and r = j*5 + k for V, and r = k*2 + i for W
    # This doesn't work directly because the indices don't align.
    
    # Alternative: Use random search with structured initialization
    rng = np.random.RandomState(42)
    
    # Try rank 25 with sparse random initialization
    for target_rank in [25, 26, 27, 28, 29, 30]:
        if time.time() > deadline - 10:
            break
        
        max_attempts = 50
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Sparse random initialization with values in {-1, 0, 1}
            U = rng.randint(-1, 2, (n*m, target_rank)).astype(float)
            V = rng.randint(-1, 2, (m*p, target_rank)).astype(float)
            W = rng.randint(-1, 2, (p*n, target_rank)).astype(float)
            
            # Make them sparser
            U[np.abs(U) < 1] = 0
            V[np.abs(V) < 1] = 0
            W[np.abs(W) < 1] = 0
            
            if verify_decomposition(U, V, W, T):
                best_rank = target_rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                break
        
        if found:
            break
    
    # If we found a lower rank, return it
    if best_rank < schoolbook_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Otherwise, return the schoolbook decomposition
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
