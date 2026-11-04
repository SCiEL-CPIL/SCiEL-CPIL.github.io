import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank28(n, m, p):
    """Build the known rank-28 decomposition."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, schoolbook (10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j in {0,1}, k=4, schoolbook (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: j in {0,1}, k in {0,1}, Strassen (7 terms)
    # M1 = (A00+A11)(B00+B11)
    U[0*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+0][r] = 1; V[1*p+1][r] = 1
    W[0*n+0][r] = 1; W[1*n+1][r] = 1
    r += 1
    # M2 = (A10+A11)B00
    U[1*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+0][r] = 1
    W[0*n+1][r] = 1; W[1*n+1][r] = -1
    r += 1
    # M3 = A00(B11-B01)
    U[0*m+0][r] = 1
    V[0*p+1][r] = 1; V[1*p+1][r] = -1
    W[1*n+0][r] = 1; W[1*n+1][r] = 1
    r += 1
    # M4 = A11(B00-B10)
    U[1*m+1][r] = 1
    V[1*p+0][r] = 1; V[0*p+0][r] = -1
    W[0*n+0][r] = 1; W[0*n+1][r] = 1
    r += 1
    # M5 = (A00+A01)B11
    U[0*m+0][r] = 1; U[0*m+1][r] = 1
    V[1*p+1][r] = 1
    W[0*n+0][r] = -1; W[1*n+0][r] = 1
    r += 1
    # M6 = (A10-A00)(B00+B01)
    U[1*m+0][r] = 1; U[0*m+0][r] = -1
    V[0*p+0][r] = 1; V[0*p+1][r] = 1
    W[1*n+1][r] = 1
    r += 1
    # M7 = (A01-A11)(B10+B11)
    U[0*m+1][r] = 1; U[1*m+1][r] = -1
    V[1*p+0][r] = 1; V[1*p+1][r] = 1
    W[0*n+0][r] = 1
    r += 1
    
    # Part 4: j in {0,1}, k in {2,3}, Strassen (7 terms)
    # M1 = (A00+A11)(B02+B13)
    U[0*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+2][r] = 1; V[1*p+3][r] = 1
    W[2*n+0][r] = 1; W[3*n+1][r] = 1
    r += 1
    # M2 = (A10+A11)B02
    U[1*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+2][r] = 1
    W[2*n+1][r] = 1; W[3*n+1][r] = -1
    r += 1
    # M3 = A00(B13-B03)
    U[0*m+0][r] = 1
    V[0*p+3][r] = 1; V[1*p+3][r] = -1
    W[3*n+0][r] = 1; W[3*n+1][r] = 1
    r += 1
    # M4 = A11(B02-B12)
    U[1*m+1][r] = 1
    V[1*p+2][r] = 1; V[0*p+2][r] = -1
    W[2*n+0][r] = 1; W[2*n+1][r] = 1
    r += 1
    # M5 = (A00+A01)B13
    U[0*m+0][r] = 1; U[0*m+1][r] = 1
    V[1*p+3][r] = 1
    W[2*n+0][r] = -1; W[3*n+0][r] = 1
    r += 1
    # M6 = (A10-A00)(B02+B03)
    U[1*m+0][r] = 1; U[0*m+0][r] = -1
    V[0*p+2][r] = 1; V[0*p+3][r] = 1
    W[3*n+1][r] = 1
    r += 1
    # M7 = (A01-A11)(B12+B13)
    U[0*m+1][r] = 1; U[1*m+1][r] = -1
    V[1*p+2][r] = 1; V[1*p+3][r] = 1
    W[2*n+0][r] = 1
    r += 1
    
    return U, V, W, rank

def _try_reduce_rank(U, V, W, n, m, p, target_rank, time_limit=60):
    """
    Try to reduce the rank by finding linear dependencies.
    We look for a vector c such that sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0
    in the tensor space, which would allow us to remove one term.
    """
    start_time = time.time()
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    current_rank = len(U[0])
    
    # Try to find a null vector in the tensor space
    # Each rank-1 term contributes a tensor T_r where T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want sum_r c_r * T_r = 0
    # This is a linear system: for each (i,j,k), sum_r c_r * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 0
    
    num_equations = n * m * p
    # Build the coefficient matrix: A[equation, r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    A = np.zeros((num_equations, current_rank))
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(current_rank):
                    A[eq_idx, r] = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                eq_idx += 1
    
    # Find null space of A
    try:
        _, S, Vt = np.linalg.svd(A)
        # Null space vectors are rows of Vt corresponding to near-zero singular values
        tol = 1e-10
        null_mask = S < tol
        if np.any(null_mask):
            null_space = Vt[null_mask, :]
            print(f"Found {len(null_space)} null space vectors")
            for null_vec in null_space:
                # Check if removing one term works
                for remove_idx in range(current_rank):
                    if abs(null_vec[remove_idx]) > 1e-8:
                        # Try to eliminate term remove_idx
                        # New rank = current_rank - 1
                        new_U = [row[:] for row in U]
                        new_V = [row[:] for row in V]
                        new_W = [row[:] for row in W]
                        
                        # Remove column remove_idx from all matrices
                        new_rank = current_rank - 1
                        new_U = [[new_U[i][r] for r in range(new_rank) if r != remove_idx] for i in range(len(new_U))]
                        new_V = [[new_V[i][r] for r in range(new_rank) if r != remove_idx] for i in range(len(new_V))]
                        new_W = [[new_W[i][r] for r in range(new_rank) if r != remove_idx] for i in range(len(new_W))]
                        
                        # Verify
                        if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                            print(f"Successfully reduced to rank {new_rank}!")
                            return new_U, new_V, new_W, new_rank
    except Exception as e:
        print(f"Error in null space computation: {e}")
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin before 150s limit
    
    # Start with known rank-28 decomposition
    U, V, W, rank = _build_rank28(n, m, p)
    
    # Verify the rank-28 solution
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Rank-28 decomposition verification failed")
    
    # Try to reduce rank
    if time.time() < deadline - 30:
        result = _try_reduce_rank(U, V, W, n, m, p, rank - 1, time_limit=60)
        if result is not None:
            U, V, W, rank = result
            print(f"Reduced to rank {rank}")
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to known good solution
        U, V, W, rank = _build_rank28(n, m, p)
    
    return {"U": U, "V": V, "W": W}
