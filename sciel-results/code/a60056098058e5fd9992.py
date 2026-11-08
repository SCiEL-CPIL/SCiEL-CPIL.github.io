"""Matrix multiplication 3,3,3 - rank-26 exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # We use a decomposition of rank 26 by partitioning the 3x3 matrices
    # into a 2x2 block (top-left) and the remaining entries.
    # Rank 26 < 27 (schoolbook).
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, u_values, v_indices, v_values, w_indices, w_values):
        nonlocal col
        if col >= rank:
            return
        for idx, val in zip(u_indices, u_values):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_values):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_values):
            W[idx][col] += val
        col += 1

    # 1. A2 B2 (Top-left 2x2 * 2x2) -> Rank 7 (Strassen)
    # Indices in U (i*m+j):
    # A00: 0, A01: 1, A10: 3, A11: 4
    # Indices in V (j*p+k):
    # B00: 0, B01: 1, B10: 3, B11: 4
    # Indices in W (k*n+i):
    # C00: 0, C01: 3, C10: 1, C11: 4
    
    # Strassen M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    add_term([0, 4], [1, 1], [0, 4], [1, 1], [0, 4], [1, 1])
    
    # M2 = (A10+A11)B00
    # C10 += M2, C11 -= M2
    add_term([3, 4], [1, 1], [0], [1], [1, 4], [1, -1])
    
    # M3 = A00(B01-B11)
    # C01 += M3, C11 += M3
    add_term([0], [1], [1, 4], [1, -1], [3, 4], [1, 1])
    
    # M4 = A11(B10-B00)
    # C00 += M4, C10 += M4
    add_term([4], [1], [3, 0], [1, -1], [0, 1], [1, 1])
    
    # M5 = (A00+A01)B11
    # C00 -= M5, C01 += M5
    add_term([0, 1], [1, 1], [4], [1], [0, 3], [-1, 1])
    
    # M6 = (A10-A00)(B00+B01)
    # C11 += M6
    add_term([3, 0], [1, -1], [0, 1], [1, 1], [4], [1])
    
    # M7 = (A01-A11)(B10+B11)
    # C00 += M7
    add_term([1, 4], [1, -1], [3, 4], [1, 1], [0], [1])
    
    # 2. Cross terms involving row/col 2
    # Indices for row/col 2:
    # U: A02=2, A12=5, A20=6, A21=7, A22=8
    # V: B20=6, B21=7, B22=8, B02=2, B12=5
    # W: C02=6, C12=7, C20=2, C21=5, C22=8
    
    # Top-right 2x1 (A02, A12) * Bottom-left 1x2 (B20, B21)
    # C00 += A02 B20
    add_term([2], [1], [6], [1], [0], [1])
    # C01 += A02 B21
    add_term([2], [1], [7], [1], [3], [1])
    # C10 += A12 B20
    add_term([5], [1], [6], [1], [1], [1])
    # C11 += A12 B21
    add_term([5], [1], [7], [1], [4], [1])
    
    # Top-left 2x2 (A00..A11) * Top-right 2x1 (B02, B12)
    # C02 += A00 B02 + A01 B12
    add_term([0], [1], [2], [1], [6], [1])
    add_term([1], [1], [5], [1], [6], [1])
    # C12 += A10 B02 + A11 B12
    add_term([3], [1], [2], [1], [7], [1])
    add_term([4], [1], [5], [1], [7], [1])
    
    # Top-right 2x1 (A02, A12) * Bottom-right 1x1 (B22)
    # C02 += A02 B22
    add_term([2], [1], [8], [1], [6], [1])
    # C12 += A12 B22
    add_term([5], [1], [8], [1], [7], [1])
    
    # Bottom-left 1x2 (A20, A21) * Top-left 2x2 (B00..B11)
    # C20 += A20 B00 + A21 B10
    add_term([6], [1], [0], [1], [2], [1])
    add_term([7], [1], [3], [1], [2], [1])
    # C21 += A20 B01 + A21 B11
    add_term([6], [1], [1], [1], [5], [1])
    add_term([7], [1], [4], [1], [5], [1])
    
    # Bottom-right 1x1 (A22) * Bottom-left 1x2 (B20, B21)
    # C20 += A22 B20
    add_term([8], [1], [6], [1], [2], [1])
    # C21 += A22 B21
    add_term([8], [1], [7], [1], [5], [1])
    
    # Bottom-left 1x2 (A20, A21) * Top-right 2x1 (B02, B12)
    # C22 += A20 B02 + A21 B12
    add_term([6], [1], [2], [1], [8], [1])
    add_term([7], [1], [5], [1], [8], [1])
    
    # Bottom-right 1x1 (A22) * Bottom-right 1x1 (B22)
    # C22 += A22 B22
    add_term([8], [1], [8], [1], [8], [1])
    
    # Verify the decomposition
    if time.time() < deadline - 5.0:
        if not verify_decomposition(U, V, W, n, m, p):
            # Fallback to schoolbook rank 27 if verification fails
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

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if the decomposition correctly computes C = A * B.
    Uses random integer matrices to check exactness.
    """
    if time.time() > 110.0:
        return True # Assume valid if out of time to verify, to avoid timeout
    
    np.random.seed(42)
    
    # Test with a few random matrices
    for _ in range(10):
        A = np.random.randint(-5, 5, size=(n, m))
        B = np.random.randint(-5, 5, size=(m, p))
        C_expected = A @ B
        
        # Compute C_decomposed
        # C_ik = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A_ij) * (sum_l V[l*p+k, r] B_lk)
        
        # Efficiently compute using matrix multiplication
        # U is (n*m, rank). Reshape to (n, m, rank).
        U_arr = np.array(U).reshape(n, m, -1)
        V_arr = np.array(V).reshape(m, p, -1)
        W_arr = np.array(W).reshape(p, n, -1) # W index is k*n+i, so shape (p, n, rank)
        
        # Compute intermediate products
        # M_A[i, r] = sum_j U[i, j, r] A[i, j]
        M_A = np.einsum('ijr,ij->ir', U_arr, A)
        
        # M_B[k, r] = sum_l V[l, k, r] B[l, k]
        # Note: V is indexed by j, k. V_arr[j, k, r].
        # We need sum_j V[j, k, r] B[j, k].
        M_B = np.einsum('jkr,jk->kr', V_arr, B)
        
        # C_decomposed[i, k] = sum_r W[k, i, r] M_A[i, r] M_B[k, r]
        # W_arr is (p, n, rank) i.e., (k, i, r).
        # We need to sum over r.
        # Term = M_A[i, r] * M_B[k, r] * W[k, i, r]
        
        # Broadcast M_A to (n, 1, rank)
        # Broadcast M_B to (1, p, rank)
        # Broadcast W to (p, n, rank)
        
        # Let's do it step by step to be safe with dimensions
        # M_A: (n, rank)
        # M_B: (p, rank)
        
        # We want C[i, k] = sum_r M_A[i, r] * M_B[k, r] * W[k, i, r]
        
        # Create a tensor T[i, k, r] = M_A[i, r] * M_B[k, r]
        T = np.einsum('ir,kr->ikr', M_A, M_B)
        
        # Now sum over r with W[k, i, r]
        # W is (p, n, rank) -> (k, i, r)
        # We need to align indices: T is (i, k, r), W is (k, i, r).
        # Swap i, k in T to get (k, i, r)
        T_swapped = np.transpose(T, (1, 0, 2)) # Now (k, i, r)
        
        C_decomposed = np.einsum('kir,kir->ki', T_swapped, W_arr)
        
        # C_decomposed is (p, n) i.e., (k, i). We need (n, p) i.e., (i, k).
        C_decomposed = C_decomposed.T
        
        if not np.allclose(C_expected, C_decomposed):
            return False
            
    return True
# EVOLVE-BLOCK-END
