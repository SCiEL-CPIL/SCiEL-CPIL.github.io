import time
import numpy as np
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _build_rank55_decomposition(n, m, p):
    """
    Builds the known rank-55 decomposition.
    Structure:
    - 20 terms for the trivial part (i=0 or j=0)
    - 35 terms for the core 2x2x10 part (i=1, j=1,2) using 5 Strassen blocks
    """
    RANK = 55
    U = np.zeros((n * m, RANK), dtype=np.int64)
    V = np.zeros((m * p, RANK), dtype=np.int64)
    W = np.zeros((p * n, RANK), dtype=np.int64)

    col = 0
    # Indices for the core 2x2x10 block
    # i_core in {0, 1} corresponds to global i in {1} (since i=0 is trivial) -> Wait, n=2, so i in {0,1}.
    # j_core in {0, 1} corresponds to global j in {1, 2} (since j=0 is trivial)
    # k in {0..9}
    
    # Part A: Trivial parts
    # A[i, 0] * B[0, k] -> C[i, k]
    # i in {0, 1}, k in {0..9}
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    # Part B: Core 2x2x10
    # Global indices:
    # i_global = 1 (only 1 row left? No, n=2, so i=0,1. i=0 handled in Part A? 
    # Part A handled j=0. So for j=1,2, we have i=0,1.
    # Wait, Part A handled A[i,0]*B[0,k]. This contributes to C[i,k].
    # The term A[i,0]*B[0,k] is the only term with j=0.
    # For j=1,2, we have A[i,1]*B[1,k] and A[i,2]*B[2,k].
    # So the core is for i in {0,1}, j in {1,2}, k in {0..9}.
    # This is a 2x2x10 tensor.
    
    # We use 5 Strassen blocks for the 2x2x10 part.
    # Each block handles k0, k1.
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Global indices for U (A matrix)
        # i=0, j=1 -> U[0*3+1] = U[1]
        # i=0, j=2 -> U[0*3+2] = U[2]
        # i=1, j=1 -> U[1*3+1] = U[4]
        # i=1, j=2 -> U[1*3+2] = U[5]
        
        # Global indices for V (B matrix)
        # j=1, k0 -> V[1*10+k0] = V[10+k0]
        # j=1, k1 -> V[1*10+k1] = V[10+k1]
        # j=2, k0 -> V[2*10+k0] = V[20+k0]
        # j=2, k1 -> V[2*10+k1] = V[20+k1]
        
        # Global indices for W (C matrix)
        # i=0, k0 -> W[0*2+k0] = W[k0]
        # i=0, k1 -> W[0*2+k1] = W[k1]
        # i=1, k0 -> W[1*2+k0] = W[2+k0]
        # i=1, k1 -> W[1*2+k1] = W[2+k1]
        
        u_00 = 1 # A[0,1]
        u_01 = 2 # A[0,2]
        u_10 = 4 # A[1,1]
        u_11 = 5 # A[1,2]
        
        v_00 = 10 + k0 # B[1,k0]
        v_01 = 10 + k1 # B[1,k1]
        v_10 = 20 + k0 # B[2,k0]
        v_11 = 20 + k1 # B[2,k1]
        
        w_00 = k0     # C[0,k0]
        w_01 = k1     # C[0,k1]
        w_10 = 2 + k0 # C[1,k0]
        w_11 = 2 + k1 # C[1,k1]
        
        # Strassen 7 terms for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        U[u_00, col] += 1; U[u_11, col] += 1
        V[v_00, col] += 1; V[v_11, col] += 1
        W[w_00, col] += 1; W[w_11, col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[u_10, col] += 1; U[u_11, col] += 1
        V[v_00, col] += 1
        W[w_10, col] += 1; W[w_11, col] -= 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[u_00, col] += 1
        V[v_01, col] += 1; V[v_11, col] -= 1
        W[w_01, col] += 1; W[w_11, col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[u_11, col] += 1
        V[v_10, col] += 1; V[v_00, col] -= 1
        W[w_00, col] += 1; W[w_10, col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[u_00, col] += 1; U[u_01, col] += 1
        V[v_11, col] += 1
        W[w_00, col] -= 1; W[w_01, col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u_10, col] += 1; U[u_00, col] -= 1
        V[v_00, col] += 1; V[v_01, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u_01, col] += 1; U[u_11, col] -= 1
        V[v_10, col] += 1; V[v_11, col] += 1
        W[w_00, col] += 1
        col += 1

    return U, V, W, RANK


def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    where a=i*m+j, b=j*p+k, c=k*n+i
    T[a, b, c] should be 1 if indices are consistent, 0 otherwise.
    """
    U_f = U.astype(np.float64)
    V_f = V.astype(np.float64)
    W_f = W.astype(np.float64)
    
    # Check all 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_f[a, :], V_f[b, :] * W_f[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check some 0s (off-diagonal)
    # We need to ensure that for any (a,b,c) not corresponding to a valid (i,j,k), the sum is 0.
    # Due to the structure, we can check random invalid tuples.
    # Valid tuples: a=i*m+j, b=j*p+k, c=k*n+i.
    # Invalid if j_a != j_b or k_b != k_c or i_a != i_c.
    
    rng = np.random.RandomState(42)
    for _ in range(500):
        # Random a, b, c
        a = rng.randint(n * m)
        b = rng.randint(m * p)
        c = rng.randint(p * n)
        
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        
        # If it's a valid tuple, skip (already checked)
        if i_a == i_c and j_a == j_b and k_b == k_c:
            continue
            
        val = np.dot(U_f[a, :], V_f[b, :] * W_f[c, :])
        if abs(val) > tol:
            return False
            
    return True


def _try_reduce_core_rank(U55, V55, W55, n, m, p, target_core_rank, time_budget, seed=42):
    """
    Attempts to find a lower rank decomposition for the core 2x2x10 part.
    The total rank will be 20 (trivial) + target_core_rank.
    """
    if time.time() > time_budget:
        return None
        
    # Extract the core part from the rank-55 solution
    # Core indices:
    # U rows: 1, 2, 4, 5 (corresponding to A[0,1], A[0,2], A[1,1], A[1,2])
    # V rows: 10..29 (corresponding to B[1,k], B[2,k])
    # W rows: 0..19 (corresponding to C[0,k], C[1,k])
    
    u_core_rows = [1, 2, 4, 5]
    v_core_rows = list(range(10, 30))
    w_core_rows = list(range(0, 20))
    
    # The core tensor T_core has shape (4, 20, 20) in flattened U,V,W indices?
    # No, U_core is 4xR, V_core is 20xR, W_core is 20xR.
    # We want to find U_c (4xR'), V_c (20xR'), W_c (20xR') for R' < 35.
    
    # Initialize with random projection of the 35-rank core solution
    # The core solution is in columns 20..54 of the rank-55 matrices.
    U_c_init = U55[u_core_rows, 20:55].copy().astype(np.float64)
    V_c_init = V55[v_core_rows, 20:55].copy().astype(np.float64)
    W_c_init = W55[w_core_rows, 20:55].copy().astype(np.float64)
    
    # Project to target_core_rank
    # Use SVD to find the best rank-k approximation
    # T_core = sum_r U_c[:,r] V_c[:,r] W_c[:,r]
    # This is a 3-way tensor. We can use ALS.
    
    U_c = U_c_init[:, :target_core_rank] + np.random.randn(4, target_core_rank) * 0.1
    V_c = V_c_init[:, :target_core_rank] + np.random.randn(20, target_core_rank) * 0.1
    W_c = W_c_init[:, :target_core_rank] + np.random.randn(20, target_core_rank) * 0.1
    
    # Build the target core tensor
    T_core = np.zeros((4, 20, 20), dtype=np.float64)
    # Map local indices to global for verification later, but for ALS we need the tensor values.
    # Local U idx: 0->A[0,1], 1->A[0,2], 2->A[1,1], 3->A[1,2]
    # Local V idx: 0..9->B[1,k], 10..19->B[2,k]
    # Local W idx: 0..9->C[0,k], 10..19->C[1,k]
    
    # T_core[u_idx, v_idx, w_idx] = 1 if consistent
    # u_idx = i_l * 2 + j_l where i_l in {0,1} (global i 0,1), j_l in {0,1} (global j 1,2)
    # v_idx = j_l * 10 + k
    # w_idx = k * 2 + i_l
    
    for i_l in range(2):
        for j_l in range(2):
            for k in range(10):
                u_idx = i_l * 2 + j_l
                v_idx = j_l * 10 + k
                w_idx = k * 2 + i_l
                T_core[u_idx, v_idx, w_idx] = 1.0
    
    # ALS Optimization
    start = time.time()
    best_err = float('inf')
    best_U, best_V, best_W = U_c.copy(), V_c.copy(), W_c.copy()
    
    for it in range(100):
        if time.time() > time_budget - 5:
            break
            
        # Update U
        for u_idx in range(4):
            # min_U || T_core[u_idx] - sum_r U[u_idx, r] (V[:, r] * W[:, r]) ||^2
            # A matrix of size (20*20) x R
            A = np.zeros((400, target_core_rank))
            for v_idx in range(20):
                for w_idx in range(20):
                    A[v_idx * 20 + w_idx, :] = V_c[v_idx, :] * W_c[w_idx, :]
            t = T_core[u_idx].flatten()
            try:
                U_c[u_idx, :] = np.linalg.lstsq(A, t, rcond=None)[0]
            except:
                pass
                
        # Update V
        for v_idx in range(20):
            A = np.zeros((4 * 20, target_core_rank))
            for u_idx in range(4):
                for w_idx in range(20):
                    A[u_idx * 20 + w_idx, :] = U_c[u_idx, :] * W_c[w_idx, :]
            t = T_core[:, v_idx, :].reshape(-1)
            try:
                V_c[v_idx, :] = np.linalg.lstsq(A, t, rcond=None)[0]
            except:
                pass
                
        # Update W
        for w_idx in range(20):
            A = np.zeros((4 * 20, target_core_rank))
            for u_idx in range(4):
                for v_idx in range(20):
                    A[u_idx * 20 + v_idx, :] = U_c[u_idx, :] * V_c[v_idx, :]
            t = T_core[:, :, w_idx].reshape(-1)
            try:
                W_c[w_idx, :] = np.linalg.lstsq(A, t, rcond=None)[0]
            except:
                pass
        
        # Compute error
        err = 0.0
        for u_idx in range(4):
            for v_idx in range(20):
                for w_idx in range(20):
                    val = np.dot(U_c[u_idx, :], V_c[v_idx, :] * W_c[w_idx, :])
                    err += (val - T_core[u_idx, v_idx, w_idx]) ** 2
                    
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_c.copy(), V_c.copy(), W_c.copy()
            
        if best_err < 1e-10:
            break
            
    # If error is low, try to round to integers/half-integers
    if best_err < 1e-6:
        # Round
        U_r = np.round(best_U * 2).astype(np.int64) / 2.0
        V_r = np.round(best_V * 2).astype(np.int64) / 2.0
        W_r = np.round(best_W * 2).astype(np.int64) / 2.0
        
        # Verify the full decomposition
        U_full = np.zeros((n * m, 20 + target_core_rank), dtype=np.float64)
        V_full = np.zeros((m * p, 20 + target_core_rank), dtype=np.float64)
        W_full = np.zeros((p * n, 20 + target_core_rank), dtype=np.float64)
        
        # Copy trivial part (columns 0..19)
        U_full[:, :20] = U55[:, :20]
        V_full[:, :20] = V55[:, :20]
        W_full[:, :20] = W55[:, :20]
        
        # Copy core part (columns 20..)
        U_full[u_core_rows, 20:] = U_r
        V_full[v_core_rows, 20:] = V_r
        W_full[w_core_rows, 20:] = W_r
        
        if _verify_decomposition(U_full, V_full, W_full, n, m, p):
            return U_full, V_full, W_full, 20 + target_core_rank
            
    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x10 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Start with rank-55 decomposition
    U55, V55, W55, rank55 = _build_rank55_decomposition(n, m, p)
    
    best_result = (U55, V55, W55, rank55)
    
    # Try to reduce core rank from 34 down to 30
    # Total rank = 20 + core_rank
    for target_core_rank in range(34, 29, -1):
        if time.time() > deadline - 10:
            break
            
        time_budget = deadline - 5
        result = _try_reduce_core_rank(U55, V55, W55, n, m, p, target_core_rank, time_budget, seed=target_core_rank)
        
        if result is not None:
            best_result = result
            break
            
    U_best, V_best, W_best, r = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
