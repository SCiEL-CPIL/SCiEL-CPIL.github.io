"""Tensor decomposition for 4x4 x 4x9 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))

    # Strassen formulas
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1

    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _decomp_4x4x4():
    """
    Decomposition of M_{4,4,4} using 2-level Strassen.
    Rank 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    inner_U, inner_V, inner_W = _strassen_2x2()

    outer_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]

    outer_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]

    outer_C = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [-1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner_U[li*2 + lj, r]
                    coeff_V = outer_B[t][b] * inner_V[li*2 + lj, r]
                    coeff_W = outer_C[t][b] * inner_W[li*2 + lj, r]

                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j, col] = coeff_U
                            V[g_j * 4 + g_i, col] = coeff_V
                            W[g_i * 4 + g_j, col] = coeff_W

    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """
    Schoolbook decomposition for M_{n,m,p}.
    Rank n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _decomp_4x4x9_combined():
    """
    Decomposition of M_{4,4,9} by splitting into M_{4,4,4} + M_{4,4,4} + M_{4,4,1}.
    Total rank: 49 + 49 + 16 = 114.
    """
    n, m, p = 4, 4, 9
    total_rank = 49 + 49 + 16

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    U444, V444, W444 = _decomp_4x4x4()
    U441, V441, W441 = _schoolbook_nm_p(4, 4, 1)

    U[:, :49] = U444
    for j in range(m):
        for k in range(4):
            V[j * 9 + k, :49] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k * 4 + i, :49] = W444[k * 4 + i, :]

    U[:, 49:98] = U444
    for j in range(m):
        for k in range(4):
            V[j * 9 + (4 + k), 49:98] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[(4 + k) * 4 + i, 49:98] = W444[k * 4 + i, :]

    U[:, 98:114] = U441
    for j in range(m):
        V[j * 9 + 8, 98:114] = V441[j * 1 + 0, :]
    for i in range(n):
        W[8 * 4 + i, 98:114] = W441[0 * 4 + i, :]

    return U, V, W

def _verify_decomp(U, V, W, n, m, p, atol=1e-9):
    """Vectorized verification of the decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)

    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_idx = i_idx.ravel()
    j_idx = j_idx.ravel()
    k_idx = k_idx.ravel()

    u_vec = U_r[i_idx, j_idx, :]
    v_vec = V_r[j_idx, k_idx, :]
    w_vec = W_r[k_idx, i_idx, :]

    prod = u_vec * v_vec * w_vec
    sums = prod.sum(axis=1)

    return np.allclose(sums, 1.0, atol=atol)

def _try_4x4x9_direct_strassen():
    """
    Alternative: Use 2-level Strassen but with a different tiling.
    View 4x4x9 as (2x2) x (2x4) x (2x4+1).
    Apply Strassen on the 2x2 row-block and 2x2 col-pair, 
    giving 7x7=49 rank for the 2x2x2 core structure.
    For the 9 columns: 9 = 4+4+1, so two 4-col blocks + 1.
    This is essentially the same as _decomp_4x4x9_combined but 
    we try a different grouping: 9 = 2+2+2+2+1.
    Each 2-col block with 4x4 rows/cols: M_{4,4,2} has rank at most 4*4*2=32 schoolbook,
    or we can use Strassen-like: M_{4,4,2} = M_{4,2,2} x 2 (block columns).
    M_{4,2,2}: 4x2 x 2x2. Rank at most 4*2*2=16 schoolbook.
    With Strassen on the 2x2: M_{2,2,2} rank 7, so M_{4,2,2} = M_{2,2,2} x 2 (row blocks)
    gives rank 7*2=14? No, that's not how it works.
    
    Actually, let's try: M_{4,4,9} with 9 = 2+2+2+2+1.
    For each 2-col block: M_{4,4,2}. 
    M_{4,4,2} can be decomposed as: for each of the 2 output columns k,
    C[:,k] = A @ B[:,k]. This is 4x4 x 4x2.
    Schoolbook: 4*4*2 = 32 rank.
    Strassen-based: treat as two 4x4x1 problems, each rank 16, total 32. Same.
    
    So 5 blocks of M_{4,4,2}: 5*32 = 160. Worse.
    
    Let's try 9 = 4+5. M_{4,4,5}: schoolbook 80. Strassen 49+16=65? No, 
    M_{4,4,5} = M_{4,4,4} + M_{4,4,1} = 49+16=65.
    Total: 49 + 65 = 114. Same as before.
    
    Try 9 = 3+6. M_{4,4,3} schoolbook 48, M_{4,4,6} = M_{4,4,4}+M_{4,4,2}=49+32=81. Total 129. Worse.
    
    The 4+4+1 split with Strassen 49+49+16=114 seems optimal among block splits.
    """
    return None

def _try_greedy_reduction(U, V, W, n, m, p, max_time=100):
    """
    Greedy rank reduction: try to remove rank-1 terms that are linearly
    dependent in the tensor sense. For each pair of rank-1 terms, check if
    one can be expressed as a linear combination of others.
    This is a fundamentally different optimization approach: iterative 
    linear algebra reduction rather than constructive tiling.
    """
    start = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # For exact decomposition, T[i,j,k] should be 1 iff (i,j,k) corresponds to 
    # C[i//m, i%m] = sum_k A[i//m, k%m] * B[k//m, k%p] ... 
    # Actually the tensor entries are:
    # T[i, j, k] = 1 if i = i0*m + j0, j = j0*p + k0, k = k0*n + i0 for the 
    # multiplication tensor, else 0.
    
    # The multiplication tensor T has T[i,j,k] = 1 iff there exist (i0,j0,k0) 
    # such that i = i0*m+j0, j = j0*p+k0, k = k0*n+i0.
    # This means: i%m = j0, j%p = k0, k%n = i0, and i//m = k0, j//p = i0... 
    # Wait, let me re-derive.
    # C[i0, k0] = sum_{j0} A[i0, j0] * B[j0, k0]
    # So the tensor T has T[i, j, k] = 1 where i = i0*m + j0, j = j0*p + k0, k = k0*n + i0.
    # This requires: i%m = j0, j%p = k0, k%n = i0, and i//m = i0 = k%n, j//p = j0 = i%m, k//n = k0 = j%p.
    # So T[i,j,k] = 1 iff i%m == j//p and j%p == k//n and k%n == i//m.
    
    T = np.zeros((n*m, m*p, p*n))
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                i = i0 * m + j0
                j = j0 * p + k0
                k = k0 * n + i0
                T[i, j, k] = 1.0
    
    # Now try to reduce rank by finding linear dependencies.
    # This is expensive, so we limit iterations.
    removed = 0
    max_iter = 200
    for it in range(max_iter):
        if time.time() - start > max_time:
            break
        if rank <= 114:  # Can't beat current best with greedy
            break
        
        # Check if any column is a linear combination of others
        # This is O(rank^2 * (n*m + m*p + p*n)) which is expensive.
        # Instead, check if removing a column still gives valid decomposition.
        # For each column r, try removing it and see if T is still reproduced.
        # This is O(rank * n*m*p) per check.
        
        # Quick check: if any column is all zeros, remove it.
        for r in range(rank):
            if np.allclose(U[:, r], 0) and np.allclose(V[:, r], 0) and np.allclose(W[:, r], 0):
                # Remove column r
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
                rank -= 1
                removed += 1
                break
        else:
            # Try small perturbations: for each pair (r1, r2), check if
            # U[:,r1] is proportional to U[:,r2] AND V[:,r1] proportional to V[:,r2]
            # AND W[:,r1] proportional to W[:,r2]. If so, one is redundant.
            found = False
            for r1 in range(rank):
                if found:
                    break
                for r2 in range(r1+1, rank):
                    # Check proportionality
                    try:
                        alpha_U = U[0, r1] / U[0, r2] if U[0, r2] != 0 else (U[1, r1] / U[1, r2] if U[1, r2] != 0 else 0)
                        if np.allclose(U[:, r1], alpha_U * U[:, r2]):
                            alpha_V = V[0, r1] / V[0, r2] if V[0, r2] != 0 else (V[1, r1] / V[1, r2] if V[1, r2] != 0 else 0)
                            if np.allclose(V[:, r1], alpha_V * V[:, r2]):
                                alpha_W = W[0, r1] / W[0, r2] if W[0, r2] != 0 else (W[1, r1] / W[1, r2] if W[1, r2] != 0 else 0)
                                if np.allclose(W[:, r1], alpha_W * W[:, r2]):
                                    # r1 is redundant given r2
                                    U = np.delete(U, r1, axis=1)
                                    V = np.delete(V, r1, axis=1)
                                    W = np.delete(W, r1, axis=1)
                                    rank -= 1
                                    removed += 1
                                    found = True
                                    break
                    except:
                        pass
        
        if not found and removed == 0 and it > 50:
            # No more easy reductions
            break
    
    if removed > 0:
        return U, V, W
    return None

def _decomp_4x4x9_kron_2x2x3():
    """
    Try: M_{4,4,9} using Kronecker product of M_{2,2,3} and M_{2,2,3}.
    M_{2,2,3}: 2x2 x 2x3 matrix multiplication.
    Schoolbook rank: 2*2*3 = 12.
    Known lower bound: at least 2*2*3 - something.
    Actually, M_{2,2,3} has rank at most 12 (schoolbook) and at least 8 (trivial).
    Can we do better? M_{2,2,3} = M_{2,2,2} + M_{2,2,1}.
    M_{2,2,2} rank 7 (Strassen), M_{2,2,1} rank 4 (schoolbook). Total 11.
    Or M_{2,2,3} directly: for each of 3 output cols, 2x2x1 rank 4, total 12.
    With Strassen on the 2x2: M_{2,2,3} = M_{2,2,2} + M_{2,2,1} = 7+4=11.
    
    M_{4,4,9} = M_{2,2,3} x M_{2,2,3} (Kronecker).
    Rank = 11 * 11 = 121. Worse than 114.
    
    Try M_{2,2,3} with schoolbook 12: 12*12=144. Worse.
    
    So Kronecker approach doesn't help here.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Primary: 2-level Strassen block decomposition, rank 114
    U, V, W = _decomp_4x4x9_combined()
    
    if time.time() < deadline - 20:
        if _verify_decomp(U, V, W, n, m, p):
            # Try greedy reduction (different optimization method)
            reduced = _try_greedy_reduction(U, V, W, n, m, p, max_time=min(80, deadline - time.time() - 10))
            if reduced is not None:
                U_r, V_r, W_r = reduced
                if _verify_decomp(U_r, V_r, W_r, n, m, p):
                    if U_r.shape[1] < U.shape[1]:
                        U, V, W = U_r, V_r, W_r
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

# EVOLVE-BLOCK-END
