"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition reconstructs the MM tensor T[i,j,k]=1."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    
    for c in range(rank):
        U_col = U[:, c].reshape(n, m)
        V_col = V[:, c].reshape(m, p)
        W_col = W[:, c].reshape(p, n)
        
        W_col_T = W_col.T  # shape (n, p)
        
        tmp1 = np.einsum('ij,jk->ijk', U_col, V_col)
        tmp2 = tmp1 * W_col_T[:, np.newaxis, :]
        T_computed += tmp2
    
    T_expected = np.ones((n, m, p), dtype=np.float64)
    diff = np.abs(T_computed - T_expected)
    return np.max(diff) < tol


def _construct_schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W, "rank": rank}


def _strassen_2x2_2x2_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 = 2x2 multiplication.
    Returns U (4x7), V (4x7), W (4x7) such that
    sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = 1 for all i,j,k in {0,1}
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # i=0,j=0
        [0, 0, 0, 0, 1, 0, 1],    # i=0,j=1
        [0, 1, 0, 0, 0, 1, 0],    # i=1,j=0
        [1, 1, -1, 1, 0, 0, -1]   # i=1,j=1
    ], dtype=np.float64)
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # j=0,k=0
        [0, 0, 0, 1, 0, 1, 0],    # j=0,k=1
        [0, 0, 1, 0, 0, 0, 1],    # j=1,k=0
        [1, 0, -1, 0, 1, 0, 1]    # j=1,k=1
    ], dtype=np.float64)
    
    W = np.array([
        [1, 0, 0, -1, 1, 0, 1],   # k=0,i=0
        [0, 1, 0, 0, 0, 0, 0],    # k=0,i=1
        [0, 0, 1, 0, 1, 0, 0],    # k=1,i=0
        [1, 0, 0, 1, 0, -1, -1]   # k=1,i=1
    ], dtype=np.float64)
    
    return U, V, W


def _build_4x4x4_strassen():
    """
    Build a rank-28 decomposition for the 4x4x4 all-ones tensor using Strassen.
    
    Partition i into blocks {0,1} and {2,3}, k into blocks {0,1} and {2,3}.
    For each (ib, kb) pair, use Strassen's 7-product method for the 2x4x2 sub-tensor.
    """
    n, m, p = 4, 4, 4
    U_s, V_s, W_s = _strassen_2x2_2x2_2x2()
    
    total_rank = 4 * 7  # 4 (i,k) blocks * 7 Strassen terms
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    for ib in range(2):
        for kb in range(2):
            for c in range(7):
                col = ib * 14 + kb * 7 + c
                
                # U: for i in {ib*2, ib*2+1}, j in {0,1,2,3}
                for li in range(2):
                    for lj in range(2):
                        # A[ib*2+li, lj] and A[ib*2+li, 2+lj]
                        i = ib * 2 + li
                        j1 = lj
                        j2 = 2 + lj
                        U_full[i * 4 + j1, col] += U_s[li * 2 + lj, c]
                        U_full[i * 4 + j2, col] += U_s[li * 2 + lj, c]
                
                # V: B coefficients for this product
                for lj in range(2):
                    for lk in range(2):
                        j1 = lj
                        j2 = 2 + lj
                        k = kb * 2 + lk
                        V_full[j1 * 4 + k, col] += V_s[lj * 2 + lk, c]
                        V_full[j2 * 4 + k, col] += V_s[lj * 2 + lk, c]
                
                # W: C coefficients for this product
                for lk in range(2):
                    for li in range(2):
                        i = ib * 2 + li
                        k = kb * 2 + lk
                        W_full[k * 4 + i, col] += W_s[lk * 2 + li, c]
    
    return U_full, V_full, W_full


def _build_4x4x5_decomposition():
    """
    Build a rank-28+1=29 decomposition for 4x4x5.
    Use Strassen for the 4x4x4 core (rank 28) and rank-1 all-ones for k=4 (rank 1).
    """
    n, m, p = 4, 4, 5
    
    # Build 4x4x4 core
    U_core, V_core, W_core = _build_4x4x4_strassen()
    
    # Verify core
    T_core = np.zeros((4, 4, 4))
    for c in range(28):
        Uc = U_core[:, c].reshape(4, 4)
        Vc = V_core[:, c].reshape(4, 4)
        Wc = W_core[:, c].reshape(4, 4)
        T_core += np.einsum('ij,jk->ijk', Uc, Vc) * Wc.T[:, np.newaxis, :]
    
    if np.max(np.abs(T_core - 1.0)) > 1e-8:
        return None  # Core verification failed
    
    # Build k=4 boundary (rank-1 all-ones slice)
    U_bound = np.zeros((16, 1))
    V_bound = np.zeros((20, 1))
    W_bound = np.zeros((20, 1))
    
    # T[i,j,4] = 1 for all i,j
    # We need: sum_c U[i*4+j, c] * V[j*5+4, c] * W[4*4+i, c] = 1
    # With rank 1: U[i*4+j, 0] * V[j*5+4, 0] * W[16+i, 0] = 1
    # Set U[i*4+j, 0] = 1, V[j*5+4, 0] = 1, W[16+i, 0] = 1
    for i in range(4):
        for j in range(4):
            U_bound[i * 4 + j, 0] = 1
            V_bound[j * 5 + 4, 0] = 1
            W_bound[16 + i, 0] = 1
    
    # Combine
    total_rank = 28 + 1
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((20, total_rank))
    W_full = np.zeros((20, total_rank))
    
    U_full[:, :28] = U_core
    V_full[:, :28] = V_core
    W_full[:, :28] = W_core
    
    U_full[:, 28:] = U_bound
    V_full[:, 28:] = V_bound
    W_full[:, 28:] = W_bound
    
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist(),
        "rank": total_rank
    }


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Leave safety margin
    
    # Start with schoolbook
    schoolbook = _construct_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = schoolbook["rank"]
    
    # Try Strassen-based 4x4x4 core + rank-1 boundary
    if time.time() < deadline:
        result_strassen = _build_4x4x5_decomposition()
        if result_strassen is not None:
            if _verify_decomposition(result_strassen["U"], result_strassen["V"], result_strassen["W"], n, m, p, tol=1e-8):
                best_result = result_strassen
                best_rank = result_strassen["rank"]
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

# EVOLVE-BLOCK-END
