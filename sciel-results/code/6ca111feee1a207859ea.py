import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    alpha = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    beta = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    w_coeff = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    for s in range(1, 8):
        for t in range(4):
            c = 4 * (s - 1) + t
            k_a = 0 if t in [0, 2] else 1
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            k_c = 0 if t in [0, 1] else 1
            
            for (i, j), val in alpha[s].items():
                if val != 0:
                    u_idx = i * m + 2 * j + k_a
                    U[u_idx][c] += val
            
            for (i, j), val in beta[s].items():
                if val != 0:
                    v_row = 2 * i + r_b
                    v_col = 2 * j + c_b
                    v_idx = v_row * p + v_col
                    V[v_idx][c] += val
            
            for (i, j), val in w_coeff[s].items():
                if val != 0:
                    w_col = 2 * j + k_c
                    w_row = i
                    w_idx = w_col * n + w_row
                    W[w_idx][c] += val
    
    return U, V, W

def try_rank_reduction(U, V, W, n=2, m=4, p=4):
    """Try to reduce rank by greedily removing redundant terms."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = U_arr.shape[1]
    
    for pass_num in range(20):
        if time_up():
            break
        removed_any = False
        for r in range(current_rank):
            if time_up():
                break
            mask = [i for i in range(current_rank) if i != r]
            new_U = U_arr[:, mask]
            new_V = V_arr[:, mask]
            new_W = W_arr[:, mask]
            
            T_recon = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] = np.sum(new_U[i*m+j, :] * new_V[j*p+k, :] * new_W[k*n+i, :])
            
            if np.max(np.abs(T_recon - 1.0)) < 1e-9:
                U_arr = new_U
                V_arr = new_V
                W_arr = new_W
                current_rank = len(mask)
                removed_any = True
                break
        
        if not removed_any:
            break
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), current_rank

def try_linear_combination_removal(U, V, W, n=2, m=4, p=4, tol=1e-10):
    """Try to find linear combinations of terms that sum to zero."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = U_arr.shape[1]
    
    n_, m_, p_ = n, m, p
    num_eq = n_ * m_ * p_
    
    A = np.zeros((num_eq, current_rank))
    for i in range(n_):
        for j in range(m_):
            for k in range(p_):
                idx = (i * m_ + j) * p_ + k
                A[idx, :] = U_arr[i*m_+j, :] * V_arr[j*p_+k, :] * W_arr[k*n_+i, :]
    
    try:
        _, S, Vt = np.linalg.svd(A, full_matrices=True)
    except:
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), current_rank
    
    rank_A = np.sum(S > 1e-10)
    if rank_A >= current_rank:
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), current_rank
    
    null_basis = Vt[rank_A:, :]
    
    for c in null_basis:
        support = np.where(np.abs(c) > 1e-10)[0]
        if len(support) < 2:
            continue
        
        for r in support:
            if time_up():
                break
            mask = [i for i in range(current_rank) if i != r]
            new_U = U_arr[:, mask]
            new_V = V_arr[:, mask]
            new_W = W_arr[:, mask]
            
            T_recon = np.zeros((n_, m_, p_))
            for i in range(n_):
                for j in range(m_):
                    for k in range(p_):
                        T_recon[i,j,k] = np.dot(new_U[i*m_+j, :], new_V[j*p_+k, :] * new_W[k*n_+i, :])
            
            if np.max(np.abs(T_recon - 1.0)) < tol:
                U_arr = new_U
                V_arr = new_V
                W_arr = new_W
                current_rank = len(mask)
                break
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), current_rank

def construct():
    n, m, p = 2, 4, 4
    best_result = None
    best_rank = float('inf')
    
    # Start with Strassen
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if valid:
        best_rank = 28
        best_result = {"U": U, "V": V, "W": W}
    
    # Try to reduce rank with greedy removal
    if not time_up():
        U2, V2, W2, rank2 = try_rank_reduction(U, V, W, n, m, p)
        if rank2 < best_rank:
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = rank2
                best_result = {"U": U2, "V": V2, "W": W2}
    
    # Try linear combination removal
    if not time_up() and best_rank > 24:
        U3, V3, W3, rank3 = try_linear_combination_removal(U2, V2, W2, n, m, p)
        if rank3 < best_rank:
            valid, err = verify_decomposition(U3, V3, W3, n, m, p)
            if valid:
                best_rank = rank3
                best_result = {"U": U3, "V": V3, "W": W3}
    
    if best_result is None:
        # Fallback: naive decomposition
        rank = 32
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
