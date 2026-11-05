"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Build rank-54 decomposition using Strassen for 2x2 blocks and naive for 2x1 blocks
    rank_base = 54
    U = np.zeros((n * m, rank_base), dtype=np.float64)
    V = np.zeros((m * p, rank_base), dtype=np.float64)
    W = np.zeros((p * n, rank_base), dtype=np.float64)
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # M0: (A00+A11)(B00+B11)
        U[i_base*m+j_base, col] += 1
        U[(i_base+1)*m+(j_base+1), col] += 1
        V[j_base*p+k_base, col] += 1
        V[(j_base+1)*p+(k_base+1), col] += 1
        W[k_base*n+i_base, col] += 1
        W[(k_base+1)*n+(i_base+1), col] += 1
        col += 1
        # M1: (A10+A11)B00
        U[(i_base+1)*m+j_base, col] += 1
        U[(i_base+1)*m+(j_base+1), col] += 1
        V[j_base*p+k_base, col] += 1
        W[k_base*n+(i_base+1), col] += 1
        W[(k_base+1)*n+(i_base+1), col] -= 1
        col += 1
        # M2: A00(B01-B11)
        U[i_base*m+j_base, col] += 1
        V[j_base*p+(k_base+1), col] += 1
        V[(j_base+1)*p+(k_base+1), col] -= 1
        W[(k_base+1)*n+i_base, col] += 1
        W[(k_base+1)*n+(i_base+1), col] += 1
        col += 1
        # M3: A11(B10-B00)
        U[(i_base+1)*m+(j_base+1), col] += 1
        V[(j_base+1)*p+k_base, col] += 1
        V[j_base*p+k_base, col] -= 1
        W[k_base*n+i_base, col] += 1
        W[k_base*n+(i_base+1), col] += 1
        col += 1
        # M4: (A00+A01)B11
        U[i_base*m+j_base, col] += 1
        U[i_base*m+(j_base+1), col] += 1
        V[(j_base+1)*p+(k_base+1), col] += 1
        W[k_base*n+i_base, col] -= 1
        W[(k_base+1)*n+i_base, col] += 1
        col += 1
        # M5: (A10-A00)(B00+B01)
        U[(i_base+1)*m+j_base, col] += 1
        U[i_base*m+j_base, col] -= 1
        V[j_base*p+k_base, col] += 1
        V[j_base*p+(k_base+1), col] += 1
        W[(k_base+1)*n+(i_base+1), col] += 1
        col += 1
        # M6: (A01-A11)(B10+B11)
        U[i_base*m+(j_base+1), col] += 1
        U[(i_base+1)*m+(j_base+1), col] -= 1
        V[(j_base+1)*p+k_base, col] += 1
        V[(j_base+1)*p+(k_base+1), col] += 1
        W[k_base*n+i_base, col] += 1
        col += 1

    def add_naive_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base*m+j_base, col] += 1
        V[j_base*p+k_base, col] += 1
        W[k_base*n+i_base, col] += 1
        col += 1
        U[i_base*m+j_base, col] += 1
        V[j_base*p+(k_base+1), col] += 1
        W[(k_base+1)*n+i_base, col] += 1
        col += 1
        U[(i_base+1)*m+j_base, col] += 1
        V[j_base*p+k_base, col] += 1
        W[k_base*n+(i_base+1), col] += 1
        col += 1
        U[(i_base+1)*m+j_base, col] += 1
        V[j_base*p+(k_base+1), col] += 1
        W[(k_base+1)*n+(i_base+1), col] += 1
        col += 1

    # Split m=5 into 2+3, p=6 into 2+2+2
    # 6 Strassen blocks for the 2x2 portion (rows 0-1 of A, cols 0-1 of B) with k=0,2,4
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    # 3 naive blocks for the 2x1 portion (rows 0-1 of A, col 4 of B) with k=0,2,4
    add_naive_block(0, 4, 0)
    add_naive_block(0, 4, 2)
    add_naive_block(0, 4, 4)
    
    if col != rank_base:
        # Fallback to naive rank-60
        rank = n*m*p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify decomposition
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    T_dec = np.einsum('ar,br,cr->abc', U, V, W)
    
    if not np.allclose(T_dec, T, atol=1e-10):
        # Fallback to naive rank-60
        rank = n*m*p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by checking for linearly dependent columns
    if time.time() < deadline - 10.0:
        # Check if any column is zero
        for r in range(rank_base):
            if np.allclose(U[:, r], 0) or np.allclose(V[:, r], 0) or np.allclose(W[:, r], 0):
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
                rank_base -= 1
                break
        
        # Try removing each column and see if the remaining still spans
        for r in range(rank_base):
            if time.time() > deadline - 5.0:
                break
            U_test = np.delete(U, r, axis=1)
            V_test = np.delete(V, r, axis=1)
            W_test = np.delete(W, r, axis=1)
            T_test = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
            if np.allclose(T_test, T, atol=1e-10):
                U = U_test
                V = V_test
                W = W_test
                rank_base -= 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
