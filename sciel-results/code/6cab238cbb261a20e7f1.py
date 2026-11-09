import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _strassen_2x2x2():
    """Strassen decomposition for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) for 2x2x2 multiplication.
    U rows: 0=a(i=0,j=0), 1=b(i=0,j=1), 2=c(i=1,j=0), 3=d(i=1,j=1)
    V rows: 0=e(j=0,k=0), 1=f(j=0,k=1), 2=g(j=1,k=0), 3=h(j=1,k=1)
    W rows: 0=c11(k=0,i=0), 1=c12(k=0,i=1), 2=c21(k=1,i=0), 3=c22(k=1,i=1)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (c+d)*e
    U[2,0] = 1; U[3,0] = 1
    V[0,0] = 1
    W[1,0] = 1  # c12
    
    # M2 = (c+d)*(f-h)
    U[2,1] = 1; U[3,1] = 1
    V[1,1] = 1; V[3,1] = -1
    W[0,1] = -1  # c11
    W[1,1] = 1   # c12
    
    # M3 = a*(e-f)
    U[0,2] = 1
    V[0,2] = 1; V[1,2] = -1
    W[2,2] = 1   # c21
    W[3,2] = -1  # c22
    
    # M4 = d*(g-h)
    U[3,3] = 1
    V[2,3] = 1; V[3,3] = -1
    W[0,3] = 1   # c11
    W[2,3] = 1   # c21
    
    # M5 = (a+b)*h
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = 1   # c11
    W[3,4] = 1   # c22
    
    # M6 = (c-a)*(e+f)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[0,5] = 1   # c11
    
    # M7 = (b-d)*(g+h)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[3,6] = -1  # c22
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0
                rhs = 0.0
                for l in range(rank):
                    u_val = U[i*m + j, l]
                    v_val = V[j*p + k, l]
                    w_val = W[k*n + i, l]
                    rhs += u_val * v_val * w_val
                if abs(lhs - rhs) > tol:
                    return False
    return True

def _block_decomp(n, m, p):
    """Block-structured decomposition using Strassen for 2x2x2 sub-blocks.
    Partition p=7 into [2,2,2,1].
    For each 2-column group, use two independent 2x2x2 Strassen blocks.
    For the 1-column group, use schoolbook (rank 8).
    Total rank: 3 * 14 + 8 = 50
    """
    U_list = []
    V_list = []
    W_list = []
    
    # Three groups of 2 output columns
    for k_group in range(3):
        k_offset = k_group * 2
        # Two independent 2x2x2 Strassen blocks for j=0,1 and j=2,3
        for j_group in range(2):
            j_offset = j_group * 2
            
            sU, sV, sW = _strassen_2x2x2()
            
            U_b = np.zeros((n*m, 7))
            V_b = np.zeros((m*p, 7))
            W_b = np.zeros((p*n, 7))
            
            # Map Strassen local indices to global indices
            # U local: 0=(i=0,j=0), 1=(i=0,j=1), 2=(i=1,j=0), 3=(i=1,j=1)
            # U global row: i*m + j
            u_map = [
                0*m + (j_offset + 0),    # i=0, j=j_offset
                0*m + (j_offset + 1),    # i=0, j=j_offset+1
                1*m + (j_offset + 0),    # i=1, j=j_offset
                1*m + (j_offset + 1)     # i=1, j=j_offset+1
            ]
            
            # V local: 0=(j=0,k=0), 1=(j=0,k=1), 2=(j=1,k=0), 3=(j=1,k=1)
            # V global row: j*p + k
            v_map = [
                (j_offset + 0)*p + (k_offset + 0),    # j=j_offset, k=k_offset
                (j_offset + 0)*p + (k_offset + 1),    # j=j_offset, k=k_offset+1
                (j_offset + 1)*p + (k_offset + 0),    # j=j_offset+1, k=k_offset
                (j_offset + 1)*p + (k_offset + 1)     # j=j_offset+1, k=k_offset+1
            ]
            
            # W local: 0=(k=0,i=0), 1=(k=0,i=1), 2=(k=1,i=0), 3=(k=1,i=1)
            # W global row: k*n + i
            w_map = [
                (k_offset + 0)*n + 0,    # k=k_offset, i=0
                (k_offset + 0)*n + 1,    # k=k_offset, i=1
                (k_offset + 1)*n + 0,    # k=k_offset+1, i=0
                (k_offset + 1)*n + 1     # k=k_offset+1, i=1
            ]
            
            for l in range(7):
                for local_idx, global_idx in enumerate(u_map):
                    U_b[global_idx, l] = sU[local_idx, l]
                for local_idx, global_idx in enumerate(v_map):
                    V_b[global_idx, l] = sV[local_idx, l]
                for local_idx, global_idx in enumerate(w_map):
                    W_b[global_idx, l] = sW[local_idx, l]
            
            U_list.append(U_b)
            V_list.append(V_b)
            W_list.append(W_b)
    
    # One group of 1 output column (k=6)
    k_offset = 6
    U_b = np.zeros((n*m, 8))
    V_b = np.zeros((m*p, 8))
    W_b = np.zeros((p*n, 8))
    col = 0
    for i in range(n):
        for j in range(m):
            U_b[i*m + j, col] = 1
            V_b[j*p + k_offset, col] = 1
            W_b[k_offset*n + i, col] = 1
            col += 1
    
    U_list.append(U_b)
    V_list.append(V_b)
    W_list.append(W_b)
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return U, V, W

def _schoolbook_decomp():
    """Standard (schoolbook) decomposition of 2x4x7 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
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

def construct():
    deadline = time.time() + 145.0  # Leave 5 seconds safety margin
    
    n, m, p = DIMENSIONS
    
    # Try block decomposition with Strassen (rank 50)
    try:
        U, V, W = _block_decomp(n, m, p)
        if _verify_decomp(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 56)
    return _schoolbook_decomp()
