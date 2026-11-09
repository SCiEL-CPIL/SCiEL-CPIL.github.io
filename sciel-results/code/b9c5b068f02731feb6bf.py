import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]
_DEADLINE = 120.0
_START = None

def _remaining():
    global _START
    if _START is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank34():
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal c
        # M1 = (a00+a11)(b00+b11)
        U[a00][c] = 1; U[a11][c] = 1
        V[b00][c] = 1; V[b11][c] = 1
        W[c00][c] = 1; W[c11][c] = 1
        c += 1
        # M2 = (a10+a11)b00
        U[a10][c] = 1; U[a11][c] = 1
        V[b00][c] = 1
        W[c10][c] = 1; W[c11][c] = -1
        c += 1
        # M3 = a00(b01-b11)
        U[a00][c] = 1
        V[b01][c] = 1; V[b11][c] = -1
        W[c01][c] = 1; W[c11][c] = 1
        c += 1
        # M4 = a11(b10-b00)
        U[a11][c] = 1
        V[b10][c] = 1; V[b00][c] = -1
        W[c00][c] = 1; W[c10][c] = 1
        c += 1
        # M5 = (a00+a01)b11
        U[a00][c] = 1; U[a01][c] = 1
        V[b11][c] = 1
        W[c00][c] = -1; W[c01][c] = 1
        c += 1
        # M6 = (a10-a00)(b00+b01)
        U[a10][c] = 1; U[a00][c] = -1
        V[b00][c] = 1; V[b01][c] = 1
        W[c11][c] = 1
        c += 1
        # M7 = (a01-a11)(b10+b11)
        U[a01][c] = 1; U[a11][c] = -1
        V[b10][c] = 1; V[b11][c] = 1
        W[c00][c] = 1
        c += 1

    def c_idx(i, k): return k * n + i
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k

    # Block 1: rows 0-1 of A, cols 0-1 of B -> rows 0-1 of C, cols 0-1
    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1), 
                 c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1))

    # Block 2: rows 0-1 of A, cols 2-3 of B -> rows 0-1 of C, cols 2-3
    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3), 
                 c_idx(0,2), c_idx(0,3), c_idx(1,2), c_idx(1,3))

    # Remaining terms for row 2 of A and col 2 of A
    terms = [
        (u_idx(0,2), v_idx(2,0), c_idx(0,0)),
        (u_idx(0,2), v_idx(2,1), c_idx(0,1)),
        (u_idx(1,2), v_idx(2,0), c_idx(1,0)),
        (u_idx(1,2), v_idx(2,1), c_idx(1,1)),
        (u_idx(0,2), v_idx(2,2), c_idx(0,2)),
        (u_idx(0,2), v_idx(2,3), c_idx(0,3)),
        (u_idx(1,2), v_idx(2,2), c_idx(1,2)),
        (u_idx(1,2), v_idx(2,3), c_idx(1,3)),
        (u_idx(2,0), v_idx(0,0), c_idx(2,0)),
        (u_idx(2,0), v_idx(0,1), c_idx(2,1)),
        (u_idx(2,1), v_idx(1,0), c_idx(2,0)),
        (u_idx(2,1), v_idx(1,1), c_idx(2,1)),
        (u_idx(2,0), v_idx(0,2), c_idx(2,2)),
        (u_idx(2,0), v_idx(0,3), c_idx(2,3)),
        (u_idx(2,1), v_idx(1,2), c_idx(2,2)),
        (u_idx(2,1), v_idx(1,3), c_idx(2,3)),
        (u_idx(2,2), v_idx(2,0), c_idx(2,0)),
        (u_idx(2,2), v_idx(2,1), c_idx(2,1)),
        (u_idx(2,2), v_idx(2,2), c_idx(2,2)),
        (u_idx(2,2), v_idx(2,3), c_idx(2,3)),
    ]
    
    for u_i, v_i, w_i in terms:
        U[u_i][c] = 1
        V[v_i][c] = 1
        W[w_i][c] = 1
        c += 1

    return {"U": U, "V": V, "W": W}

def _verify_mm(U, V, W):
    """Verify using random matrices."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Generate random integer matrices
    A = np.random.randint(-5, 5, (n, m))
    B = np.random.randint(-5, 5, (m, p))
    C = A @ B
    
    # Compute C from decomposition
    Uf = np.array(U, dtype=float)
    Vf = np.array(V, dtype=float)
    Wf = np.array(W, dtype=float)
    
    # C[i,k] = sum_j sum_r U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k] * W[k,i,r]
    # Wait, the decomposition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (if tensor is all ones)
    # For MM, the condition is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The algorithm computes:
    # C_alg[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_k V[j,k,r] B[j,k]) * W[k,i,r]?
    # No, the standard bilinear form is:
    # C[i,k] = sum_r W[k,i,r] * (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k])?
    # No, the factors are:
    # X_r = sum_j U[i,j,r] A[i,j] ? No, U depends on i,j.
    # The bilinear algorithm is:
    # 1. Compute M_r = (sum_j U[i,j,r] A[i,j]) * (sum_k V[j,k,r] B[j,k]) ?
    # No, the terms are:
    # M_r = (linear form of A) * (linear form of B)
    # C[i,k] = sum_r W[k,i,r] M_r
    
    # So:
    # M_r = (sum_{i,j} U[i,j,r] A[i,j]) * (sum_{j,k} V[j,k,r] B[j,k])
    # C[i,k] = sum_r W[k,i,r] M_r
    
    # Let's compute M_r
    # U is (n*m, rank). A is (n,m).
    # Flatten A to (n*m).
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    # M_r = (U[:,r] @ A_flat) * (V[:,r] @ B_flat)
    Uf = np.array(U).T # (rank, n*m)
    Vf = np.array(V).T # (rank, m*p)
    Wf = np.array(W).T # (rank, p*n)
    
    U_A = Uf @ A_flat # (rank,)
    V_B = Vf @ B_flat # (rank,)
    M = U_A * V_B # (rank,)
    
    # C[i,k] = sum_r W[k,i,r] M_r
    # W is (p*n, rank). Index k*n+i.
    # C is (n, p).
    C_alg = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            C_alg[i,k] = np.dot(Wf[:, k*n+i], M)
            
    if not np.allclose(C, C_alg, atol=1e-5):
        return False
    return True

def _try_reduce_rank(start_rank=33, max_rank=34):
    n, m, p = DIMENSIONS
    best_result = None
    
    for rank in range(start_rank, max_rank + 1):
        if _remaining() < 10:
            break
        for attempt in range(50):
            if _remaining() < 10:
                break
            # Initialize with random values
            U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)]
            
            if _verify_mm(U, V, W):
                # Try to round to integers or half-integers
                U_rounded = [[round(val * 2) / 2 for val in row] for row in U]
                V_rounded = [[round(val * 2) / 2 for val in row] for row in V]
                W_rounded = [[round(val * 2) / 2 for val in row] for row in W]
                if _verify_mm(U_rounded, V_rounded, W_rounded):
                    return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
    
    return None

def construct():
    global _START
    _START = time.time()
    try:
        # First try the known rank 34 solution
        result = _construct_rank34()
        if _verify_mm(result["U"], result["V"], result["W"]):
            # Try to find a lower rank
            lower = _try_reduce_rank(33, 34)
            if lower is not None:
                return lower
            return result
        else:
            return _schoolbook()
    except:
        return _schoolbook()
