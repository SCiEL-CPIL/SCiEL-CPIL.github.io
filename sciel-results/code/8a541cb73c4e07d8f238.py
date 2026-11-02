import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The multiplication tensor T[i,j,k] = delta[i*k + j, m*k + l] for C[i,l] = sum_j A[i,j]*B[j,l]
    # In our indexing:
    # A has n*m elements indexed by i*m + j
    # B has m*p elements indexed by j*p + l  
    # C has p*n elements indexed by l*n + i
    # T[(i*m+j), (j*p+l), (l*n+i)] = 1 if valid, 0 otherwise
    
    # Compute the tensor from the decomposition
    # T_decomposed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build the true tensor
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a_idx = i * m + j
                b_idx = j * p + l
                c_idx = l * n + i
                T_true[a_idx, b_idx, c_idx] = 1.0
    
    return np.allclose(T_decomp, T_true, atol=1e-10)

def _try_rank(rank, n=2, m=3, p=8, timeout=10):
    """Try to find a decomposition of given rank using random initialization + gradient descent."""
    start = time.time()
    np.random.seed(rank * 1000 + n * m * p)
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                T[i*m+j, j*p+l, l*n+i] = 1.0
    
    lr = 0.01
    for step in range(2000):
        if time.time() - start > timeout:
            break
        # Compute current tensor
        T_cur = np.einsum('ar,br,cr->abc', U, V, W)
        # Gradient
        dT = T_cur - T
        dU = np.einsum('abc,br,cr->ar', dT, V, W)
        dV = np.einsum('abc,ar,cr->br', dT, U, W)
        dW = np.einsum('abc,ar,br->cr', dT, U, V)
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Check
    T_final = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_final - T))
    if err < 1e-6:
        # Try to snap to small rationals
        return U, V, W, err
    return None, None, None, err

def construct():
    n, m, p = 2, 3, 8
    deadline = time.time() + 120
    
    # First, verify the known rank-44 solution
    U44 = [[0] * 44 for _ in range(n * m)]
    V44 = [[0] * 44 for _ in range(m * p)]
    W44 = [[0] * 44 for _ in range(p * n)]
    
    col = 0
    # Part 1: schoolbook for column 0
    for i in range(2):
        for k in range(8):
            U44[i * 3 + 0][col] = 1
            V44[0 * 8 + k][col] = 1
            W44[k * 2 + i][col] = 1
            col += 1
    
    # Part 2: Strassen for 4 blocks of 2x2
    for block in range(4):
        # m1 = (a00+a11)(b00+b11)
        U44[1][col] = 1; U44[5][col] = 1
        V44[1*8 + block*2][col] = 1; V44[2*8 + block*2+1][col] = 1
        W44[(block*2)*2 + 0][col] = 1; W44[(block*2+1)*2 + 1][col] = 1
        col += 1
        # m2 = (a10+a11)b00
        U44[4][col] = 1; U44[5][col] = 1
        V44[1*8 + block*2][col] = 1
        W44[(block*2)*2 + 1][col] = 1; W44[(block*2+1)*2 + 1][col] = -1
        col += 1
        # m3 = a00(b01-b11)
        U44[1][col] = 1
        V44[1*8 + block*2+1][col] = 1; V44[2*8 + block*2+1][col] = -1
        W44[(block*2+1)*2 + 0][col] = 1; W44[(block*2+1)*2 + 1][col] = 1
        col += 1
        # m4 = a11(b10-b00)
        U44[5][col] = 1
        V44[2*8 + block*2][col] = 1; V44[1*8 + block*2][col] = -1
        W44[(block*2)*2 + 0][col] = 1; W44[(block*2)*2 + 1][col] = 1
        col += 1
        # m5 = (a00+a01)b11
        U44[1][col] = 1; U44[2][col] = 1
        V44[2*8 + block*2+1][col] = 1
        W44[(block*2)*2 + 0][col] = -1; W44[(block*2+1)*2 + 0][col] = 1
        col += 1
        # m6 = (a10-a00)(b00+b01)
        U44[4][col] = 1; U44[1][col] = -1
        V44[1*8 + block*2][col] = 1; V44[1*8 + block*2+1][col] = 1
        W44[(block*2+1)*2 + 1][col] = 1
        col += 1
        # m7 = (a01-a11)(b10+b11)
        U44[2][col] = 1; U44[5][col] = -1
        V44[2*8 + block*2][col] = 1; V44[2*8 + block*2+1][col] = 1
        W44[(block*2)*2 + 0][col] = 1
        col += 1
    
    # Verify
    valid44 = _verify_decomposition(U44, V44, W44, n, m, p)
    
    if not valid44:
        # The rank-44 solution is incorrect; fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find lower rank solutions
    best_rank = 44
    best_result = (U44, V44, W44)
    
    # Try ranks from 43 down, with limited time per attempt
    for r in range(43, 39, -1):
        if time.time() > deadline - 20:
            break
        U_r, V_r, W_r, err = _try_rank(r, n, m, p, timeout=15)
        if U_r is not None and err < 1e-6:
            # Verify
            if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                best_rank = r
                best_result = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                break
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
