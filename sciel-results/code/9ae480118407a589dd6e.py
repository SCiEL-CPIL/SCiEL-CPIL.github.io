import numpy as np
import time

def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 145.0

    def verify(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # T[u,v,w] = sum_t U[u,t]*V[v,t]*W[w,t]
        # Efficient check: compute T_calc and compare to known T
        T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
        # Check against expected
        T_exp = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_exp[i*m+j, j*p+k, k*n+i] = 1.0
        return np.allclose(T_calc, T_exp, atol=1e-10)

    def to_lists(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build a decomposition via Kronecker/block structure.
    # We can write C = A*B where A is 4x5, B is 5x5.
    # Split A into 2 blocks of 2 rows each: A1 (2x5), A2 (2x5)
    # Split B into blocks... this gets complicated.

    # Alternative: use known small mat-mul decompositions combined.
    # For 2x2 mat-mul, Strassen gives rank 7.
    # For 1x1 it's rank 1.
    # We can pad to powers of 2.

    # Actually, the simplest guaranteed approach: schoolbook with rank 100.
    # But we can do better by using the structure.

    # Let's try: decompose 4x5x5 as a combination of smaller blocks.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # We can view this as:
    # - Use a 4x5x5 tensor directly.
    # 
    # Let me try a structured approach: 
    # Write the multiplication as using rank-1 "gates" but with shared structure.

    # Simplest reliable: schoolbook rank = 4*5*5 = 100
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1

    if verify(U, V, W):
        return to_lists(U, V, W)

    # If schoolbook verification fails (shouldn't), return as-is.
    return to_lists(U, V, W)
