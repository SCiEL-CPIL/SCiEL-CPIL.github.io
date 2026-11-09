import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def _add_term(U, V, W, a_coeffs, b_coeffs, c_coeffs, col, n, m, p):
    for (i, j), val in a_coeffs.items():
        U[i*m + j][col] += val
    for (j, k), val in b_coeffs.items():
        V[j*p + k][col] += val
    for (k, i), val in c_coeffs.items():
        W[k*n + i][col] += val

def _verify(U, V, W, n, m, p, trials=20):
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for _ in range(trials):
        A = np.random.randint(-5, 6, size=(n, m)).astype(float)
        B = np.random.randint(-5, 6, size=(m, p)).astype(float)
        C = A @ B
        Aflat = A.reshape(n*m)
        Bflat = B.reshape(m*p)
        u = Ua @ Aflat
        v = Va @ Bflat
        Crec = (Wa * (u*v)).sum(axis=1)
        Crec = Crec.reshape(p, n).T
        if not np.allclose(Crec, C, atol=1e-6):
            return False
    return True

def _build_rank15(n, m, p):
    rank = 15
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    # Strassen 2x2x2 for C[0:2,0:2]
    # M1=(A00+A11)(B00+B11)
    _add_term(U, V, W, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, col, n, m, p); col+=1
    # M2=(A10+A11)B00
    _add_term(U, V, W, {(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):-1,(1,0):1}, col, n, m, p); col+=1
    # M3=A00(B01-B11)
    _add_term(U, V, W, {(0,0):1}, {(0,1):1,(1,1):-1}, {(0,0):-1,(0,1):1}, col, n, m, p); col+=1
    # M4=A11(B10-B00)
    _add_term(U, V, W, {(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}, col, n, m, p); col+=1
    # M5=(A00+A01)B11
    _add_term(U, V, W, {(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}, col, n, m, p); col+=1
    # M6=(A10-A00)(B00+B01)
    _add_term(U, V, W, {(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}, col, n, m, p); col+=1
    # M7=(A01-A11)(B10+B11)
    _add_term(U, V, W, {(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}, col, n, m, p); col+=1
    # Row 2, cols 0-2
    for k in range(3):
        _add_term(U, V, W, {(2,0):1,(2,1):1,(2,2):1}, {(0,k):1,(1,k):1,(2,k):1}, {(k,2):1}, col, n, m, p); col+=1
    # Cols 3-4, rows 0-1 (4 elements: C[0,3],C[0,4],C[1,3],C[1,4])
    # Combine C[0,3] and C[1,3] into one term using combined A and B
    # C[0,3]=A[0,0]B[0,3]+A[0,1]B[1,3]+A[0,2]B[2,3]
    # C[1,3]=A[1,0]B[0,3]+A[1,1]B[1,3]+A[1,2]B[2,3]
    # Can't combine directly. Use separate terms.
    for k in (3, 4):
        for i in (0, 1):
            _add_term(U, V, W, {(i,0):1,(i,1):1,(i,2):1}, {(0,k):1,(1,k):1,(2,k):1}, {(k,i):1}, col, n, m, p); col+=1
    # Row 2, cols 3-4
    for k in (3, 4):
        _add_term(U, V, W, {(2,0):1,(2,1):1,(2,2):1}, {(0,k):1,(1,k):1,(2,k):1}, {(k,2):1}, col, n, m, p); col+=1
    # That's 7+3+4+2=16, but we need 15
    # Merge: combine C[2,3] and C[2,4] with something? No.
    # Actually we have 7+3+4+2=16. Need to save 1.
    # Alternative: don't use Strassen, use a different approach
    raise ValueError(f"Built {col} terms, need {rank}")

def _build_rank16(n, m, p):
    rank = 16
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    # Strassen 2x2x2 for C[0:2,0:2]
    _add_term(U, V, W, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, col, n, m, p); col+=1
    _add_term(U, V, W, {(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):-1,(1,0):1}, col, n, m, p); col+=1
    _add_term(U, V, W, {(0,0):1}, {(0,1):1,(1,1):-1}, {(0,0):-1,(0,1):1}, col, n, m, p); col+=1
    _add_term(U, V, W, {(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}, col, n, m, p); col+=1
    _add_term(U, V, W, {(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}, col, n, m, p); col+=1
    _add_term(U, V, W, {(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}, col, n, m, p); col+=1
    _add_term(U, V, W, {(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}, col, n, m, p); col+=1
    # Row 2, cols 0-2
    for k in range(3):
        _add_term(U, V, W, {(2,0):1,(2,1):1,(2,2):1}, {(0,k):1,(1,k):1,(2,k):1}, {(k,2):1}, col, n, m, p); col+=1
    # Cols 3-4, all rows
    for k in (3, 4):
        for i in range(3):
            _add_term(U, V, W, {(i,0):1,(i,1):1,(i,2):1}, {(0,k):1,(1,k):1,(2,k):1}, {(k,i):1}, col, n, m, p); col+=1
    if col != rank:
        raise ValueError(f"Built {col} terms, need {rank}")
    return U, V, W

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    # Try rank-16 construction
    try:
        U, V, W = _build_rank16(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    # fallback schoolbook
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
