import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _target_tensor(n, m, p):
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                key = (i*m+j, j*p+k, k*n+i)
                T[key] = T.get(key, 0) + 1
    return T

def _from_columns(cols, n, m, p, rank):
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for r, (Ac, Bc, Cc) in enumerate(cols):
        for (i, j), v in Ac.items():
            U[i*m+j][r] += v
        for (j, k), v in Bc.items():
            V[j*p+k][r] += v
        for (i, k), v in Cc.items():
            W[k*n+i][r] += v
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    target = _target_tensor(n, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                s = 0
                for r in range(rank):
                    s += U[a][r]*V[b][r]*W[c][r]
                if abs(s - target.get((a, b, c), 0)) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _build_low_rank(n, m, p):
    cols = []
    
    # Strassen for 2x2 block multiplication
    # A is 2x2, B is 2x2, C is 2x2
    # Indices are local to the block
    def strassen(ai, aj, bj, bk, ci, ck):
        def A(i, j): return (ai+i, aj+j)
        def B(j, k): return (bj+j, bk+k)
        def C(i, k): return (ci+i, ck+k)
        # Standard Strassen 7 products
        M = [
            ([(A(0,0),1),(A(1,1),1)], [(B(0,0),1),(B(1,1),1)], [(C(0,0),1),(C(1,1),1)]),
            ([(A(1,0),1),(A(1,1),1)], [(B(0,0),1)], [(C(1,0),1),(C(1,1),-1)]),
            ([(A(0,0),1)], [(B(1,0),1),(B(1,1),-1)], [(C(0,1),1),(C(1,1),1)]),
            ([(A(1,1),1)], [(B(0,1),1),(B(1,0),-1)], [(C(0,0),1),(C(1,0),1)]),
            ([(A(0,0),1),(A(0,1),1)], [(B(1,0),1)], [(C(0,0),-1),(C(0,1),1)]),
            ([(A(0,1),1),(A(0,0),-1)], [(B(0,0),1),(B(1,0),1)], [(C(1,1),1)]),
            ([(A(1,0),1),(A(1,1),-1)], [(B(1,1),1),(B(0,1),1)], [(C(0,0),1),(C(0,1),-1)]),
        ]
        for Ac, Bc, Cc in M:
            cols.append((dict(Ac), dict(Bc), dict(Cc)))

    # A is 3x3, B is 3x4
    # Block 1: A[0:2, 0:2] (2x2), B[0:2, 0:2] (2x2) -> C[0:2, 0:2] (2x2)
    strassen(0, 0, 0, 0, 0, 0)
    
    # Block 2: A[0:2, 0:2] (2x2), B[0:2, 2:4] (2x2) -> C[0:2, 2:4] (2x2)
    strassen(0, 0, 0, 2, 0, 2)
    
    # Block 3: A[0:2, 2:3] (2x1), B[2:3, 0:2] (1x2) -> C[0:2, 0:2] (2x2)
    # This is an outer product. Rank 1? No, A is 2x1, B is 1x2.
    # C = A * B.
    # C_{ik} = A_{i0} B_{0k}.
    # We can do this with 1 rank term if we factor it?
    # U_{i, r} = A_{i0}, V_{0k, r} = B_{0k}, W_{ik, r} = 1?
    # Wait, the decomposition is sum_r U_a V_b W_c.
    # If we use 1 rank term:
    # U_{i*m+j, r} is non-zero for j=2 (col 2 of A).
    # V_{j*p+k, r} is non-zero for j=2 (row 2 of B).
    # W_{k*n+i, r} is non-zero for i in 0,1 and k in 0,1.
    # Let's check the term:
    # U_{i*3+2, r} = 1 for i=0,1? No, A_{i2}.
    # The code in inspiration 1 uses:
    # cols.append(( {(0,2):1,(1,2):1}, {(2,0):1,(2,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1} ))
    # This assumes A_{02}=1, A_{12}=1, B_{20}=1, B_{21}=1.
    # But A and B are variables. The coefficients must be 1 to be a valid decomposition for ALL A, B.
    # Wait, the decomposition is for the TENSOR, not specific matrices.
    # The tensor entries are 1.
    # So U, V, W are fixed numbers.
    # The term $A_{i2} B_{2k} C_{ik}$ corresponds to the monomial in the tensor.
    # In the tensor, the coefficient is 1.
    # So we need $U_{i3+2, r} V_{24+k, r} W_{k3+i, r} = 1$.
    # If we set $U_{0,3+2, r} = 1, U_{1,3+2, r} = 1$, etc., it works?
    # Let's check:
    # If $U_{a,r}=1$ for $a \in \{2, 5\}$ (i=0,1, j=2), $V_{b,r}=1$ for $b \in \{8, 9\}$ (j=2, k=0,1), $W_{c,r}=1$ for $c \in \{0, 1, 3, 4\}$ (k=0,1, i=0,1).
    # Then sum $U V W = 1*1*1 = 1$. Correct.
    # This adds 1 column.
    cols.append(( {(0,2):1,(1,2):1}, {(2,0):1,(2,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1} ))
    
    # Block 4: A[0:2, 2:3] (2x1), B[2:3, 2:4] (1x2) -> C[0:2, 2:4] (2x2)
    cols.append(( {(0,2):1,(1,2):1}, {(2,2):1,(2,3):1}, {(0,2):1,(0,3):1,(1,2):1,(1,3):1} ))
    
    # Block 5: A[2:3, 0:2] (1x2), B[0:2, 0:2] (2x2) -> C[2:3, 0:2] (1x2)
    # C_{2k} = A_{20}B_{0k} + A_{21}B_{1k}.
    # Two terms.
    for k in range(2):
        cols.append(( {(2,0):1,(2,1):1}, {(0,k):1,(1,k):1}, {(2,k):1} ))
        
    # Block 6: A[2:3, 0:2] (1x2), B[0:2, 2:4] (2x2) -> C[2:3, 2:4] (1x2)
    for k in range(2):
        cols.append(( {(2,0):1,(2,1):1}, {(0,2+k):1,(1,2+k):1}, {(2,2+k):1} ))
        
    # Block 7: A[2:3, 2:3] (1x1), B[2:3, 0:2] (1x2) -> C[2:3, 0:2] (1x2)
    cols.append(( {(2,2):1}, {(2,0):1,(2,1):1}, {(2,0):1,(2,1):1} ))
    
    # Block 8: A[2:3, 2:3] (1x1), B[2:3, 2:4] (1x2) -> C[2:3, 2:4] (1x2)
    cols.append(( {(2,2):1}, {(2,2):1,(2,3):1}, {(2,2):1,(2,3):1} ))
    
    rank = len(cols)
    return _from_columns(cols, n, m, p, rank)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the low-rank decomposition
    try:
        U, V, W = _build_low_rank(n, m, p)
        if _verify(U, V, W, n, m, p, len(U[0])):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback to schoolbook
    if time.time() < deadline:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Should not reach here
    return {"U": [[0]], "V": [[0]], "W": [[0]]}
