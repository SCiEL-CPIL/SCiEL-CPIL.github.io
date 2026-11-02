import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that the decomposition exactly matches the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
        
    # Convert to numpy for fast einsum verification
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute decomposition tensor
    # T_dec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T_dec = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    
    return np.allclose(T_dec, T, atol=tol)

def _build_rank_39():
    """
    Build the known rank 39 decomposition using pure Python lists for exactness and speed.
    5 Strassen blocks (2x2x2) + 1 Schoolbook block (2x2x1).
    """
    n, m, p = DIMENSIONS
    rank = 39
    # U shape: (n*m, rank) = (4, 39)
    # V shape: (m*p, rank) = (22, 39)
    # W shape: (p*n, rank) = (22, 39)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Helper to add a Strassen block for columns k0, k1
    def add_strassen_block(k0, k1):
        nonlocal r
        # B indices: B[j, k] -> V[j*p + k]
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = p + k0   # B[1, k0]
        v_h = p + k1   # B[1, k1]

        # W indices: C[i, k] -> W[k*n + i]
        w_c00 = k0 * n     # C[0, k0]
        w_c10 = k0 * n + 1 # C[1, k0]
        w_c01 = k1 * n     # C[0, k1]
        w_c11 = k1 * n + 1 # C[1, k1]

        # Strassen terms
        # m1 = (a00+a11)(b00+b11)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1

        # m2 = (a10+a11)b00
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1

        # m3 = a00(b01-b11)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1

        # m4 = a11(b10-b00)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1

        # m5 = (a00+a01)b11
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1

        # m6 = (a10-a00)(b00+b01)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1

        # m7 = (a01-a11)(b10+b11)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1

    # 5 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7), (8,9)
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        add_strassen_block(k0, k1)

    # Last column k=10: Schoolbook
    k2 = 10
    v_e2 = k2      # B[0, 10]
    v_g2 = p + k2  # B[1, 10]
    w_c02 = k2 * n      # C[0, 10]
    w_c12 = k2 * n + 1  # C[1, 10]

    # Q1: a00 * B[0, 10]
    U[0][r] += 1
    V[v_e2][r] += 1
    W[w_c02][r] += 1
    r += 1

    # Q2: a01 * B[1, 10]
    U[1][r] += 1
    V[v_g2][r] += 1
    W[w_c02][r] += 1
    r += 1

    # Q3: a10 * B[0, 10]
    U[2][r] += 1
    V[v_e2][r] += 1
    W[w_c12][r] += 1
    r += 1

    # Q4: a11 * B[1, 10]
    U[3][r] += 1
    V[v_g2][r] += 1
    W[w_c12][r] += 1
    r += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # 1. Build the known rank 39 solution
    U, V, W = _build_rank_39()
    
    # 2. Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook rank 44
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n*m)]
        V_fb = [[0] * rank for _ in range(m*p)]
        W_fb = [[0] * rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # 3. Return the rank 39 solution
    # We do not attempt rank 38 search here because:
    # a) It is computationally expensive and unlikely to succeed.
    # b) The rank 39 solution is verified and robust.
    # c) The directive suggests keeping the design family, and the primary value is in the correct rank 39 construction.
    
    return {"U": U, "V": V, "W": W}
