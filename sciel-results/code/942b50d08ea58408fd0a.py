import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    # Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    # U shape: 4x7, V shape: 4x7, W shape: 4x7
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Standard 7 Strassen products
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B11-B21)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    defs = [
        # P1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,-1),(1,0,-1),(1,1,1)]),
        # P2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # P3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # P4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,-1)]),
        # P5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # P6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # P7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for t, (ue, ve, we) in enumerate(defs):
        for (i, j, c) in ue: U[i*2+j][t] += c
        for (j, k, c) in ve: V[j*2+k][t] += c
        for (k, i, c) in we: W[k*2+i][t] += c
        
    return U, V, W

def _build_4x4x4():
    # 2-level Strassen for 4x4x4. Rank 49.
    # U: 16x49, V: 16x49, W: 16x49
    # Indices: U[i*4+j], V[j*4+k], W[k*4+i]
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2()
    
    # 7 outer Strassen steps, each using 7 inner Strassen steps
    outer_defs = [
        # P1: (A11+A22)(B11+B22) -> C11 +=, C12 -=, C21 -=, C22 +=
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,-1),(1,0,-1),(1,1,1)]),
        # P2: (A21+A22)B11 -> C21 +=, C22 -=
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # P3: A11(B21-B22) -> C12 +=, C22 +=
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # P4: A22(B11-B21) -> C21 +=, C11 -=
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,-1)]),
        # P5: (A11+A12)B22 -> C11 -=, C12 +=
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # P6: (A21-A11)(B11+B12) -> C22 +=
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # P7: (A12-A22)(B21+B22) -> C11 +=
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for s, (ue, ve, we) in enumerate(outer_defs):
        for t in range(7):
            col = s*7 + t
            # Inner Strassen indices t
            for (bi, bj, c) in ue:
                for i in range(2):
                    for j in range(2):
                        gi = bi*2 + i
                        gj = bj*2 + j
                        a_loc = i*2 + j
                        cc = U2[a_loc][t]
                        if cc:
                            U[gi*4 + gj][col] += cc * c
            for (bj, bk, c) in ve:
                for j in range(2):
                    for k in range(2):
                        gj = bj*2 + j
                        gk = bk*2 + k
                        b_loc = j*2 + k
                        cc = V2[b_loc][t]
                        if cc:
                            V[gj*4 + gk][col] += cc * c
            for (bk, bi, c) in we:
                for k in range(2):
                    for i in range(2):
                        gk = bk*2 + k
                        gi = bi*2 + i
                        c_loc = k*2 + i
                        cc = W2[c_loc][t]
                        if cc:
                            W[gk*4 + gi][col] += cc * c
                            
    return U, V, W

def _build_4x1x6():
    # Outer product for (4x1) * (1x6) -> 4x6. Rank 24.
    # U: 4x24, V: 6x24, W: 24x4 (W[k*4+i])
    U = [[0]*24 for _ in range(4)]
    V = [[0]*24 for _ in range(6)]
    W = [[0]*24 for _ in range(24)]
    
    c = 0
    for i in range(4):
        for k in range(6):
            U[i][c] = 1
            V[k][c] = 1
            W[k*4 + i][c] = 1
            c += 1
            
    return U, V, W

def _combine_4x5x6():
    # Combine 4x4x4 (rank 49) and 4x1x6 (rank 24)
    # Total rank = 73
    # U: 20x73 (4x5)
    # V: 30x73 (5x6)
    # W: 24x73 (6x4)
    
    n, m, p = 4, 5, 6
    rank = 73
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Part 1: 4x4x4 (A[0:4, 0:4], B[0:4, 0:4], C[0:4, 0:4])
    U1, V1, W1 = _build_4x4x4()
    for i in range(4):
        for j in range(4):
            u_idx = i*4 + j
            for t in range(49):
                U[i*m + j][t] = U1[u_idx][t]
    for j in range(4):
        for k in range(4):
            v_idx = j*4 + k
            for t in range(49):
                V[j*p + k][t] = V1[v_idx][t]
    for k in range(4):
        for i in range(4):
            w_idx = k*4 + i
            for t in range(49):
                W[k*n + i][t] = W1[w_idx][t]
                
    # Part 2: 4x1x6 (A[0:4, 4], B[4, 0:6], C[0:4, 0:6])
    U2, V2, W2 = _build_4x1x6()
    offset = 49
    for i in range(4):
        for t in range(24):
            U[i*m + 4][offset + t] = U2[i][t]
    for k in range(6):
        for t in range(24):
            V[4*p + k][offset + t] = V2[k][t]
    for k in range(6):
        for i in range(4):
            w_idx = k*4 + i
            for t in range(24):
                W[k*n + i][offset + t] = W2[w_idx][t]
                
    return U, V, W

def _verify(U, V, W, n, m, p):
    # Verify with a few random integer matrices
    rng = random.Random(42)
    rank = len(U[0])
    for _ in range(5):
        A = [[rng.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j]*B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        for i in range(n):
            for k in range(p):
                val = 0
                for t in range(rank):
                    au = sum(U[i*m + j][t] * A[i][j] for j in range(m))
                    bu = sum(V[j*p + k][t] * B[j][k] for j in range(m))
                    val += W[k*n + i][t] * au * bu
                if val != C[i][k]:
                    return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the rank 73 decomposition
    try:
        U, V, W = _combine_4x5x6()
        if _verify(U, V, W, n, m, p):
            if time.time() - start_time < 140: # Safety margin
                return {"U": U, "V": V, "W": W}
            else:
                # If we are too close to timeout, still return if valid
                return {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook if verification fails (should not happen)
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
            return {"U": U, "V": V, "W": W}
    except Exception:
        # Fallback to schoolbook on error
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
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
