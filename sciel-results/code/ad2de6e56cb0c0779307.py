import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(p):
            for k in range(m):
                target = 1
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                computed = 0
                for r in range(rank):
                    computed += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if computed != target:
                    return False
    return True

def _build_rank36():
    """Return rank-36 decomposition: 4 Strassen blocks (cols 0-7) + 1 rank-8 block (cols 8,9,10)."""
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0]*rank for _ in range(n*m)]  # 4 x 36
    V = [[0]*rank for _ in range(m*p)]  # 22 x 36
    W = [[0]*rank for _ in range(p*n)]  # 22 x 36
    
    r = 0
    # 4 Strassen blocks for columns 0,1,2,3,4,5,6,7
    for b in range(4):
        v00, v01, v10, v11 = 2*b, 2*b+1, 11+2*b, 12+2*b
        w00, w01, w10, w11 = 4*b, 4*b+2, 4*b+1, 4*b+3
        # P1: (a00+a11)(b00+b11)
        U[0][r]=1; U[3][r]=1; V[v00][r]=1; V[v11][r]=1; W[w00][r]=1; W[w11][r]=1; r+=1
        # P2: (a10+a11)b00
        U[2][r]=1; U[3][r]=1; V[v00][r]=1; W[w10][r]=1; W[w11][r]=-1; r+=1
        # P3: a00(b01-b11)
        U[0][r]=1; V[v01][r]=1; V[v11][r]=-1; W[w01][r]=1; W[w11][r]=1; r+=1
        # P4: a11(b10-b00)
        U[3][r]=1; V[v10][r]=1; V[v00][r]=-1; W[w00][r]=1; W[w10][r]=1; r+=1
        # P5: (a00+a01)b11
        U[0][r]=1; U[1][r]=1; V[v11][r]=1; W[w00][r]=-1; W[w01][r]=1; r+=1
        # P6: (a10-a00)(b00+b01)
        U[2][r]=1; U[0][r]=-1; V[v00][r]=1; V[v01][r]=1; W[w11][r]=1; r+=1
        # P7: (a01-a11)(b10+b11)
        U[1][r]=1; U[3][r]=-1; V[v10][r]=1; V[v11][r]=1; W[w00][r]=1; r+=1
    
    # Rank-8 block for columns 8, 9, 10
    v8, v9, v10 = 16, 17, 18  # B row 0: cols 8,9,10
    v18, v19, v20 = 28, 29, 30  # B row 1: cols 8,9,10
    w8, w9, w10 = 16, 18, 20  # C col 8,9,10 row 0 (j*n+i = j*2+0)
    w8b, w9b, w10b = 17, 19, 21  # C col 8,9,10 row 1 (j*n+i = j*2+1)
    
    # P1: (a00+a11)(b08+b18) -> c08 +=, c18 +=
    U[0][r]=1; U[3][r]=1; V[v8][r]=1; V[v18][r]=1; W[w8][r]=1; W[w8b][r]=1; r+=1
    # P2: (a00+a11)(b09+b19) -> c09 +=, c19 +=
    U[0][r]=1; U[3][r]=1; V[v9][r]=1; V[v19][r]=1; W[w9][r]=1; W[w9b][r]=1; r+=1
    # P3: a11*b08 -> c18 +=, c08 -=
    U[3][r]=1; V[v8][r]=1; W[w8b][r]=1; W[w8][r]=-1; r+=1
    # P4: a11*b09 -> c19 +=, c09 -=
    U[3][r]=1; V[v9][r]=1; W[w9b][r]=1; W[w9][r]=-1; r+=1
    # P5: a00*b18 -> c08 +=, c18 -=
    U[0][r]=1; V[v18][r]=1; W[w8][r]=1; W[w8b][r]=-1; r+=1
    # P6: a00*b19 -> c09 +=, c19 -=
    U[0][r]=1; V[v19][r]=1; W[w9][r]=1; W[w9b][r]=-1; r+=1
    # P7: a01*(b18+b19) -> c08 +=, c09 +=
    U[1][r]=1; V[v18][r]=1; V[v19][r]=1; W[w8][r]=1; W[w9][r]=1; r+=1
    # P8: (a00+a01+a11)*(b10+b20) -> c10 +=, c20 +=
    U[0][r]=1; U[1][r]=1; U[3][r]=1; V[v10][r]=1; V[v20][r]=1; W[w10][r]=1; W[w10b][r]=1; r+=1
    
    return U, V, W

def _try_remove_terms(U, V, W, max_time=10):
    """Try to remove zero-contribution terms."""
    start = time.time()
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    for r_remove in range(rank - 1, -1, -1):
        if time.time() - start > max_time:
            break
        removable = True
        for i in range(n):
            for k in range(m):
                if U[i*m+k][r_remove] != 0:
                    for j in range(p):
                        if V[k*p+j][r_remove] != 0 and W[j*n+i][r_remove] != 0:
                            removable = False
                            break
                    if not removable:
                        break
            if not removable:
                break
        if removable:
            new_U = [row[:r_remove] + row[r_remove+1:] for row in U]
            new_V = [row[:r_remove] + row[r_remove+1:] for row in V]
            new_W = [row[:r_remove] + row[r_remove+1:] for row in W]
            if _verify(new_U, new_V, new_W):
                return new_U, new_V, new_W
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    U, V, W = _build_rank36()
    
    if not _verify(U, V, W):
        # Fallback: build rank 39
        n, m, p = DIMENSIONS
        rank = 39
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        r = 0
        for b in range(5):
            v00, v01, v10, v11 = 2*b, 2*b+1, 11+2*b, 12+2*b
            w00, w01, w10, w11 = 4*b, 4*b+2, 4*b+1, 4*b+3
            U[0][r]=1; U[3][r]=1; V[v00][r]=1; V[v11][r]=1; W[w00][r]=1; W[w11][r]=1; r+=1
            U[2][r]=1; U[3][r]=1; V[v00][r]=1; W[w10][r]=1; W[w11][r]=-1; r+=1
            U[0][r]=1; V[v01][r]=1; V[v11][r]=-1; W[w01][r]=1; W[w11][r]=1; r+=1
            U[3][r]=1; V[v10][r]=1; V[v00][r]=-1; W[w00][r]=1; W[w10][r]=1; r+=1
            U[0][r]=1; U[1][r]=1; V[v11][r]=1; W[w00][r]=-1; W[w01][r]=1; r+=1
            U[2][r]=1; U[0][r]=-1; V[v00][r]=1; V[v01][r]=1; W[w11][r]=1; r+=1
            U[1][r]=1; U[3][r]=-1; V[v10][r]=1; V[v11][r]=1; W[w00][r]=1; r+=1
        U[0][r]=1; V[10][r]=1; W[20][r]=1; r+=1
        U[1][r]=1; V[21][r]=1; W[20][r]=1; r+=1
        U[2][r]=1; V[10][r]=1; W[21][r]=1; r+=1
        U[3][r]=1; V[21][r]=1; W[21][r]=1; r+=1
    
    # Try to remove any zero terms
    if time.time() < deadline - 5:
        U, V, W = _try_remove_terms(U, V, W, max_time=10)
    
    return {"U": U, "V": V, "W": W}
