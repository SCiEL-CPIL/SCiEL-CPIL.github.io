import time

DIMENSIONS = [3, 3, 6]

def _true_verify(U, V, W, n, m, p, rank, tol=1e-9):
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[jp*p+k][c] * W[k*n+i][c]
                    target = 1.0 if j == jp else 0.0
                    if abs(s - target) > tol:
                        return False
    return True

def _schoolbook(n, m, p):
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

def _block_3x3x2():
    n, m, p = 3, 3, 2
    rank = 10
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen 7 terms for i,j in {0,1}, k in {0,1}
    # Term 1
    U[0][0]=1; U[4][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[4][0]=1
    # Term 2
    U[3][1]=1; U[4][1]=1; V[0][1]=1; W[0][1]=-1; W[3][1]=1
    # Term 3
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[4][2]=-1
    # Term 4
    U[4][3]=1; V[3][3]=1; V[0][3]=-1; W[0][3]=1; W[4][3]=1
    # Term 5
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[1][4]=1; W[4][4]=1
    # Term 6
    U[3][5]=1; U[4][5]=-1; V[1][5]=1; V[3][5]=1; W[3][5]=1
    # Term 7
    U[1][6]=1; U[3][6]=-1; V[0][6]=1; V[3][6]=1; W[0][6]=-1; W[4][6]=1
    
    # Term 8: j=2, j'=2 for all i
    for i in range(n):
        U[i*m+2][7] = 1
    for k in range(p):
        V[2*p+k][7] = 1
    for k in range(p):
        for i in range(n):
            W[k*n+i][7] = 1
            
    # Term 9: i=2, j=0, j'=0
    U[2*m+0][8] = 1
    for k in range(p):
        V[0*p+k][8] = 1
    for k in range(p):
        W[k*n+2][8] = 1
        
    # Term 10: i=2, j=1, j'=1
    U[2*m+1][9] = 1
    for k in range(p):
        V[1*p+k][9] = 1
    for k in range(p):
        W[k*n+2][9] = 1
        
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110.0
    try:
        blk = _block_3x3x2()
        per = 10
        U = [[0]*(3*per) for _ in range(n*m)]
        V = [[0]*(3*per) for _ in range(m*p)]
        W = [[0]*(3*per) for _ in range(p*n)]
        
        # Assemble 3 blocks
        for blk_i in range(3):
            base = blk_i*per
            # U is same for all blocks?
            # U[i*m+j, c] depends on i,j.
            # The block decomposition assumes A is split into 3 blocks of 3x3?
            # No, A is 3x3. B is 3x6.
            # We split B into 3 blocks of 3x2.
            # So U is the same for all 3 blocks?
            # U[i*m+j, c] is part of A.
            # If we use the same U for all blocks, it works.
            for i in range(n*m):
                for c in range(per):
                    U[i][base+c] = blk["U"][i][c]
            
            # V is split by k.
            # Block 0: k in {0,1}
            # Block 1: k in {2,3}
            # Block 2: k in {4,5}
            for j in range(m):
                for t in range(2):
                    k = 2*blk_i + t
                    for c in range(per):
                        V[j*p+k][base+c] = blk["V"][j*2+t][c]
                        
            # W is split by k.
            for t in range(2):
                k = 2*blk_i + t
                for i in range(n):
                    for c in range(per):
                        W[k*n+i][base+c] = blk["W"][t*3+i][c]
                        
        if _true_verify(U, V, W, n, m, p, 3*per):
            if time.time() < deadline:
                return {"U": U, "V": V, "W": W}
        return _schoolbook(n, m, p)
    except Exception:
        return _schoolbook(n, m, p)
