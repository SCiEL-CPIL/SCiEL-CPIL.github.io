"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
_start = time.time()

def _small_decomp(n, m, p):
    """Return (U, V, W) for small tensor M_{n,m,p} with minimal known rank."""
    n, m, p = int(n), int(m), int(p)
    if n == 1:
        # M_{1,m,p}: rank = m*p, trivial
        rank = m * p
        U = [[0]*rank for _ in range(1*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*1)]
        col = 0
        for j in range(m):
            for k in range(p):
                U[j][col] = 1
                V[j*p+k][col] = 1
                W[k][col] = 1
                col += 1
        return U, V, W, rank
    if m == 1:
        rank = n * p
        U = [[0]*rank for _ in range(n*1)]
        V = [[0]*rank for _ in range(1*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i][col] = 1
                V[k][col] = 1
                W[k*n+i][col] = 1
                col += 1
        return U, V, W, rank
    if p == 1:
        rank = n * m
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*1)]
        W = [[0]*rank for _ in range(1*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m+j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W, rank
    if n == 2 and m == 2 and p == 2:
        # Strassen
        # M1=(A11+A22)(B11+B22), M2=(A21+A22)B11, M3=A11(B12-B22), M4=A22(B21-B11)
        # M5=(A11+A12)B22, M6=(A21-A11)(B11+B12), M7=(A12-A22)(B21+B22)
        # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        # A indexing: A[0]=A11, A[1]=A12, A[2]=A21, A[3]=A22
        # B indexing: B[0]=B11, B[1]=B12, B[2]=B21, B[3]=B22
        # C indexing (trace): W[k*n+i] = C_{ik}, so W[0]=C11, W[1]=C21, W[2]=C12, W[3]=C22
        # M1
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[3][0]=1
        W[0][0]=1; W[3][0]=1
        # M2
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[1][1]=1
        # M3
        U[0][2]=1
        V[1][2]=1; V[3][2]=-1
        W[2][2]=1
        # M4
        U[3][3]=1
        V[2][3]=1; V[0][3]=-1
        W[0][3]=1; W[1][3]=1
        # M5
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        W[0][4]=-1; W[2][4]=1
        # M6
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[3][5]=1
        # M7
        U[1][6]=1; U[3][6]=-1
        V[2][6]=1; V[3][6]=1
        W[0][6]=1
        return U, V, W, rank
    # Default: schoolbook
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col]=1
                V[j*p+k][col]=1
                W[k*n+i][col]=1
                col+=1
    return U, V, W, rank

def _decompose(n, m, p):
    """Recursively decompose M_{n,m,p} using block splits."""
    if time.time() - _start > 140:
        # Fallback to schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col=0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col]=1; V[j*p+k][col]=1; W[k*n+i][col]=1; col+=1
        return U, V, W, rank
    # Try small cases first
    if n<=2 and m<=2 and p<=2:
        return _small_decomp(n, m, p)
    best = None
    best_rank = n*m*p
    # Try splitting along each dimension
    # Split m (columns of A / rows of B)
    for a in range(1, m):
        if time.time() - _start > 140:
            break
        U1,V1,W1,r1 = _decompose(n, a, p)
        U2,V2,W2,r2 = _decompose(n, m-a, p)
        r = r1 + r2
        if r < best_rank:
            # Combine
            U = U1 + [[0]*r for _ in range(n*(m-a))]
            for row in U2:
                U.append(row + [0]*r1)
            # Wait, need to be more careful with dimensions
            # U1 has n*a rows, r1 cols; U2 has n*(m-a) rows, r2 cols
            # Combined U has n*m rows, r cols
            U = [[0]*r for _ in range(n*m)]
            for i in range(n*a):
                for c in range(r1):
                    U[i][c] = U1[i][c]
            for i in range(n*(m-a)):
                for c in range(r2):
                    U[n*a+i][r1+c] = U2[i][c]
            V = [[0]*r for _ in range(m*p)]
            for j in range(a):
                for k in range(p):
                    for c in range(r1):
                        V[j*p+k][c] = V1[j*p+k][c]
            for j in range(m-a):
                for k in range(p):
                    for c in range(r2):
                        V[(a+j)*p+k][r1+c] = V2[j*p+k][c]
            W = [[0]*r for _ in range(p*n)]
            for k in range(p):
                for i in range(n):
                    for c in range(r1):
                        W[k*n+i][c] = W1[k*n+i][c]
                    for c in range(r2):
                        W[k*n+i][r1+c] = W2[k*n+i][c]
            best = (U, V, W)
            best_rank = r
    # Split p (columns of B / output cols)
    for b in range(1, p):
        if time.time() - _start > 140:
            break
        U1,V1,W1,r1 = _decompose(n, m, b)
        U2,V2,W2,r2 = _decompose(n, m, p-b)
        r = r1 + r2
        if r < best_rank:
            U = [[0]*r for _ in range(n*m)]
            for i in range(n*m):
                for c in range(r1):
                    U[i][c] = U1[i][c]
                for c in range(r2):
                    U[i][r1+c] = U2[i][c]
            V = [[0]*r for _ in range(m*p)]
            for j in range(m):
                for k in range(b):
                    for c in range(r1):
                        V[j*p+k][c] = V1[j*b+k][c]
                for k in range(p-b):
                    for c in range(r2):
                        V[j*p+(b+k)][r1+c] = V2[j*(p-b)+k][c]
            W = [[0]*r for _ in range(p*n)]
            for k in range(b):
                for i in range(n):
                    for c in range(r1):
                        W[k*n+i][c] = W1[k*n+i][c]
            for k in range(p-b):
                for i in range(n):
                    for c in range(r2):
                        W[(b+k)*n+i][r1+c] = W2[k*n+i][c]
            best = (U, V, W)
            best_rank = r
    # Split n (rows of A / output rows) - but n=2 usually, limited splits
    for a in range(1, n):
        if time.time() - _start > 140:
            break
        U1,V1,W1,r1 = _decompose(a, m, p)
        U2,V2,W2,r2 = _decompose(n-a, m, p)
        r = r1 + r2
        if r < best_rank:
            U = [[0]*r for _ in range(n*m)]
            for i in range(a*m):
                for c in range(r1):
                    U[i][c] = U1[i][c]
            for i in range((n-a)*m):
                for c in range(r2):
                    U[a*m+i][r1+c] = U2[i][c]
            V = [[0]*r for _ in range(m*p)]
            for j in range(m):
                for k in range(p):
                    for c in range(r1):
                        V[j*p+k][c] = V1[j*p+k][c]
                    for c in range(r2):
                        V[j*p+k][r1+c] = V2[j*p+k][c]
            W = [[0]*r for _ in range(p*n)]
            for k in range(p):
                for i in range(a):
                    for c in range(r1):
                        W[k*n+i][c] = W1[k*a+i][c]
                for i in range(n-a):
                    for c in range(r2):
                        W[k*n+(a+i)][r1+c] = W2[k*(n-a)+i][c]
            best = (U, V, W)
            best_rank = r
    if best is not None:
        return best[0], best[1], best[2], best_rank
    else:
        U, V, W, r = _small_decomp(n, m, p)
        return U, V, W, r

def construct():
    n, m, p = DIMENSIONS
    U, V, W, rank = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
