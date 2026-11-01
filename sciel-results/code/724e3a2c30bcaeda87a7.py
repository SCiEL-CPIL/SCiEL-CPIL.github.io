"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_TIMEOUT = 120

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x2x2 matrix multiplication.
    Uses Strassen's algorithm: rank 7 (proven optimal for 2x2 matrix multiplication).
    W follows the official trace convention W[k*n+i, r].
    """
    if time.time() - _START_TIME > _TIMEOUT:
        raise TimeoutError("Time limit exceeded")

    n, m, p = DIMENSIONS
    rank = 7

    # Strassen's 7 products for A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e),
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C[0,0]=M1+M4-M5+M7, C[0,1]=M3+M5, C[1,0]=M2+M4, C[1,1]=M1-M2+M3+M6
    #
    # Indexing: U[i*m+j,r], V[j*p+k,r], W[k*n+i,r]
    # a=U[0], b=U[1], c=U[2], d=U[3]
    # e=V[0], f=V[1], g=V[2], h=V[3]
    # C00=W[0], C10=W[1], C01=W[2], C11=W[3]

    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    # U: row=i*m+j, col=r
    # r0: M1=(a+d)(e+h): a=U[0],d=U[3]
    U[0][0] = 1; U[3][0] = 1
    # r1: M2=(c+d)e: c=U[2],d=U[3]
    U[2][1] = 1; U[3][1] = 1
    # r2: M3=a(f-h): a=U[0]
    U[0][2] = 1
    # r3: M4=d(g-e): d=U[3]
    U[3][3] = 1
    # r4: M5=(a+b)h: a=U[0],b=U[1]
    U[0][4] = 1; U[1][4] = 1
    # r5: M6=(c-a)(e+f): c=U[2],a=U[0]
    U[2][5] = 1; U[0][5] = -1
    # r6: M7=(b-d)(g+h): b=U[1],d=U[3]
    U[1][6] = 1; U[3][6] = -1

    # V: row=j*p+k, col=r
    # r0: e=V[0],h=V[3]
    V[0][0] = 1; V[3][0] = 1
    # r1: e=V[0]
    V[0][1] = 1
    # r2: f=V[1],h=V[3]
    V[1][2] = 1; V[3][2] = -1
    # r3: g=V[2],e=V[0]
    V[2][3] = 1; V[0][3] = -1
    # r4: h=V[3]
    V[3][4] = 1
    # r5: e=V[0],f=V[1]
    V[0][5] = 1; V[1][5] = 1
    # r6: g=V[2],h=V[3]
    V[2][6] = 1; V[3][6] = 1

    # W: row=k*n+i, col=r
    # W[0]=C[0,0]: M1+M4-M5+M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # W[1]=C[1,0]: M2+M4
    W[1][1] = 1; W[1][3] = 1
    # W[2]=C[0,1]: M3+M5
    W[2][2] = 1; W[2][4] = 1
    # W[3]=C[1,1]: M1-M2+M3+M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1

    # Verify: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if s != 1:
                    raise ValueError(f"Decomposition error at ({i},{j},{k}): got {s}")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
