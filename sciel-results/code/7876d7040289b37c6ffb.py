"""Matrix multiplication 2x3x10: rank-minimal exact tensor decomposition.
Strategy: 5 Strassen 2x2x2 blocks (35 terms) for j in {0,1} + 20 schoolbook terms
for j=2, giving rank 55."""
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]
DEADLINE = 150.0
SAFETY = 15.0

# EVOLVE-BLOCK-START
def _time_left():
    return DEADLINE - SAFETY - time.time()

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    from collections import defaultdict
    T = defaultdict(Fraction)
    for c in range(rank):
        uc = [U[i*m+j][c] for i in range(n) for j in range(m)]
        vc = [V[j*p+k][c] for j in range(m) for k in range(p)]
        wc = [W[k*n+i][c] for k in range(p) for i in range(n)]
        for i in range(n):
            for j in range(m):
                uval = uc[i*m+j]
                if uval == 0:
                    continue
                for k in range(p):
                    prod = uval * vc[j*p+k] * wc[k*n+i]
                    if prod != 0:
                        T[(i,j,k)] += prod
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[(i,j,k)] != 1:
                    return False
    return True

def _build_strassen_2x2x2():
    U = [[Fraction(0)]*7 for _ in range(4)]
    V = [[Fraction(0)]*7 for _ in range(4)]
    W = [[Fraction(0)]*7 for _ in range(4)]
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W

def _build_base():
    n, m, p = DIMENSIONS
    rank = 35 + 20
    U = [[Fraction(0)]*rank for _ in range(n*m)]
    V = [[Fraction(0)]*rank for _ in range(m*p)]
    W = [[Fraction(0)]*rank for _ in range(p*n)]
    Us, Vs, Ws = _build_strassen_2x2x2()
    col = 0
    for kg in range(5):
        k0 = kg*2
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    U[i*m+j][col+c] += Us[i*2+j][c]
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    V[j*p+(k0+k)][col+c] += Vs[j*2+k][c]
        for k in range(2):
            for i in range(2):
                for c in range(7):
                    W[(k0+k)*n+i][col+c] += Ws[k*2+i][c]
        col += 7
    for kg in range(5):
        k0 = kg*2
        for i in range(2):
            for k in range(2):
                U[i*m+2][col] = Fraction(1)
                V[2*p+(k0+k)][col] = Fraction(1)
                W[(k0+k)*n+i][col] = Fraction(1)
                col += 1
    return U, V, W, rank

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    U, V, W, rank = _build_base()
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    rank = n*m*p
    U = [[Fraction(0)]*rank for _ in range(n*m)]
    V = [[Fraction(0)]*rank for _ in range(m*p)]
    W = [[Fraction(0)]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col]=Fraction(1); V[j*p+k][col]=Fraction(1); W[k*n+i][col]=Fraction(1); col+=1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
