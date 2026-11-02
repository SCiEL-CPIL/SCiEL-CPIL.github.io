"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import random

DIMENSIONS = [2, 3, 7]

def _strassen_222():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7) for A[0,0],A[0,1],A[1,0],A[1,1]
    V: (4, 7) for B[0,0],B[0,1],B[1,0],B[1,1]
    W: (4, 7) for C[0,0],C[0,1],C[1,0],C[1,1] with W[k*2+i, r]
    """
    terms = [
        ([1,0,0,1], [1,0,1,0], [1,0,0,-1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    for r, (u, v, w) in enumerate(terms):
        for idx in range(4):
            U[idx][r] = u[idx]
            V[idx][r] = v[idx]
            W[idx][r] = w[idx]
    return U, V, W

def _schoolbook_221():
    """2x2x1 schoolbook, rank 4."""
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    U[0][0]=1; V[0][0]=1; W[0][0]=1
    U[1][1]=1; V[1][1]=1; W[0][1]=1
    U[2][2]=1; V[0][2]=1; W[1][2]=1
    U[3][3]=1; V[1][3]=1; W[1][3]=1
    return U, V, W

def _schoolbook_217():
    """2x1x7 schoolbook, rank 14."""
    rank = 14
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(7)]
    W = [[0]*rank for _ in range(14)]
    col = 0
    for i in range(2):
        for k in range(7):
            U[i][col]=1; V[k][col]=1; W[k*2+i][col]=1; col+=1
    return U, V, W

def _combine_227():
    """2x2x7: split 7 into 2+2+2+1, use Strassen for each 2x2x2, schoolbook for 2x2x1. Rank=25."""
    sU,sV,sW = _strassen_222()
    sU2,sV2,sW2 = _strassen_222()
    sU3,sV3,sW3 = _strassen_222()
    scU,scV,scW = _schoolbook_221()
    tr = 25
    U = [[0]*tr for _ in range(4)]
    V = [[0]*tr for _ in range(14)]
    W = [[0]*tr for _ in range(14)]
    for r in range(7):
        for i in range(4): U[i][r]=sU[i][r]
        for j in range(2):
            for k in range(2): V[j*2+k][r]=sV[j*2+k][r]
        for k in range(2):
            for i in range(2): W[k*2+i][r]=sW[k*2+i][r]
    for r in range(7):
        for i in range(4): U[i][7+r]=sU2[i][r]
        for j in range(2):
            for k in range(2): V[j*2+2+k][7+r]=sV2[j*2+k][r]
        for k in range(2):
            for i in range(2): W[(2+k)*2+i][7+r]=sW2[k*2+i][r]
    for r in range(7):
        for i in range(4): U[i][14+r]=sU3[i][r]
        for j in range(2):
            for k in range(2): V[j*2+4+k][14+r]=sV3[j*2+k][r]
        for k in range(2):
            for i in range(2): W[(4+k)*2+i][14+r]=sW3[k*2+i][r]
    for r in range(4):
        for i in range(4): U[i][21+r]=scU[i][r]
        for j in range(2): V[j*2+6][21+r]=scV[j][r]
        for i in range(2): W[6*2+i][21+r]=scW[i][r]
    return U, V, W

def _combine_full():
    """Full 2x3x7: split m=3 into 2+1. A1*B1 (2x2x7, rank 25) + A2*B2 (2x1x7, rank 14) = 39."""
    U227,V227,W227 = _combine_227()
    U217,V217,W217 = _schoolbook_217()
    tr = 39
    U = [[0]*tr for _ in range(6)]
    V = [[0]*tr for _ in range(21)]
    W = [[0]*tr for _ in range(14)]
    for r in range(25):
        for i in range(2):
            for j in range(2): U[i*3+j][r]=U227[i*2+j][r]
        for j in range(2):
            for k in range(7): V[j*7+k][r]=V227[j*2+k][r]
        for k in range(7):
            for i in range(2): W[k*2+i][r]=W227[k*2+i][r]
    for r in range(14):
        for i in range(2): U[i*3+2][25+r]=U217[i][r]
        for k in range(7): V[14+k][25+r]=V217[k][r]
        for k in range(7):
            for i in range(2): W[k*2+i][25+r]=W217[k*2+i][r]
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] for r in range(rank))
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    result = _combine_full()
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    rank = n*m*p
    U=[[0]*rank for _ in range(n*m)]; V=[[0]*rank for _ in range(m*p)]; W=[[0]*rank for _ in range(p*n)]
    col=0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col]=1; V[j*p+k][col]=1; W[k*n+i][col]=1; col+=1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
