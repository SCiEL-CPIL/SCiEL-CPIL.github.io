"""Matrix multiplication 2,3,5 — rank-28 decomposition via two Strassen 2x2x2 blocks."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication using rank-28 ansatz.
    Two disjoint Strassen 2x2x2 blocks + 14 schoolbook terms.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    # Block A: Strassen for i∈{0,1}, j∈{0,1}, k∈{0,1}
    U[0][c]=1; U[4][c]=1; V[0][c]=1; V[6][c]=1; W[0][c]=1; W[3][c]=1; c+=1
    U[3][c]=1; U[4][c]=1; V[0][c]=1; W[1][c]=1; W[3][c]=-1; c+=1
    U[0][c]=1; V[1][c]=1; V[6][c]=-1; W[2][c]=1; W[3][c]=1; c+=1
    U[4][c]=1; V[5][c]=1; V[0][c]=-1; W[0][c]=1; W[1][c]=1; c+=1
    U[0][c]=1; U[1][c]=1; V[6][c]=1; W[0][c]=-1; W[2][c]=1; c+=1
    U[3][c]=1; U[0][c]=-1; V[0][c]=1; V[1][c]=1; W[3][c]=1; c+=1
    U[1][c]=1; U[4][c]=-1; V[5][c]=1; V[6][c]=1; W[0][c]=1; c+=1
    
    # Block B: Strassen for i∈{0,1}, j∈{0,1}, k∈{3,4}
    U[0][c]=1; U[4][c]=1; V[3][c]=1; V[9][c]=1; W[6][c]=1; W[9][c]=1; c+=1
    U[3][c]=1; U[4][c]=1; V[3][c]=1; W[7][c]=1; W[9][c]=-1; c+=1
    U[0][c]=1; V[4][c]=1; V[9][c]=-1; W[8][c]=1; W[9][c]=1; c+=1
    U[4][c]=1; V[8][c]=1; V[3][c]=-1; W[6][c]=1; W[7][c]=1; c+=1
    U[0][c]=1; U[1][c]=1; V[9][c]=1; W[6][c]=-1; W[8][c]=1; c+=1
    U[3][c]=1; U[0][c]=-1; V[3][c]=1; V[4][c]=1; W[9][c]=1; c+=1
    U[1][c]=1; U[4][c]=-1; V[8][c]=1; V[9][c]=1; W[6][c]=1; c+=1
    
    # 14 schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0,1) and k in (0,1): continue
                if j in (0,1) and k in (3,4): continue
                U[i*m+j][c]=1; V[j*p+k][c]=1; W[k*n+i][c]=1; c+=1
    
    assert c == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
